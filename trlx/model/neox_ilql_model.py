import importlib
import os
from abc import abstractmethod
from functools import partial
from dataclasses import astuple
from time import time
from typing import Any, Dict, Iterable, Sequence, Tuple, Union, TypeVar

import torch
import torch.nn.functional as F
import sys

import deepspeed
import megatron  # type: ignore
from megatron.neox_arguments import NeoXArgs
from megatron.utils import Timers, get_total_params
from megatron import print_rank_0, mpu

from trlx.model.nn.ilql_models import ILQLConfig, ILQLHeads
from trlx.model.nn.neox_ilql_models import GPTNeoXWithValueHeads, preprocess_batch
from trlx.data.ilql_types import ILQLBatch
from transformers import AutoTokenizer

import wandb
from trlx.data.configs import TRLConfig
from trlx.model import BaseRLModel, register_model

if importlib.util.find_spec("rich") is not None:
    from tqdm.rich import tqdm
else:
    from tqdm import tqdm

T = TypeVar("T")


class NeoXTokenizerWrapper(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def decode(self, token_ids: torch.LongTensor):
        return self.tokenizer.decode(token_ids.tolist())

    def encode(self, text: str):
        return self(text)

    def __call__(self, text: str) -> torch.Tensor:
        return self.tokenizer.encode(text).ids


@register_model
class NeoXILQLModel(BaseRLModel):
    """
    RL Model that uses NeoX for training
    """

    def __init__(
        self,
        config: TRLConfig,
        neox_args: megatron.NeoXArgs = None,
        logit_mask=None,
        metric_fn=None,
    ):

        super().__init__(config)

        if neox_args is None:
            neox_args = NeoXArgs.consume_neox_args()
            neox_args.configure_distributed_args()
            neox_args.build_tokenizer()

        neox_args.is_pipe_parallel = True
        neox_args.iteration = 0
        megatron.utils.init_wandb(neox_args=neox_args)

        self.timers = Timers(
            use_wandb=neox_args.use_wandb,
            tensorboard_writer=neox_args.tensorboard_writer,
        )

        megatron.initialize.initialize_megatron(neox_args)

        model = GPTNeoXWithValueHeads(config=neox_args, ilql_config=config.method)

        from megatron.training import get_optimizer, get_learning_rate_scheduler

        optimizer, param_groups = get_optimizer(model, neox_args)
        lr_scheduler = get_learning_rate_scheduler(
            optimizer=optimizer, neox_args=neox_args
        )

        if neox_args.deepspeed:
            print_rank_0("DeepSpeed is enabled.")
            if neox_args.no_load_optim:
                assert optimizer is None
                _model_params = None
                _lr_scheduler = None
            else:
                _model_params = param_groups if optimizer is None else None
                _lr_scheduler = lr_scheduler

            model, optimizer, _, lr_scheduler = deepspeed.initialize(
                model=model,
                optimizer=optimizer,
                args=neox_args,
                lr_scheduler=_lr_scheduler,
                dist_init_required=True,
                model_parameters=_model_params,
                config_params=neox_args.deepspeed_config,
                mpu=mpu if not neox_args.is_pipe_parallel else None,
            )
            model.total_params = get_total_params(model.module)
            print_rank_0(f' > total params: {"{:,}".format(model.total_params)}')
            model.set_batch_fn(partial(preprocess_batch, neox_args))

            if neox_args.is_pipe_parallel:
                model.set_has_attention_mask(True)
        else:
            raise ValueError("Must be using deepspeed to run neox")

        self.neox_args = neox_args
        self.tokenizer = NeoXTokenizerWrapper(neox_args.tokenizer.tokenizer)
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def tokenize(self, text: Union[Sequence[str], Sequence[torch.LongTensor]]):
        """
        Tokenize a batch of text after adding bos token to each of the samples
        """
        if isinstance(text[0], torch.LongTensor):
            return text

        return [self.tokenizer(e) for e in text]

    def learn(self):
        """
        Samples batches from `self.store`, updates model and periodically evaluates it on `self.eval_dataloader`
        """

        self.model.train()

        train_dataloader = self.store.create_loader(
            self.neox_args.train_micro_batch_size_per_gpu
        )
        eval_dataloader = self.eval_pipeline.create_loader(self.config.train.batch_size)

        for i in range(self.config.train.epochs):
            it: Iterable[ILQLBatch] = iter(
                deepspeed.utils.RepeatingLoader(train_dataloader)
            )
            for batch_idx in range(len(train_dataloader)):
                print_rank_0(self.model.train_batch(data_iter=it))
                if batch_idx % self.config.method.steps_for_target_q_sync == 0:

                    def sync_target_q(m):
                        if isinstance(m, ILQLHeads):
                            m.sync_target_q_heads()

                    self.model.apply(sync_target_q)
