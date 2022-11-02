import importlib
import os
from abc import abstractmethod
from time import time
from typing import Any, Dict, Iterable, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
import sys

import deepspeed
import megatron  # type: ignore
from megatron.utils import Timers, get_total_params
from megatron import print_rank_0, mpu

from trlx.model.nn.ilql_models import GPTNeoXWithValueHeads
from trlx.data.ilql_types import ILQLBatch
from transformers import AutoTokenizer

import wandb
from trlx.data.configs import TRLConfig
from trlx.model import BaseRLModel, register_model

if importlib.util.find_spec("rich") is not None:
    from tqdm.rich import tqdm
else:
    from tqdm import tqdm


@register_model
class NeoXRLModel(BaseRLModel):
    """
    RL Model that uses NeoX for training
    """

    def __init__(
        self, config: TRLConfig, neox_args: megatron.NeoXArgs, train_mode=True
    ):
        super().__init__(config, train_mode)

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
            model.set_batch_fn(None)

            if neox_args.is_pipe_parallel:
                model.set_has_attention_mask(True)
        else:
            raise ValueError("Must be using deepspeed to run neox")

        # Retrieves model equipped for ppo, ilql, etc
        self.neox_args = neox_args
        self.tokenizer = neox_args.tokenizer.tokenizer
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def tokenize(self, text: Union[Sequence[str], Sequence[torch.LongTensor]]):
        """
        Tokenize a batch of text after adding bos token to each of the samples
        """
        if isinstance(text[0], torch.LongTensor):
            return text

        encoded = self.tokenizer.encode_batch(
            text,
            # truncation=True,
            # max_length=self.config.train.seq_length,
            # return_tensors="pt",
        )

        return [e.ids for e in encoded]

    def learn(self):
        """
        Samples batches from `self.store`, updates model and periodically evaluates it on `self.eval_dataloader`
        """

        self.model.train()

        train_dataloader = self.store.create_loader(
            self.neox_args.train_micro_batch_size_per_gpu
        )
        eval_dataloader = self.eval_pipeline.create_loader(self.config.train.batch_size)

        from megatron.utils import get_ltor_masks_and_position_ids

        def broadcast_dataclass(obj: object):
            d = {
                k: mpu.broadcast_data([0], [v], v.dtype)[0]
                for k, v in obj.__dict__.items()
            }
            return type(obj)(**d)

        def preprocess(b: ILQLBatch):
            b = broadcast_dataclass(b)
            # print(b)
            tokens = b.input_ids  # [:, :-1]
            attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
                data=tokens,
                eod_token=self.neox_args.tokenizer.eod,
                eod_mask_loss=self.neox_args.eod_mask_loss,
            )
            # return (tokens, position_ids, attention_mask), (
            #     b.input_ids[:, 1:],
            #     loss_mask,
            # )

            return (tokens, position_ids, attention_mask), b

        it: Iterable[ILQLBatch] = iter(train_dataloader)
        flattened = map(preprocess, it)
        for i in range(len(train_dataloader) * self.config.train.epochs):
            print_rank_0(self.model.train_batch(flattened))
        from megatron.training import train
        import dataclasses
