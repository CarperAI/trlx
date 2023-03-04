from pathlib import Path
from typing import List, cast

import torch
import transformers
from nemo.utils import logging
from omegaconf.omegaconf import OmegaConf

from trlx.data.configs import TRLConfig
from trlx.models.modeling_nemo_sft import SFTGPT
from trlx.trainer import BaseRLTrainer, register_trainer
from trlx.trainer.accelerate_sft_trainer import SFTConfig
from trlx.trainer.nemo_ilql_trainer import ShuffledCyclicSequence, megatron_trainer


@register_trainer
class NeMoSFTTrainer(BaseRLTrainer):
    def __init__(
        self,
        config: TRLConfig,
        reward_fn=None,
        logit_mask=None,
        metric_fn=None,
        stop_sequences=None,
        train_mode=True,
        megatron_cfg=None,
        pretrained_model=None,
    ):
        super().__init__(config, train_mode)
        self.logit_mask = logit_mask
        self.metric_fn = metric_fn
        self.reward_fn = None

        if not isinstance(config.method, SFTConfig):
            raise ValueError("config.method must be SFTConfig")

        self.sft_config: SFTConfig = cast(SFTConfig, config.method)
        if isinstance(megatron_cfg, str):
            cfg_path = Path(__file__).parent.parent.parent / "configs" / "nemo_configs" / megatron_cfg
            logging.info(f"Loading NeMo config from {cfg_path=}")
            megatron_cfg = OmegaConf.load(cfg_path)

        elif megatron_cfg is None:
            raise ValueError("megatron_cfg must be a path or a config")

        self.trainer = megatron_trainer(megatron_cfg)
        self.model = SFTGPT(
            sft_config=self.sft_config,
            cfg=megatron_cfg.model,
            trainer=self.trainer,
            metric_fn=self.metric_fn,
        )

        if pretrained_model is not None:
            self.model.load_from_pretrained(pretrained_model)

        self.batch_size = megatron_cfg.model.global_batch_size
        self.tokenizer = self.model.tokenizer.tokenizer
        self.tokenizer.truncation_side = config.tokenizer.truncation_side
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = megatron_cfg.model.encoder_seq_length

        if stop_sequences is not None and len(stop_sequences) > 0:
            logging.warning(f"Ignoring stop_sequences {stop_sequences=}")

    def learn(self):
        def collate_fn(elems: List[transformers.BatchEncoding]):
            input_ids = self.tokenizer.pad(
                elems,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )["input_ids"]
            return [input_ids]

        train_samples = self.model.cfg.global_batch_size * self.trainer.max_steps
        train_dataset = ShuffledCyclicSequence(train_samples, self.store, self.config.train.seed)
        self.model.set_train_dataset(train_dataset, collate_fn=collate_fn)

        def add_bos_if_not_present(x):
            if len(x) == 0:
                return [self.tokenizer.bos_token_id]
            elif x[0] != self.tokenizer.bos_token_id:
                return [self.tokenizer.bos_token_id] + x
            else:
                return x

        def eval_collate(elems):
            context_tokens = [e["input_ids"] for e in elems]
            context_tokens = [add_bos_if_not_present(x) for x in context_tokens]

            max_new_tokens = self.sft_config.gen_kwargs.get("max_new_tokens", 64)

            context_lengths = [len(x) for x in context_tokens]
            max_context = max(context_lengths)

            pad_id = self.tokenizer.eos_token_id
            padded = [x + [pad_id] * (max_context + max_new_tokens - len(x)) for x in context_tokens]

            return [
                torch.as_tensor(padded, device="cpu"),
                torch.as_tensor(context_lengths, device="cpu"),
            ]

        max_train_steps = self.trainer.max_steps
        eval_iters = (max_train_steps // self.trainer.val_check_interval + 1) * self.trainer.limit_val_batches
        eval_samples = eval_iters * self.model.cfg.global_batch_size

        eval_dataset = ShuffledCyclicSequence(
            new_length=eval_samples,
            data=self.eval_pipeline,
            seed=self.config.train.seed,
        )

        self.model.set_valid_dataset(eval_dataset, collate_fn=eval_collate)

        torch.set_float32_matmul_precision("medium")
        self.trainer.fit(self.model)
