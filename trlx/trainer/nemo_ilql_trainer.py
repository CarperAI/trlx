from pathlib import Path
from typing import Iterable, Sequence, Union, cast

import torch
from nemo.collections.nlp.modules.common.text_generation_strategy import pad_batch
from nemo.collections.nlp.parts.nlp_overrides import (
    GradScaler,
    MegatronHalfPrecisionPlugin,
    NLPDDPStrategy,
    PipelineMixedPrecisionPlugin,
)
from nemo.utils import logging
from nemo.utils.exp_manager import StatelessTimer, exp_manager
from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.timer import Timer
from pytorch_lightning.trainer.connectors.checkpoint_connector import (
    CheckpointConnector,
)
from transformers import DataCollatorWithPadding

from trlx.data.configs import TRLConfig
from trlx.data.ilql_types import ILQLBatch, ILQLElement, flatten_dataclass
from trlx.pipeline.offline_pipeline import ILQLRolloutStorage, ilql_collate_fn
from trlx.trainer import register_trainer
from trlx.trainer.nemo.gpt import ILQLGPT
from trlx.trainer.nn.ilql_models import ILQLConfig

from . import BaseRLTrainer


def megatron_trainer(cfg, pretrained_model=None):
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    megatron_amp_o2 = cfg.model.get("megatron_amp_O2", False)
    with_distributed_adam = cfg.model.optim.get("name") == "distributed_fused_adam"

    plugins = []
    strategy = NLPDDPStrategy(
        no_ddp_communication_hook=True,  # we don't use DDP for async grad allreduce
        gradient_as_bucket_view=cfg.model.gradient_as_bucket_view,
        find_unused_parameters=False,
    )
    if cfg.trainer.precision in [16, "bf16"]:
        scaler = None
        if cfg.trainer.precision == 16:
            scaler = GradScaler(
                init_scale=cfg.model.get("native_amp_init_scale", 2**32),
                growth_interval=cfg.model.get("native_amp_growth_interval", 1000),
                hysteresis=cfg.model.get("hysteresis", 2),
            )
        if megatron_amp_o2 and not with_distributed_adam:
            plugins.append(
                MegatronHalfPrecisionPlugin(
                    precision=cfg.trainer.precision, device="cuda", scaler=scaler
                )
            )
        else:
            plugins.append(
                PipelineMixedPrecisionPlugin(
                    precision=cfg.trainer.precision, device="cuda", scaler=scaler
                )
            )

    trainer = Trainer(plugins=plugins, strategy=strategy, **cfg.trainer)
    try:
        exp_manager(trainer, cfg.exp_manager)
    except FileNotFoundError:
        print(
            f"exp_manager failed to find git-rev, continuing anyway, {FileNotFoundError}"
        )
    # update resume from checkpoint found by exp_manager
    if cfg.model.resume_from_checkpoint is not None:
        resume_from_checkpoint = cfg.model.resume_from_checkpoint
    else:
        resume_from_checkpoint = (
            trainer._checkpoint_connector.resume_from_checkpoint_fit_path
        )

    logging.info(f"Resuming training from checkpoint: {resume_from_checkpoint}")

    trainer._checkpoint_connector = CheckpointConnector(
        trainer, resume_from_checkpoint=resume_from_checkpoint
    )
    # Override timer callback to a stateless one
    for idx, callback in enumerate(trainer.callbacks):
        if isinstance(callback, Timer):
            trainer.callbacks[idx] = StatelessTimer(
                cfg.trainer.max_time,
            )
    # hydra interpolation does not work here as the interpolation key is lost when PTL saves hparams
    with open_dict(cfg):
        cfg.model.precision = cfg.trainer.precision

    return trainer


@register_trainer
class NeMoILQLTrainer(BaseRLTrainer):
    store: ILQLRolloutStorage

    def __init__(
        self,
        config: TRLConfig,
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

        if not isinstance(config.method, ILQLConfig):
            raise ValueError("config.method must be ILQLConfig")

        self.ilql_config: ILQLConfig = cast(ILQLConfig, config.method)
        if isinstance(megatron_cfg, str):
            cfg_path = (
                Path(__file__).parent.parent.parent
                / "configs"
                / "nemo_configs"
                / megatron_cfg
            )
            logging.info(f"Loading NeMo config from {cfg_path=}")
            megatron_cfg = OmegaConf.load(cfg_path)

        elif megatron_cfg is None:
            raise ValueError("megatron_cfg must be a path or a config")

        self.trainer = megatron_trainer(megatron_cfg, pretrained_model)
        self.model = ILQLGPT(
            ilql_config=self.ilql_config,
            metric_fn=self.metric_fn,
            cfg=megatron_cfg.model,
            trainer=self.trainer,
        )

        if pretrained_model is not None:
            self.model.load_from_pretrained(pretrained_model)

        self.batch_size = megatron_cfg.model.global_batch_size
        self.tokenizer = self.model.tokenizer.tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = megatron_cfg.model.encoder_seq_length

        self.tokenizer.truncation_side = config.tokenizer.truncation_side

        if stop_sequences is not None and len(stop_sequences) > 0:
            logging.warning(f"Ignoring stop_sequences {stop_sequences=}")

    def tokenize(self, texts: Union[Sequence[str], Sequence[torch.LongTensor]]):
        if isinstance(texts[0], torch.LongTensor):
            return texts

        tokenized = self.tokenizer(
            [self.tokenizer.bos_token + x + self.tokenizer.eos_token for x in texts],
            max_length=self.max_length,
            truncation=True,
            # NOTE: We manually add special tokens (bos) above so we set this False
            # to avoid models that automatically add special tokens (e.g. OPT)
            # adding them twice more.
            add_special_tokens=False,
        )
        input_ids = list(map(torch.as_tensor, tokenized.input_ids))
        return input_ids

    def learn(self):
        def collate_fn(elems: Iterable[ILQLElement]):
            batch = ilql_collate_fn(elems)
            return flatten_dataclass(ILQLBatch)(batch)

        self.model.set_train_dataset(self.store, collate_fn=collate_fn)

        max_new_tokens = self.ilql_config.gen_kwargs.get("max_new_tokens", 64)

        def eval_collate(elems):
            context_tokens = [e["input_ids"] for e in elems]
            context_tokens, context_lengths = pad_batch(
                context_tokens, self.tokenizer.eos_token_id, max_new_tokens
            )
            return [torch.as_tensor(context_tokens), torch.as_tensor(context_lengths)]

        max_train_steps = self.trainer.max_steps
        eval_iters = (
            max_train_steps // self.trainer.val_check_interval + 1
        ) * self.trainer.limit_val_batches
        eval_samples = eval_iters * self.model.cfg.global_batch_size
        long_pipe = [
            self.eval_pipeline[i % len(self.eval_pipeline)] for i in range(eval_samples)
        ]

        self.model.set_valid_dataset(long_pipe, collate_fn=eval_collate)

        self.trainer.fit(self.model)
