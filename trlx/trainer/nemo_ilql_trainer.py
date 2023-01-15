import os
from typing import Iterable, Sequence, Union, cast

import torch
import torch.nn.functional as F
from apex.transformer import parallel_state
from nemo.collections.nlp.modules.common.megatron.megatron_init import (
    fake_initialize_model_parallel,
)
from nemo.collections.nlp.modules.common.text_generation_utils import (
    get_default_length_params,
    get_default_sampling_params,
)
from nemo.collections.nlp.parts.nlp_overrides import (
    GradScaler,
    MegatronHalfPrecisionPlugin,
    NLPDDPStrategy,
    NLPSaveRestoreConnector,
    PipelineMixedPrecisionPlugin,
)
from nemo.utils import logging
from nemo.utils.app_state import AppState
from nemo.utils.exp_manager import StatelessTimer, exp_manager
from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.timer import Timer
from pytorch_lightning.trainer.connectors.checkpoint_connector import (
    CheckpointConnector,
)
from torch.nn.utils.rnn import pad_sequence

from trlx.data.configs import TRLConfig
from trlx.data.ilql_types import ILQLBatch, ILQLElement, flatten_dataclass
from trlx.pipeline.offline_pipeline import ILQLRolloutStorage
from trlx.trainer import register_trainer
from trlx.trainer.nemo.gpt import ILQLGPT
from trlx.trainer.nn.ilql_models import ILQLConfig

from . import BaseRLTrainer


def train_megatron(ilql_config, cfg, pretrained_model=None):
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

    model = ILQLGPT(ilql_config=ilql_config, cfg=cfg.model, trainer=trainer)
    if pretrained_model is not None:
        model.load_from_pretrained(pretrained_model)
    print("model initialized")
    return trainer, model


@register_trainer
class NemoILQLTrainer(BaseRLTrainer):
    store: ILQLRolloutStorage

    def __init__(
        self,
        config: TRLConfig,
        logit_mask=None,
        metric_fn=None,
        train_mode=True,
    ):
        super().__init__(config, train_mode)
        self.logit_mask = logit_mask
        self.metric_fn = metric_fn
        self.reward_fn = None

        if not isinstance(config.method, ILQLConfig):
            raise ValueError("config.method must be ILQLConfig")

        self.ilql: ILQLConfig = cast(ILQLConfig, config.method)
        megatron_cfg = OmegaConf.load("/mnt/nvme/home/uwu/megatron_20b.yaml")
        pretrained = "/mnt/nvme/home/uwu/nemo-megatron-gpt-20B/"
        self.trainer, self.model = train_megatron(self.ilql, megatron_cfg, pretrained)
        self.batch_size = megatron_cfg.model.global_batch_size
        self.tokenizer = self.model.tokenizer.tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = megatron_cfg.model.encoder_seq_length

    def tokenize(self, texts: Union[Sequence[str], Sequence[torch.LongTensor]]):
        if isinstance(texts[0], torch.LongTensor):
            return texts

        tokenized = self.tokenizer(
            [self.tokenizer.bos_token + x + self.tokenizer.eos_token for x in texts],
            max_length=self.max_length,
            truncation=True,
        )
        input_ids = list(map(torch.as_tensor, tokenized.input_ids))
        return input_ids

    def learn(self):
        # self.model.cfg.sequence_parallel = False
        # gen = self.model.generate(["hello world"] * 16, length_params=None)
        # print(f"{gen=}")
        # self.model.cfg.sequence_parallel = True
        def collate_fn(elems: Iterable[ILQLElement]):
            batch = ILQLBatch(
                pad_sequence(
                    [x.input_ids for x in elems], batch_first=True, padding_value=0
                ),
                pad_sequence(
                    [x.attention_mask for x in elems], batch_first=True, padding_value=0
                ),
                pad_sequence(
                    [x.rewards for x in elems], batch_first=True, padding_value=0.0
                ),
                pad_sequence(
                    [x.states_ixs for x in elems], batch_first=True, padding_value=0
                ),
                pad_sequence(
                    [x.actions_ixs for x in elems], batch_first=True, padding_value=0
                ),
                pad_sequence(
                    [x.dones for x in elems], batch_first=True, padding_value=0
                ),
            )
            return flatten_dataclass(ILQLBatch)(batch)

        # train_dataloader = self.model.build_data_loader(self.store, collate_fn=collate_fn)
        # print(f"{len(train_dataloader)=}")

        # train_dataloader = map(flatten_dataclass(ILQLBatch), train_dataloader)
        # # def handler(signum, frame):
        # #     raise Exception("end of time")

        self.model.set_train_dataset(self.store, collate_fn=collate_fn)
        torch.backends.cuda.matmul.allow_tf32 = True
        self.trainer.fit(self.model)

        # gen = self.model.generate(["hello world"] * 2)
        # print(f"{gen=}")
        # gen = self.model.generate(["hello world"] * 2)
        # print(f"{gen=}")
        # gen = self.model.generate(["hello world"] * 2)
        # print(f"{gen=}")

        # gen = self.model.generate(
        #     ["hello world"],
        #     dict(max_length=200, min_length=10),
        #     sampling_params,
        # )
