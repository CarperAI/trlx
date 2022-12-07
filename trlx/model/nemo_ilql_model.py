from typing import Iterable, Sequence, Union, cast

import torch
import torch.nn.functional as F

from trlx.model import register_model
from trlx.model.nn.ilql_models import ILQLConfig, CausalLMWithValueHeads
from trlx.data.ilql_types import ILQLBatch, flatten_dataclass
from trlx.data.configs import TRLConfig
from trlx.utils import to_device

from . import BaseRLModel

from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.timer import Timer
from pytorch_lightning.plugins.environments.torchelastic_environment import (
    TorchElasticEnvironment,
)
from pytorch_lightning.trainer.connectors.checkpoint_connector import (
    CheckpointConnector,
)

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import (
    MegatronGPTModel,
)
from nemo.collections.nlp.parts.nlp_overrides import (
    GradScaler,
    MegatronHalfPrecisionPlugin,
    NLPDDPStrategy,
    PipelineMixedPrecisionPlugin,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import StatelessTimer, exp_manager

from trlx.model.nemo.gpt import ILQLGPT


def train_megatron(ilql_config, cfg) -> None:
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

    if cfg.get("cluster_type", None) == "BCP":
        plugins.append(TorchElasticEnvironment())

    trainer = Trainer(plugins=plugins, strategy=strategy, **cfg.trainer)

    # exp_manager(trainer, cfg.exp_manager)

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

    return trainer, model


@register_model
class NeMoILQLModel(BaseRLModel):
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
        megatron_cfg = OmegaConf.load("/mnt/nvme/home/uwu/40b.yaml")
        self.trainer, self.model = train_megatron(self.ilql, megatron_cfg)
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

    def post_backward_callback(self):
        if self.iter_count % self.config.method.steps_for_target_q_sync == 0:
            self.accelerator.unwrap_model(self.model).sync_target_q_heads()

    def learn(self):
        train_dataloader = self.store.create_loader(self.config.train.batch_size)
        print(f"{len(train_dataloader)=}")
        eval_dataloader = self.eval_pipeline.create_loader(self.config.train.batch_size)
        train_dataloder = (ILQLBatch(**x) for x in train_dataloader)
        # print(next(iter(eval_dataloader)).keys())
        # eval_dataloader = (ILQLBatch(**x) for x in eval_dataloader)
        train_dataloader = map(flatten_dataclass(ILQLBatch), train_dataloader)
        # eval_dataloader = map(flatten_dataclass(ILQLBatch), eval_dataloader)

        self.trainer.fit(self.model, train_dataloader)
