from pathlib import Path
from typing import Iterable, Sequence, cast

import numpy as np
import torch
from nemo.collections.nlp.parts.nlp_overrides import (
    GradScaler,
    MegatronHalfPrecisionPlugin,
    NLPDDPStrategy,
    PipelineMixedPrecisionPlugin,
)
from nemo.utils import get_rank, logging
from nemo.utils.exp_manager import StatelessTimer, exp_manager
from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.timer import Timer
from pytorch_lightning.trainer.connectors.checkpoint_connector import (
    CheckpointConnector,
)
from rich.console import Console
from rich.table import Table

from trlx.data.configs import TRLConfig
from trlx.data.ilql_types import ILQLBatch, ILQLElement, flatten_dataclass
from trlx.models.modeling_ilql import ILQLConfig
from trlx.models.modeling_llama_nemo import LlamaGPT
from trlx.models.modeling_nemo_ilql import ILQLGPT
from trlx.pipeline.offline_pipeline import (
    ILQLRolloutStorage,
    ilql_collate_fn,
    tokenize_dialogue,
)
from trlx.trainer import register_trainer

from . import BaseRLTrainer


def megatron_trainer(cfg):
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
            plugins.append(MegatronHalfPrecisionPlugin(precision=cfg.trainer.precision, device="cuda", scaler=scaler))
        else:
            plugins.append(PipelineMixedPrecisionPlugin(precision=cfg.trainer.precision, device="cuda", scaler=scaler))

    trainer = Trainer(plugins=plugins, strategy=strategy, **cfg.trainer)
    try:
        exp_manager(trainer, cfg.exp_manager)
    except FileNotFoundError:
        print(f"exp_manager failed to find git-rev, continuing anyway, {FileNotFoundError}")
    # update resume from checkpoint found by exp_manager
    if cfg.model.resume_from_checkpoint is not None:
        resume_from_checkpoint = cfg.model.resume_from_checkpoint
    else:
        resume_from_checkpoint = trainer._checkpoint_connector.resume_from_checkpoint_fit_path

    logging.info(f"Resuming training from checkpoint: {resume_from_checkpoint}")

    trainer._checkpoint_connector = CheckpointConnector(trainer, resume_from_checkpoint=resume_from_checkpoint)
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


class ShuffledCyclicSequence:
    def __init__(self, new_length: int, data: Sequence, seed: int):
        self.data = data
        self.new_length = new_length

        rng = torch.Generator().manual_seed(seed)
        self.perm = torch.randperm(new_length, generator=rng, device="cpu")

    def __len__(self):
        return self.new_length

    def __getitem__(self, idx):
        permuted_idx = self.perm[idx].item()
        return self.data[permuted_idx % len(self.data)]


@register_trainer
class NeMoILQLTrainer(BaseRLTrainer):
    store: ILQLRolloutStorage

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

        if not isinstance(config.method, ILQLConfig):
            raise ValueError("config.method must be ILQLConfig")

        self.ilql_config: ILQLConfig = cast(ILQLConfig, config.method)
        if isinstance(megatron_cfg, str):
            cfg_path = Path(__file__).parent.parent.parent / "configs" / "nemo_configs" / megatron_cfg
            logging.info(f"Loading NeMo config from {cfg_path=}")
            megatron_cfg = OmegaConf.load(cfg_path)

        elif megatron_cfg is None:
            raise ValueError("megatron_cfg must be a path or a config")

        self.trainer = megatron_trainer(megatron_cfg)
        self.model = ILQLGPT(
            ilql_config=self.ilql_config,
            metric_fn=self.metric_fn,
            cfg=megatron_cfg.model,
            trainer=self.trainer,
            stop_sequences=stop_sequences,
        )

        if pretrained_model is not None:
            self.model.load_from_pretrained(pretrained_model)

        self.batch_size = megatron_cfg.model.global_batch_size
        self.tokenizer = self.model.tokenizer.tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = megatron_cfg.model.encoder_seq_length

        self.tokenizer.truncation_side = config.tokenizer.truncation_side

    def learn(self):
        def collate_fn(elems: Iterable[ILQLElement]):
            batch = ilql_collate_fn(elems)
            return flatten_dataclass(ILQLBatch)(batch)

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

            max_new_tokens = self.ilql_config.gen_kwargs.get("max_new_tokens", 64)

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

        eval_dataset = ShuffledCyclicSequence(eval_samples, self.eval_pipeline, self.config.train.seed)

        self.model.set_valid_dataset(eval_dataset, collate_fn=eval_collate)

        torch.set_float32_matmul_precision("medium")
        self.trainer.fit(self.model)

    def make_experience(self, samples, rewards, max_length=2048):
        """
        Tokenizes samples and shapes rewards into proper tensors and then inserts the resulting dataset into the trainer
        """
        logging.info("Collecting rollouts")

        if self.tokenizer:
            samples = [tokenize_dialogue(s, self.tokenizer, max_length) for s in samples]

        all_input_ids = []
        all_actions_ixs = []
        all_states_ixs = []
        all_dones = []
        for sample in samples:
            length = 0
            all_input_ids.append(torch.tensor(sum(sample, [])))
            isoutput = False
            actions_ixs = []
            for phrase in sample:
                if isoutput:
                    actions_ixs.append(torch.arange(length - 1, length + len(phrase) - 1))

                length += len(phrase)
                isoutput = not isoutput

            states_ixs = torch.hstack((*actions_ixs, torch.tensor(length - 1)))
            all_dones.append(torch.tensor([1] * (len(states_ixs) - 1) + [0], dtype=int))
            all_actions_ixs.append(torch.hstack(actions_ixs))
            all_states_ixs.append(states_ixs)

        if get_rank.is_global_rank_zero():
            logging.info("Logging sample example")
            prompt = self.tokenizer.decode(all_input_ids[0][: all_states_ixs[0][1]])
            response = self.tokenizer.decode(all_input_ids[0][all_states_ixs[0][1] :])
            columns = ["Prompt", "Response", "Reward"]
            table = Table(*columns, title="Sample Example", show_lines=True)
            table.add_row(prompt, response, str(rewards[0]))
            Console().print(table)

        sample_lengths = np.array(list(map(len, all_input_ids)))
        output_lengths = np.array(list(map(len, all_actions_ixs)))
        prompt_lengths = sample_lengths - output_lengths
        returns = torch.tensor(rewards, dtype=float)

        if get_rank.is_global_rank_zero():
            logging.info("Logging experience string statistics")
            columns = ["Prompt Length", "Output Length", "Sample Length"]
            table = Table(*columns, title="Experience String Stats (mean ∈ \[min, max])", show_lines=True)
            row = []
            for lengths in [prompt_lengths, output_lengths, sample_lengths]:
                row.append(f"{lengths.mean():.2f} ∈ [{min(lengths)}, {max(lengths)}]")
            table.add_row(*row)
            Console().print(table)

        returns = (returns - returns.mean()) / (returns.std() + 1e-30)
        rewards = [torch.zeros(len(x)) for x in all_actions_ixs]
        for rs, ret in zip(rewards, returns):
            rs[-1] = ret

        attention_mask = [torch.ones(len(x), dtype=int) for x in all_input_ids]

        self.store = ILQLRolloutStorage(
            all_input_ids,
            attention_mask,
            rewards,
            all_states_ixs,
            all_actions_ixs,
            all_dones,
        )
