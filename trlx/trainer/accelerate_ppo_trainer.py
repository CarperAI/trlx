import json
import os
import uuid
from typing import Tuple

import torch
from torchtyping import TensorType

from trlx.data.configs import TRLConfig
from trlx.data.ppo_types import PPORLBatch
from trlx.pipeline.ppo_pipeline import PPORolloutStorage
from trlx.trainer import register_trainer
from trlx.trainer.accelerate_base_trainer import AccelerateRLTrainer
from trlx.trainer.nn.ppo_models import (
    AdaptiveKLController,
    CausalLMHydraWithValueHead,
    FixedKLController,
)
from trlx.utils.modeling import logprobs_from_logits


@register_trainer
class AcceleratePPOTrainer(AccelerateRLTrainer):
    def __init__(self, config):
        super().__init__(config)

        if config.train.rollout_logging_dir is not None:
            self.log_rollouts = True
            self.setup_rollout_logging(config)
        else:
            self.log_rollouts = False

        self.store = PPORolloutStorage(self.tokenizer.pad_token_id)

        rollout_loader = self.store.create_loader(
            self.config.train.batch_size, shuffle=True
        )

        self.model, self.opt, self.scheduler, rollout_loader = self.accelerator.prepare(
            self.model, self.opt, self.scheduler, rollout_loader
        )

        self.store.clear_history()
        if config.method.target is not None:
            self.kl_ctl = AdaptiveKLController(
                config.method.init_kl_coef, config.method.target, config.method.horizon
            )
        else:
            self.kl_ctl = FixedKLController(config.method.init_kl_coef)

        self.generate_kwargs = dict(
            config.method.gen_kwargs,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

    def get_arch(self, config: TRLConfig):
        return CausalLMHydraWithValueHead(
            config.model.model_path, config.model.num_layers_unfrozen
        )

    def get_model_inputs(
        self,
        query_tensors: TensorType["batch_size", "query_size"],
        response_tensors: TensorType["batch_size", "response_size"],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = torch.cat((query_tensors, response_tensors), dim=1)[
            :, -self.max_length :
        ]
        attention_mask = (
            tokens.not_equal(self.tokenizer.pad_token_id).long().to(tokens.device)
        )
        # For a proper positional encoding in case of left padding
        position_ids = attention_mask.cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask.eq(0), 0)
        return tokens, attention_mask, position_ids

    def loss(self, batch: PPORLBatch):
        # Move `batch` data to `accelerator` device
        query_tensors = batch.query_tensors.to(self.accelerator.device)
        response_tensors = batch.response_tensors.to(self.accelerator.device)
        old_logprobs = batch.logprobs.to(self.accelerator.device)
        old_values = batch.values.to(self.accelerator.device)
        old_rewards = batch.rewards.to(self.accelerator.device)

        response_length = old_rewards.shape[1]

        advantages, returns = self.config.method.get_advantages_and_returns(
            old_values, old_rewards, response_length
        )

        tokens, attention_mask, position_ids = self.get_model_inputs(
            query_tensors, response_tensors
        )

        logits, *_, values_pred = self.model(
            tokens, attention_mask=attention_mask, position_ids=position_ids
        )
        values_pred = values_pred[:, :-1]
        logprobs = logprobs_from_logits(logits[:, :-1, :], tokens[:, 1:])
        attention_mask = attention_mask[:, :-1]

        # Only the response part of the values/logprobs is needed
        start = query_tensors.shape[1] - 1
        end = start + response_length
        logprobs, values_pred, mask = (
            logprobs[:, start:end],
            values_pred[:, start:end],
            attention_mask[:, start:end],
        )

        loss, stats = self.config.method.loss(
            logprobs=logprobs,
            values=values_pred,
            old_logprobs=old_logprobs,
            old_values=old_values,
            advantages=advantages,
            returns=returns,
            mask=mask,
        )
        self.approx_kl = stats["policy/approx_kl"]  # Update kl controller stats
        return loss, stats

    def setup_rollout_logging(self, config):
        # Make rollout logging dir for this run and store config
        exists = os.path.exists(config.train.rollout_logging_dir)
        isdir = os.path.isdir(config.train.rollout_logging_dir)
        assert exists and isdir

        self.run_id = f"run-{uuid.uuid4()}"
        self.rollout_logging_dir = os.path.join(
            config.train.rollout_logging_dir, self.run_id
        )
        os.mkdir(self.rollout_logging_dir)

        with open(os.path.join(self.rollout_logging_dir, "config.json"), "w") as f:
            f.write(json.dumps(config.to_dict(), indent=2))

    def post_epoch_callback(self):
        if self.log_rollouts:
            self.store.export_history(location=self.rollout_logging_dir)
        self.store.clear_history()
        self.orch.make_experience(
            self.config.method.num_rollouts, self.iter_count
        )  # Collect more rollouts for training

    def post_backward_callback(self):
        self.kl_ctl.update(self.approx_kl, n_steps=self.config.train.batch_size)

    def prepare_learning(self):
        eval_dataloader = self.eval_pipeline.create_loader(self.config.train.batch_size)

        train_dataloader = self.store.create_loader(
            self.config.train.batch_size, shuffle=True
        )

        self.train_dataloader, self.eval_dataloader = self.accelerator.prepare(
            train_dataloader, eval_dataloader
        )

        self.n_updates_per_batch = self.config.method.ppo_epochs
        self.total_steps = (
            self.config.train.epochs
            * self.n_updates_per_batch
            * len(self.train_dataloader)
        )
        self.total_steps = min(self.total_steps, self.config.train.total_steps)
