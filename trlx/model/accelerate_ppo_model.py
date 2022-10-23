import numpy as np
import torch

from trlx.data.configs import TRLConfig
from trlx.model import register_model
from trlx.model.accelerate_base_model import AccelerateRLModel
from trlx.model.nn.ppo_models import GPTHydraHeadWithValueModel
from trlx.pipeline.ppo_pipeline import PPORolloutStorage
from trlx.utils.modeling import clip_by_value, logprobs_from_logits, whiten


class AdaptiveKLController:
    def __init__(self, init_kl_coef, target, horizon):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps):
        target = self.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current, n_steps):
        pass


@register_model
class AcceleratePPOModel(AccelerateRLModel):
    def __init__(self, config):
        super().__init__(config)

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
        return GPTHydraHeadWithValueModel(
            self.config.model.model_path, self.config.model.num_layers_unfrozen
        )

    def ref_generate(self, input_ids, attention_mask=None, **kwargs):
        """Wraps hf's `generate` adding some specific method's defaults"""
        input_ids = input_ids.to(self.accelerator.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.accelerator.device)

        kwargs = dict(self.generate_kwargs, **kwargs)

        with torch.no_grad():
            return self.model.ref_model.generate(
                input_ids=input_ids, attention_mask=attention_mask, **kwargs
            )

    def loss(self, batch):
        query_tensors = batch.query_tensors.to(self.accelerator.device)
        response_tensors = batch.response_tensors.to(self.accelerator.device)
        all_logprobs = batch.logprobs.to(self.accelerator.device)
        all_values = batch.values.to(self.accelerator.device)
        all_rewards = batch.rewards.to(self.accelerator.device)

        lastgaelam = 0
        advantages_reversed = []
        gen_len = response_tensors.shape[1]
        for t in reversed(range(gen_len)):
            nextvalues = all_values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = (
                all_rewards[:, t]
                + self.config.method.gamma * nextvalues
                - all_values[:, t]
            )
            lastgaelam = (
                delta + self.config.method.gamma * self.config.method.lam * lastgaelam
            )
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)

        returns = advantages + all_values
        advantages = whiten(advantages)
        advantages = advantages.detach()

        all_tokens = torch.cat((query_tensors, response_tensors), dim=1)
        attention_mask = (
            all_tokens.not_equal(self.tokenizer.pad_token_id)
            .long()
            .to(all_tokens.device)
        )

        # for a proper positional encoding in case of left padding
        position_ids = attention_mask.cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask.eq(0), 0)

        logits, _, vpred = self.model(
            all_tokens, attention_mask, position_ids=position_ids
        )
        logprob = logprobs_from_logits(logits[:, :-1, :], all_tokens[:, 1:])

        # only the generation part of the values/logprobs is needed
        logprob, vpred = logprob[:, -gen_len:], vpred[:, -gen_len:]

        vpredclipped = clip_by_value(
            vpred,
            all_values - self.config.method.cliprange_value,
            all_values + self.config.method.cliprange_value,
        )

        mask = attention_mask[:, -gen_len:]

        vf_losses1 = (vpred - returns) ** 2
        vf_losses2 = (vpredclipped - returns) ** 2
        vf_loss = 0.5 * torch.sum(torch.max(vf_losses1, vf_losses2) * mask) / mask.sum()

        kl = logprob - all_logprobs
        # Record mean_kl for kl coef adjustment
        self.mean_kl = torch.mean(torch.sum(kl, dim=-1)).item()
        ratio = torch.exp(kl)

        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(
            ratio,
            1.0 - self.config.method.cliprange,
            1.0 + self.config.method.cliprange,
        )

        pg_loss = torch.sum(torch.max(pg_losses, pg_losses2) * mask) / mask.sum()
        loss = pg_loss + self.config.method.vf_coef * vf_loss

        stats = {
            "loss": loss,
            "pg_loss": pg_loss,
            "vf_loss": vf_loss,
        }

        return loss, stats

    def post_epoch_callback(self):
        self.store.clear_history()
        self.orch.make_experience(
            self.config.method.num_rollouts, self.iter_count
        )  # Collect more rollouts for training

    def post_backward_callback(self):
        # Update kl_coefficient
        self.kl_ctl.update(self.mean_kl, self.config.train.batch_size)

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
