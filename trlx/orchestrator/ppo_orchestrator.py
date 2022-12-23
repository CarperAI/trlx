from typing import Callable

import torch
from trlx.data.accelerate_base_datatypes import PromptBatch
from trlx.data.ppo_types import PPORLElement
from trlx.trainer import BaseRLTrainer
from trlx.orchestrator import Orchestrator, register_orchestrator
from trlx.pipeline import BasePipeline
from trlx.utils import Clock
from trlx.utils.modeling import logprobs_from_logits, RunningMoments

from time import time
import ray


@register_orchestrator
class PPOOrchestrator(Orchestrator):
    """
    Orchestrator that prepares data for PPO training: transforms samples from `pipeline` into `PPOBatch` and pushes them into trainer's `store`
    """

    def __init__(
        self,
        trainer: BaseRLTrainer,
        pipeline: BasePipeline,
        reward_fn: Callable,
        metric_fn: Callable = None,
        chunk_size: int = 512,
    ):
        self.pipeline = pipeline
        self.trainer = trainer
        self.chunk_size = chunk_size

        self.pipeline_loader = self.pipeline.create_loader(
            self.chunk_size, shuffle=True
        )
        self.pipeline_loader = self.trainer.accelerator.prepare(self.pipeline_loader)
        self.pipeline_iterator = iter(self.pipeline_loader)

        if not hasattr(self.trainer.model, "frozen_head"):
            self.ref_model = self.trainer.get_arch(self.trainer.config)

        self.trainer.orch = self
        self.trainer.reward_fn = reward_fn
        self.trainer.metric_fn = metric_fn

        self.running = RunningMoments()
        self.ref_mean = self.trainer.config.method.ref_mean
        self.ref_std = self.trainer.config.method.ref_std

    def score(self, samples):
        """
        Batched scoring function taking text and generating scalar
        """
        return self.trainer.reward_fn(samples)

    def make_experience(self, num_rollouts: int = 1024, iter_count: int = 0):
        """
        Takes `num_rollouts` prompts from `pipeline`, samples model, computes KL againts a reference model appends PPOElements to trainer's `store`
        """
        ppo_rl_elements = []
        stats = {}
        clock = Clock()
        while len(ppo_rl_elements) < num_rollouts:
            # Get next batch in prompt dataset and refresh if exhausted
            try:
                batch: PromptBatch = next(self.pipeline_iterator)
            except StopIteration:
                self.pipeline_iterator = iter(self.pipeline_loader)
                batch = next(self.pipeline_iterator)

            exp_generate_time = time()
            samples = self.trainer.generate(**batch)
            stats["time/exp_generate"] = time() - exp_generate_time

            query_tensors = batch.input_ids
            response_tensors = samples[:, query_tensors.shape[1] :]
            texts = self.trainer.tokenizer.batch_decode(
                samples, skip_special_tokens=True
            )
            exp_score_time = time()
            scores = torch.tensor(
                self.score(texts), device=samples.device, dtype=torch.float
            )
            stats["time/exp_score"] = time() - exp_score_time

            # store statistics of the initial rollout as reference
            if self.ref_mean is None:
                self.ref_mean, self.ref_std = scores.mean(), scores.std()
            all_scores_mean, all_scores_std = self.running.update(scores)
            stats["exp_scores/mean"] = all_scores_mean
            stats["exp_scores/std"] = all_scores_std
            stats["exp_scores/running_mean"] = self.running.mean
            stats["exp_scores/running_std"] = self.running.std

            if self.trainer.config.method.scale_reward == "running":
                scores /= self.running.std
            elif self.trainer.config.method.scale_reward == "ref":
                scores /= self.ref_std

            clip_reward = self.trainer.config.method.cliprange_reward
            if clip_reward:
                scores = torch.clip(scores, -clip_reward, clip_reward)

            # Precompute logprobs, values
            all_tokens, attention_mask, position_ids = self.trainer.get_model_inputs(
                query_tensors.to(response_tensors.device), response_tensors
            )
            with torch.no_grad():
                logits, *_, values = self.trainer.model(
                    all_tokens, attention_mask=attention_mask, position_ids=position_ids
                )
                # TODO(dahoas): When hydra model works need to also support generation on hydra head
                if hasattr(self.trainer.model, "frozen_head"):
                    ref_logits = self.trainer.model.forward_hydra(
                        all_tokens,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        return_dict=False,
                    )
                else:
                    ref_logits, _, *_ = self.ref_model(
                        all_tokens.cpu(),
                        attention_mask=attention_mask.cpu(),
                        position_ids=position_ids.cpu(),
                    )
                    ref_logits = ref_logits.to(self.trainer.accelerator.device)

            logprobs = logprobs_from_logits(logits[:, :-1, :], all_tokens[:, 1:])
            ref_logprobs = logprobs_from_logits(
                ref_logits[:, :-1, :], all_tokens[:, 1:]
            )

            n = samples.shape[0]
            values = values.cpu()[:, :-1]
            logprobs = logprobs.cpu()
            ref_logprobs = ref_logprobs.cpu()
            query_tensors = query_tensors.cpu()
            response_tensors = response_tensors.cpu()

            start = query_tensors.shape[1] - 1
            ends = start + attention_mask[:, start:].sum(1)
            all_values = [values[ix, start : ends[ix]] for ix in range(n)]
            all_logprobs = [logprobs[ix, start : ends[ix]] for ix in range(n)]

            # Compute rewards
            rewards = -self.trainer.kl_ctl.value * (logprobs - ref_logprobs)
            all_rewards = [None] * n
            for ix in range(n):
                rs = rewards[ix][start : ends[ix]]
                rs[-1] = scores[ix]
                all_rewards[ix] = rs

            new_ppo_rl_elements = [
                PPORLElement(
                    query_tensor=query_tensors[i],
                    response_tensor=response_tensors[i],
                    logprobs=all_logprobs[i],
                    values=all_values[i],
                    rewards=all_rewards[i],
                )
                for i in range(n)
            ]

            ppo_rl_elements += new_ppo_rl_elements
            exp_time = clock.tick()

        stats["kl_ctl_value"] = self.trainer.kl_ctl.value
        stats["time/exp"] = exp_time

        if not ray.is_initialized():
            self.trainer.accelerator.log(stats, step=iter_count)

        # Push samples and rewards to trainer's rollout storage
        self.trainer.push_to_store(ppo_rl_elements)
