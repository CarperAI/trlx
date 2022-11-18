from typing import Callable

import torch
from trlx.data.accelerate_base_datatypes import PromptBatch
from trlx.data.ppo_types import PPORLElement
from trlx.model import BaseRLModel
from trlx.model.nn.ppo_models import GPTHeadWithValueModel, GPTHydraHeadWithValueModel
from trlx.orchestrator import Orchestrator, register_orchestrator
from trlx.pipeline import BasePipeline
from trlx.utils import Clock
from trlx.utils.modeling import logprobs_from_logits, RunningMoments

from time import time
import ray
from ray.air import session


@register_orchestrator
class PPOOrchestrator(Orchestrator):
    """
    Orchestrator that prepares data for PPO training: transforms samples from `pipeline` into `PPOBatch` and pushes them into model's `store`
    """

    def __init__(
        self,
        model: BaseRLModel,
        pipeline: BasePipeline,
        reward_fn: Callable,
        metric_fn: Callable = None,
        chunk_size: int = 512,
    ):
        self.pipeline = pipeline
        self.rl_model = model
        self.chunk_size = chunk_size

        self.pipeline_loader = self.pipeline.create_loader(
            self.chunk_size, shuffle=True
        )
        self.pipeline_loader = self.rl_model.accelerator.prepare(self.pipeline_loader)
        self.pipeline_iterator = iter(self.pipeline_loader)

        if not hasattr(self.rl_model.model, "frozen_head"):
            self.ref_model = self.rl_model.get_arch(self.rl_model.config)

        self.rl_model.orch = self
        self.rl_model.reward_fn = reward_fn
        self.rl_model.metric_fn = metric_fn

        self.running = RunningMoments()
        self.ref_mean = self.rl_model.config.method.ref_mean
        self.ref_std = self.rl_model.config.method.ref_std

    def score(self, samples):
        """
        Batched scoring function taking text and generating scalar
        """
        return self.rl_model.reward_fn(samples)

    def make_experience(self, num_rollouts: int = 1024, iter_count: int = 0):
        """
        Takes `num_rollouts` prompts from `pipeline`, samples model, computes KL againts a reference model appends PPOElements to model's `store`
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
            samples = self.rl_model.generate(**batch)
            stats["exp_generate_time"] = time() - exp_generate_time

            query_tensors = batch.input_ids
            response_tensors = samples[:, query_tensors.shape[1] :]
            texts = self.rl_model.tokenizer.batch_decode(
                samples, skip_special_tokens=True
            )
            exp_score_time = time()
            scores = torch.as_tensor(self.score(texts), device=samples.device)
            stats["exp_score_time"] = time() - exp_score_time

            # store statistics of the initial rollout as reference
            if self.ref_mean is None:
                self.ref_mean, self.ref_std = scores.mean(), scores.std()
            all_scores_mean, all_scores_std = self.running.update(scores)
            stats["exp_scores_mean"] = all_scores_mean
            stats["exp_scores_std"] = all_scores_std
            stats["running_mean"] = self.running.mean
            stats["running_std"] = self.running.std

            if self.rl_model.config.method.scale_reward == "running":
                scores /= self.running.std
            elif self.rl_model.config.method.scale_reward == "ref":
                scores /= self.ref_std

            clip_reward = self.rl_model.config.method.cliprange_reward
            if clip_reward:
                scores = torch.clip(scores, -clip_reward, clip_reward)

            # Precompute logprobs, values
            all_tokens = torch.cat(
                (query_tensors.to(samples.device), response_tensors), dim=1
            )
            with torch.no_grad():
                logits, _, v = self.rl_model.model(all_tokens)
                # TODO(dahoas): When hydra model works need to also support generation on hydra head
                if hasattr(self.rl_model.model, "frozen_head"):
                    ref_logits = self.rl_model.model.forward_hydra(
                        all_tokens, return_dict=False
                    )
                else:
                    ref_logits, _, _ = self.ref_model(all_tokens.cpu())

            ref_logits = ref_logits.to(self.rl_model.accelerator.device)
            logprobs = logprobs_from_logits(logits[:, :-1, :], all_tokens[:, 1:])
            ref_logprobs = logprobs_from_logits(
                ref_logits[:, :-1, :], all_tokens[:, 1:]
            )
            start = query_tensors.size()[1] - 1
            end = query_tensors.size()[1] + response_tensors.size()[1] - 1
            all_values = v[:, start:end]
            all_logprobs = logprobs[:, start:end]
            all_ref_logprobs = ref_logprobs[:, start:end]

            # Compute rewards
            kls = all_logprobs - all_ref_logprobs
            non_score_rewards = -self.rl_model.kl_ctl.value * kls
            all_rewards = non_score_rewards.clone()
            all_rewards[:, -1] += scores.to(self.rl_model.accelerator.device)

            query_tensors = query_tensors.cpu()
            response_tensors = response_tensors.cpu()
            all_logprobs = all_logprobs.cpu()
            all_values = all_values.cpu()
            all_rewards = all_rewards.cpu()

            exp_time = clock.tick()

            new_ppo_rl_elements = [
                PPORLElement(
                    query_tensor=query_tensors[i, :],
                    response_tensor=response_tensors[i, :],
                    logprobs=all_logprobs[i, :],
                    values=all_values[i, :],
                    rewards=all_rewards[i, :],
                )
                for i in range(query_tensors.size()[0])
            ]
            ppo_rl_elements += new_ppo_rl_elements

        stats["kl_ctl_value"] = self.rl_model.kl_ctl.value
        stats["exp_time"] = exp_time

        if not ray.is_initialized():
            self.rl_model.accelerator.log(stats, step=iter_count)

        # Push samples and rewards to model's rollout storage
        self.rl_model.push_to_store(ppo_rl_elements)
