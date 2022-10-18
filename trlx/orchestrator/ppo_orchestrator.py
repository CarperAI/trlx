from typing import Callable

import torch

from trlx.data.accelerate_base_datatypes import PromptBatch
from trlx.data.ppo_types import PPORLElement
from trlx.model import BaseRLModel
from trlx.orchestrator import Orchestrator, register_orchestrator
from trlx.pipeline import BasePipeline
from trlx.utils.modeling import logprobs_from_logits


@register_orchestrator
class PPOOrchestrator(Orchestrator):
    def __init__(
        self,
        model: BaseRLModel,
        pipeline: BasePipeline,
        reward_fn,
        stats_fn: Callable = None,
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
        self.rl_model.stats_fn = stats_fn

    def score(self, samples):
        """
        Batched scoring function taking text and generating scalar
        """
        return self.rl_model.reward_fn(samples)

    def make_experience(self, num_rollouts: int = 1024, iter_count: int = 0):
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

            query_tensors, response_tensors, texts = self.rl_model.act(batch)
            scores = torch.as_tensor(self.score(texts))

            # Precompute logprobs, values
            all_tokens = torch.cat((query_tensors, response_tensors), dim=1)
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
            all_values = v[:, start - 1 : end - 1]
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

        stats = {"exp_time": exp_time}
        self.rl_model.accelerator.log(stats, step=iter_count)

        # Push samples and rewards to model's rollout storage
        self.rl_model.push_to_store(ppo_rl_elements)
