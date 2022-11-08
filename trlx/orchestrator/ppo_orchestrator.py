from typing import Callable

import torch
from trlx.data.accelerate_base_datatypes import PromptBatch
from trlx.data.ppo_types import PPORLElement
from trlx.model import BaseRLModel
from trlx.orchestrator import Orchestrator, register_orchestrator
from trlx.pipeline import BasePipeline
from trlx.utils import Clock
from trlx.utils.modeling import logprobs_from_logits, RunningMoments

from time import time
import ray


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
        experience_fn: Callable = None,
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
        self.rl_model.experience_fn = experience_fn if experience_fn is not None else self.default_experience_fn
         
    def score(self, samples):
        """
        Batched scoring function taking text and generating scalar
        """
        return self.rl_model.reward_fn(samples)

    def generate_and_calc_logprobs(self, batch):
        stats = {}
        exp_generate_time = time()
        samples = self.rl_model.generate(**batch)
        stats["exp_generate_time"] = time() - exp_generate_time

        query_tensors = batch.input_ids
        response_tensors = samples[:, query_tensors.shape[1] :]
        all_tokens = torch.cat(
            (query_tensors.to(samples.device), response_tensors), dim=1
        )
        with torch.no_grad():
            logits, _, v = self.rl_model.model(all_tokens)
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

        return {'samples': samples, 'all_logprobs': all_logprobs, 'all_ref_logprobs': all_ref_logprobs, 'query_tensors': query_tensors, 'response_tensors': response_tensors, 'all_values': all_values}, stats

    def default_experience_fn(self, batch):
        data, stats = self.generate_and_calc_logprobs(batch)
        
        samples, all_logprobs, all_ref_logprobs, query_tensors, response_tensors, all_values = data['samples'], data['all_logprobs'], data['all_ref_logprobs'], data['query_tensors'], data['response_tensors'], data['all_values']
        return {'samples': samples, 'all_logprobs': all_logprobs, 'all_ref_logprobs': all_ref_logprobs, 'all_values': all_values, 'query_tensors': query_tensors, 'response_tensors': response_tensors}, stats

    def make_experience(self, num_rollouts: int = 1024, iter_count: int = 0):
        """
        Takes `num_rollouts` prompts from `pipeline`, samples model according to experience_fn, computes KL againts a reference model appends PPOElements to model's `store`
        """
        ppo_rl_elements = []
        clock = Clock()
        while len(ppo_rl_elements) < num_rollouts:
            # Get next batch in prompt dataset and refresh if exhausted
            try:
                batch: PromptBatch = next(self.pipeline_iterator)
            except StopIteration:
                self.pipeline_iterator = iter(self.pipeline_loader)
                batch = next(self.pipeline_iterator)

            data, stats = self.rl_model.experience_fn(batch)

            kls = data['all_logprobs'] - data['all_ref_logprobs']
            non_score_rewards = -self.rl_model.kl_ctl.value * kls
            all_rewards = non_score_rewards.clone()

            print("all_rewards.shape", all_rewards.shape)

            texts = self.rl_model.tokenizer.batch_decode(
                data['samples'], skip_special_tokens=True
            )
            exp_score_time = time()
            scores = torch.as_tensor(self.score(texts), device=data['samples'].device)
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

            all_rewards[:, -1] += scores.to(self.rl_model.accelerator.device)
            print("all_rewards", all_rewards[0])
            all_rewards = all_rewards.cpu()

            data = {k: v.cpu() for k, v in data.items()}
            # print where the data tensors are
            print("data['all_logprobs']", data['all_logprobs'][0])
            print("data['all_ref_logprobs']", data['all_ref_logprobs'][0])
            print("data['all_values']", data['all_values'][0])
            print("data['query_tensors']", data['query_tensors'][0])
            print("data['response_tensors']", data['response_tensors'][0])
            print("all_rewards", all_rewards[0])
            
            exp_time = clock.tick()

            new_ppo_rl_elements = [
                PPORLElement(
                    query_tensor=data['query_tensors'][i, :],
                    response_tensor=data['response_tensors'][i, :],
                    logprobs=data['all_logprobs'][i, :],
                    values=data['all_values'][i, :],
                    rewards=all_rewards[i, :],
                )
                for i in range(data['query_tensors'].size()[0])
            ]
            ppo_rl_elements += new_ppo_rl_elements

        stats["kl_ctl_value"] = self.rl_model.kl_ctl.value
        stats["exp_time"] = exp_time

        if not ray.is_initialized():
            self.rl_model.accelerator.log(stats, step=iter_count)

        # Push samples and rewards to model's rollout storage
        self.rl_model.push_to_store(ppo_rl_elements)
