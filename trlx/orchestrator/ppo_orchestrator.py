from typing import Callable

import torch
from trlx.data.accelerate_base_datatypes import PromptBatch
from trlx.data.ppo_types import PPORLElement
from trlx.model import BaseRLModel
from trlx.model.nn.ppo_models import GPTHeadWithValueModel, GPTHydraHeadWithValueModel
from trlx.orchestrator import Orchestrator, register_orchestrator
from trlx.pipeline import BasePipeline
from trlx.utils import Clock
from trlx.utils.modeling import logprobs_from_logits
import time


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

        self.rl_model.orch = self
        self.rl_model.reward_fn = reward_fn
        self.rl_model.metric_fn = metric_fn

    def score(self, samples):
        """
        Batched scoring function taking text and generating scalar
        """
        return self.rl_model.reward_fn(samples)

    def make_experience(self, num_rollouts: int = 1024, iter_count: int = 0):
        """
        Takes `num_rollouts` prompts from `pipeline`, samples model, computes KL againts a reference model appends PPOElements to model's `store`
        """
        #print("MAKING EXPERIENCES")

        ppo_rl_elements = []
        stats = {}
        clock = Clock()
        # If using zero3 master node generates experiences in batch
        # and broadcasts to worker nodes

        while len(ppo_rl_elements) < num_rollouts:
            # Get next batch in prompt dataset and refresh if exhausted
            try:
                batch: PromptBatch = next(self.pipeline_iterator)
            except StopIteration:
                self.pipeline_iterator = iter(self.pipeline_loader)
                batch = next(self.pipeline_iterator)

            #print("GENERATING SAMPLES")
            #print("MODEL PARALLELL: ", self.rl_model.model.model_parallel)
            #self.rl_model.accelerator.wait_for_everyone()
            # Removing barriers seems to help with generation deadlock
            samples = self.rl_model.generate(**batch)
            #print("FINISHED GENERATING SAMPLES")
            

            query_tensors = batch.input_ids
            response_tensors = samples[:, query_tensors.shape[1] :]
            texts = self.rl_model.tokenizer.batch_decode(
                samples, skip_special_tokens=True
            )
            scores = torch.as_tensor(self.score(texts))

            # Score ref_texts to compute ref reward mean, var
            ## This is very slow, so pre-computation of mean, std is desirable
            ref_mean = self.rl_model.config.method.ref_mean
            ref_std = self.rl_model.config.method.ref_std
            if ref_mean is None or ref_std is None:
                ref_samples = self.rl_model.ref_generate(**batch)
                ref_texts = self.rl_model.tokenizer.batch_decode(ref_samples, skip_special_tokens=True)
                ref_scores = torch.as_tensor(self.score(ref_texts))
                ref_mean = torch.mean(ref_scores).item()
                ref_std = torch.std(ref_scores).item()
            # Normalize scores
            scores = (scores - ref_mean) / ref_std

            # Precompute logprobs, values
            all_tokens = torch.cat(
                (query_tensors.to(samples.device), response_tensors), dim=1
            )
            with torch.no_grad():
                #TODO(dahoas): probably we want to compute these logits with attention masks
                print("COMPUTING LOGITS", torch.distributed.get_rank())
                logits, v, ref_logits = self.rl_model.model(all_tokens)

            #ref_logits = ref_logits.to(self.rl_model.accelerator.device)
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

        stats = {"exp_time": exp_time}
        self.rl_model.accelerator.log(stats, step=iter_count)

            # If model parallel need to communicate dataset elements
            #if self.rl_model.model_parallel:
            #    self.rl_model.accelerator.broadcast(ppo_rl_elements, 0)

        # Wait to receive ppo_rl_elements from main process
        #print("WAITING FOR MAIN PROCESS")
        #self.rl_model.accelerator.wait_for_everyone()

        # If model parallel take subset of broadcasted data
        #if self.rl_model.model_parallel:
        #    rank = torch.distributed.get_rank()
        #    world_size = torch.distributed.get_world_size()
        #    shard_size = len(ppo_rl_elements) // world_size
        #    assert shard_size * world_size == len(ppo_rl_elements)
        #    ppo_rl_elements = ppo_rl_elements[rank*shard_size : (rank+1)*shard_size]

        # Push samples and rewards to model's rollout storage
        self.rl_model.push_to_store(ppo_rl_elements)
        #self.rl_model.accelerator.wait_for_everyone()
