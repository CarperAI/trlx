from __future__ import annotations
from time import time
from typing import List, Tuple, Optional
from dataclasses import dataclass
from dacite import from_dict

import ray
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from trlx.data.accelerate_base_datatypes import PromptBatch
from trlx.data.ppo_types import PPORLElement, RewardFnInput
from trlx.orchestrator import Orchestrator, register_orchestrator
from trlx.pipeline import BasePipeline
from trlx.trainer.accelerate_ppo_trainer import AcceleratePPOTrainer
from trlx.utils import Clock
from trlx.utils.modeling import RunningMoments, logprobs_of_labels

@dataclass
class RunElementBatch:
    # TODO have a non-batch version and base this off of that
    query_tensors: List[torch.Tensor]
    padded_samples: List[torch.Tensor]
    logprobs: List[torch.Tensor]
    values: List[torch.Tensor]
    kl_divergence_estimate: List[torch.Tensor]
    str_samples: List[str]
    str_prompts: List[str]
    str_outputs: List[str]

    # Make it so that it can be accessed as a dict
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)
    
    # Make it so that two RunElements can be added together.
    # Assume all List attributes are the same length, and add them elementwise
    # Assume the tensors can be added together
    def __add__(self, other : RunElementBatch):
        return RunElementBatch(
            query_tensors=self.query_tensors + other.query_tensors,
            padded_samples=self.padded_samples + other.padded_samples,
            logprobs=self.logprobs + other.logprobs,
            values=self.values + other.values,
            kl_divergence_estimate=self.kl_divergence_estimate + other.kl_divergence_estimate,
            str_samples=self.str_samples + other.str_samples,
            str_prompts=self.str_prompts + other.str_prompts,
            str_outputs=self.str_outputs + other.str_outputs,
        )

@register_orchestrator
class PPOOrchestrator(Orchestrator):
    """PPO Orchestrator

    Runs rollouts - generates samples from prompts using the model, calculates
    KL divergence against the reference model, and then pushes these
    samples/rewards etc to the trainers store.

    Note this class is intwined with the trainer `AcceleratePPOTrainer` - it
    adds an `orch` property to the trainer instance and also sets a `trainer`
    property on itself. See the trainer class for more details.
    """

    def __init__(
        self,
        trainer: AcceleratePPOTrainer,
        pipeline: BasePipeline,
        chunk_size: int = 512,
    ):
        """_summary_

        Args:
            trainer: Trainer
            pipeline: Dataset
            chunk_size: Batch size
        """
        self.pipeline = pipeline
        self.trainer = trainer
        self.chunk_size = chunk_size

        # Create the dataloader (for batches of prompts)
        self.pipeline_loader: DataLoader = self.pipeline.create_loader(
            self.chunk_size, shuffle=True
        )
        self.pipeline_loader = self.trainer.accelerator.prepare(self.pipeline_loader)
        self.pipeline_iterator = iter(self.pipeline_loader)

        if not hasattr(self.trainer.model, "frozen_head"):
            self.ref_model = self.trainer.get_arch(self.trainer.config)
            self.ref_model.to(self.trainer.accelerator.device)

        # Set this orchestrator as a property on the trainer, so that the
        # trainer can call `make_experience` directly for each epoch.
        self.trainer.orch = self

        self.running = RunningMoments()
        self.ref_mean = self.trainer.config.method.ref_mean
        self.ref_std = self.trainer.config.method.ref_std

        if self.trainer.experience_fn is None:
                self.trainer.experience_fn = self.default_experience_fn
    

    def generate_and_calc_logprobs(self, batch: PromptBatch):
        """
            self: PPOOrchestrator
                has a trainer property with trainer.generate method
            batch: PromptBatch

            returns: Tuple(dict, dict)
                first dict keys: 
                    {'query_tensors', 'padded_samples', 'all_logprobs', 'kl_divergence_estimate', 'response_tensors', 'all_values', 'str_samples', 'str_prompts', 'str_outputs'}
                second dict (stats) keys:
                    {'time/exp_generate', 'kl_ctl_value'}

            Does almost everything in the inner loop of make_experience, except anything related to the reward function and scores.
        """

        stats = {}
        exp_generate_time = time()
        samples = self.trainer.generate(**batch)
        stats["time/exp_generate"] = time() - exp_generate_time

        query_tensors = batch.input_ids
        device = samples.device
        str_samples, str_prompts, str_outputs = self.trainer.decode(
            query_tensors, samples
        )

        # Pad the samples
        outputs = self.trainer.tokenizer(str_outputs).input_ids
        outputs = list(map(torch.LongTensor, outputs))
        maxsize = max(map(len, outputs))
        outputs = [
            F.pad(
                output,
                (0, maxsize - len(output)),
                value=self.trainer.tokenizer.pad_token_id,
            )
            for output in outputs
        ]
        padded_samples = torch.stack(outputs).to(device)

        # Precompute logprobs, values
        if self.trainer.config.model.model_arch_type == "seq2seq":
            attention_mask = batch.attention_mask.to(device)
            query_tensors = batch.input_ids.to(device)
            with torch.no_grad():
                outputs = self.trainer.model(
                    input_ids=query_tensors,
                    attention_mask=attention_mask,
                    labels=padded_samples,
                )
                logits = outputs.logits
                values = outputs.value
                if hasattr(self.trainer.model, "frozen_head"):
                    ref_logits = self.trainer.model.forward_hydra(
                        input_ids=query_tensors,
                        attention_mask=attention_mask,
                        decoder_input_ids=padded_samples,
                    )
                else:
                    ref_logits = self.ref_model(
                        input_ids=query_tensors,
                        attention_mask=attention_mask,
                        decoder_input_ids=padded_samples,
                    ).logits
        else:
            #print("not seq2seq") # this
            all_tokens = torch.cat([query_tensors.to(device), padded_samples], dim=1)
            attention_mask = (
                all_tokens.not_equal(self.trainer.tokenizer.pad_token_id)
                .long()
                .to(device)
            )
            with torch.no_grad():
                logits, *_, values = self.trainer.model(
                    all_tokens, attention_mask=attention_mask,
                )
                # TODO(dahoas): When hydra model works need to also support generation on hydra head
                if hasattr(self.trainer.model, "frozen_head"):
                    ref_logits = self.trainer.model.forward_hydra(
                        all_tokens,
                        attention_mask=attention_mask,
                        return_dict=False,
                    )
                else:
                    #print("not frozen_head") # this
                    ref_logits, _, *_ = self.ref_model(
                        all_tokens,
                        attention_mask=attention_mask,
                        return_dict=False,
                    )
                    ref_logits = ref_logits.to(device)
        
        if self.trainer.config.model.model_arch_type == "seq2seq":
            logprobs = logprobs_of_labels(logits[:, :-1, :], padded_samples[:, 1:])
            ref_logprobs = logprobs_of_labels(
                ref_logits[:, :-1, :], padded_samples[:, 1:]
            )
        else:
            logprobs = logprobs_of_labels(logits, all_tokens)
            ref_logprobs = logprobs_of_labels(ref_logits, all_tokens)
        
        n_samples: int = samples.shape[0]
        logprobs = logprobs.cpu()
        ref_logprobs = ref_logprobs.cpu()
        query_tensors = query_tensors.cpu()
        padded_samples = padded_samples.cpu()

        # Estimate the KL divergence between the model and reference model
        if self.trainer.config.model.model_arch_type == "seq2seq":
            # Skip the beginning of sequence token
            start = 1

            # Get the number of non-padding tokens for each sample
            # This assumes all padding is on the right side
            padding_token: int = 0
            ends = (padded_samples[:, start:] != padding_token).sum(1)

            # Get the logprobs and values, for tokens that are not padding
            # or beginning of sequences tokens. These are from the model
            # (not the reference model)
            all_logprobs = [
                logprobs[ix, start : ends[ix]] for ix in range(n_samples)
            ]
            all_values = [
                values[ix, start - 1 : ends[ix] - 1] for ix in range(n_samples)
            ]

            kl_divergence_estimate: List[torch.Tensor] = [
                -self.trainer.kl_ctl.value
                * (
                    logprobs[sample_idx, start : ends[sample_idx]]
                    - ref_logprobs[sample_idx, start : ends[sample_idx]]
                )
                for sample_idx in range(n_samples)
            ]

        # Else if not seq2seq (i.e. causal)
        else:
            logprobs = logprobs_of_labels(logits[:, :-1, :], all_tokens[:, 1:])
            ref_logprobs = logprobs_of_labels(
                ref_logits[:, :-1, :], all_tokens[:, 1:]
            )

            n_samples = samples.shape[0]
            #values = values.cpu()[:, :-1]
            values = values[:, :-1]
            #logprobs = logprobs.cpu()
            #ref_logprobs = ref_logprobs.cpu()
            #query_tensors = query_tensors.cpu()
            #padded_samples = padded_samples.cpu()

            start = query_tensors.shape[1] - 1
            ends = start + attention_mask[:, start:].sum(1)
            all_values = [values[ix, start : ends[ix]] for ix in range(n_samples)]
            all_logprobs = [
                logprobs[ix, start : ends[ix]] for ix in range(n_samples)
            ]

            kl_divergence_estimate = -self.trainer.kl_ctl.value * (
                logprobs - ref_logprobs
            )
            kl_divergence_estimate = [
                rs[start : ends[ix]] for ix, rs in enumerate(kl_divergence_estimate)
            ]
        
        return RunElementBatch(
            query_tensors=query_tensors,
            padded_samples=padded_samples,
            logprobs=all_logprobs,
            values=all_values,
            kl_divergence_estimate=kl_divergence_estimate,
            str_samples=str_samples,
            str_prompts=str_prompts,
            str_outputs=str_outputs,
        ), stats


    def default_experience_fn(self, batch : PromptBatch) -> Tuple[Optional[List], RunElementBatch, dict]:
        """
        Any experience_fn should use self.generate_and_calc_logprobs(batch) in some way.
        This default_experience_fn is a wrapper around that function that returns the same data as generate_and_calc_logprobs.

        If an user-provided experience_fn returns non-None trajectories,
        passed to the reward function in the List[Any] form, batch_size outer lists, arbitrary inner lists.
        The user-provided reward function should be able to handle this format.

        The default_experience_fn returns None for trajectories, in which case the orchestrator will default
        to passing (str_samples, str_prompts, str_outputs) to the reward function.
    

        :return: trajectories, data : RunElementBatch, stats
        """ 
        data, stats = self.generate_and_calc_logprobs(batch)

        #print("data['samples'].shape", data['samples'].shape) # (batch_size, max_length)
        #print("data['samples']", data['samples']) # some tokens

        #texts = data['str_samples']
        #trajectories = [[text] for text in texts]
        #print("trajectories", trajectories)

        trajectories = None
        return trajectories, data, stats
    

    # Rewrite the make_experience function to use generate_and_calc_logprobs
    def make_experience(self, num_rollouts: int = 1024, iter_count: int = 0): 
        """Make experiences

        Takes `num_rollouts` prompts from `pipeline`, samples from the model and
        then computes the KL against a reference model. Finally it then appends
        PPOElements to trainer's `store`.

        Args:
            num_rollouts: Number of rollouts to generate
            iter_count: Total number of updates run (i.e. number of updates run
            for all batches & epochs)
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

            # Use the generate_and_calc_logprobs function to get all data needed for PPO
            trajectories, data, stats = self.trainer.experience_fn(batch)
            # data: dict with keys "query_tensors", "padded_samples", "logprobs", "values", "kl_divergence_estimate"

            query_tensors = data["query_tensors"]
            device = query_tensors.device
            assert torch.all(query_tensors == batch.input_ids.to(device))

            padded_samples = data["padded_samples"]
            all_logprobs = data["logprobs"]
            all_values = data["values"]
            kl_divergence_estimate = data["kl_divergence_estimate"]


            exp_score_time = time()

            if trajectories is None:
                str_samples = data["str_samples"]
                str_prompts = data["str_prompts"]
                str_outputs = data["str_outputs"]

                print("str_samples: ", str_samples)
                print("str_prompts: ", str_prompts)
                print("str_outputs: ", str_outputs)
                scores = torch.tensor(
                    self.trainer.reward_fn(samples=str_samples, prompts=str_prompts, outputs=str_outputs),
                    dtype=torch.float,
                ).to(device)
            else:
                scores = torch.tensor(
                    self.trainer.reward_fn(trajectories), dtype=torch.float
                ).to(device)
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

            # Compute rewards
            n_samples = len(scores)
            assert query_tensors.shape[0] == n_samples
            all_rewards = [None] * n_samples

            for sample_idx in range(n_samples):
                sample_kl_divergence_estimate = kl_divergence_estimate[sample_idx]

                if len(sample_kl_divergence_estimate) == 0:
                    sample_kl_divergence_estimate = torch.tensor([0.0])

                sample_kl_divergence_estimate[-1] += scores[sample_idx].cpu()
                # TODO refactor this code, the above line is horrifying
                all_rewards[sample_idx] = sample_kl_divergence_estimate

            new_ppo_rl_elements = [
                PPORLElement(
                    query_tensor=query_tensors[i],
                    response_tensor=padded_samples[i],
                    logprobs=all_logprobs[i],
                    values=all_values[i],
                    rewards=all_rewards[i],
                )
                for i in range(n_samples)
            ]
            ppo_rl_elements += new_ppo_rl_elements
            exp_time = clock.tick()

        stats["kl_ctl_value"] = self.trainer.kl_ctl.value
        stats["time/exp"] = exp_time

        if not ray.is_initialized():
            self.trainer.accelerator.log(stats, step=iter_count)

        # Push samples and rewards to trainer's rollout storage
        self.trainer.push_to_store(ppo_rl_elements)
