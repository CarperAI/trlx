import os
from time import time
from typing import List

import ray
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import trlx.utils.logging as logging
from trlx.data.accelerate_base_datatypes import PromptBatch
from trlx.data.ppo_types import PPORLElement
from trlx.orchestrator import Orchestrator, register_orchestrator
from trlx.pipeline import BasePipeline
from trlx.trainer.accelerate_ppo_trainer import AcceleratePPOTrainer
from trlx.utils import Clock
from trlx.utils.modeling import RunningMoments, logprobs_of_labels

logger = logging.get_logger(__name__)


@register_orchestrator
class PPOOrchestrator(Orchestrator):
    """PPO Orchestrator

    Runs rollouts - generates samples from prompts using the model, calculates
    KL divergence against the reference model, and then pushes these
    samples/rewards etc to the trainer's store.

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
        self.pipeline_loader: DataLoader = self.pipeline.create_loader(self.chunk_size, shuffle=True)
        self.pipeline_loader = self.trainer.accelerator.prepare_data_loader(self.pipeline_loader)
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

    def make_experience(self, num_rollouts: int = 1024, iter_count: int = 0):  # noqa:
        """Make experiences

        Takes `num_rollouts` prompts from `pipeline`, samples from the model and
        then computes the KL against a reference model. Finally it then appends
        PPOElements to trainer's `store`.

        Args:
            num_rollouts: Number of rollouts to generate
            iter_count: Total number of updates run (i.e. number of updates run
            for all batches & epochs)
        """
        logger.info("Collecting rollouts")
        tbar = logging.tqdm(
            total=num_rollouts,
            disable=os.environ.get("RANK", 0) != "0",
            desc=f"[rollout 0 / {num_rollouts}]",
            # Lower progress bar by 1 if we're in WARNING mode or above to avoid hiding high priority progress
            # bars (e.g. loss progress in trainers)
            position=logging.get_verbosity() >= logging.WARNING,
            # Leave progress bar if we're in INFO mode or lower to avoid spamming in suppressed verbosity levels
            leave=logging.get_verbosity() < logging.WARNING,
        )

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

            # Generate samples from the language model (similar to using
            # HuggingFace `generate` method)
            samples = self.trainer.generate(**batch)
            stats["time/exp_generate"] = time() - exp_generate_time

            prompt_tensors = batch.input_ids
            device = samples.device
            str_samples, str_prompts, str_outputs = self.trainer.decode(prompt_tensors, samples)

            # Pad the sample outputs
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
            sample_outputs = torch.vstack(outputs).to(device)

            exp_score_time = time()

            scores = torch.tensor(
                self.trainer.reward_fn(
                    samples=str_samples,
                    prompts=str_prompts,
                    outputs=str_outputs,
                ),
                dtype=torch.float,
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

            # Precompute logprobs, values
            if self.trainer.config.model.model_arch_type == "seq2seq":
                attention_mask = batch.attention_mask.to(device)
                prompt_tensors = batch.input_ids.to(device)
                with torch.no_grad():
                    outputs = self.trainer.model(
                        input_ids=prompt_tensors,
                        attention_mask=attention_mask,
                        decoder_input_ids=sample_outputs,
                    )
                    logits = outputs.logits
                    values = outputs.value
                    if hasattr(self.trainer.model, "frozen_head"):
                        ref_logits = self.trainer.model.forward_hydra(
                            input_ids=prompt_tensors,
                            attention_mask=attention_mask,
                            decoder_input_ids=sample_outputs,
                        )
                    else:
                        ref_logits = self.ref_model(
                            input_ids=prompt_tensors,
                            attention_mask=attention_mask,
                            decoder_input_ids=sample_outputs,
                        ).logits
            else:
                all_tokens = torch.cat((prompt_tensors.to(device), sample_outputs), dim=1)
                attention_mask = all_tokens.not_equal(self.trainer.tokenizer.pad_token_id).long().to(device)
                with torch.no_grad():
                    logits, *_, values = self.trainer.model(
                        all_tokens,
                        attention_mask=attention_mask,
                    )
                    # TODO(dahoas): When hydra model works need to also support generation on hydra head
                    if hasattr(self.trainer.model, "frozen_head"):
                        ref_logits = self.trainer.model.forward_hydra(
                            all_tokens,
                            attention_mask=attention_mask,
                            return_dict=False,
                        )
                    else:
                        ref_logits, _, *_ = self.ref_model(
                            all_tokens,
                            attention_mask=attention_mask,
                            return_dict=False,
                        )
                        ref_logits = ref_logits.to(device)

            if self.trainer.config.model.model_arch_type == "seq2seq":
                logprobs = logprobs_of_labels(logits[:, :-1, :], sample_outputs[:, 1:])
                ref_logprobs = logprobs_of_labels(ref_logits[:, :-1, :], sample_outputs[:, 1:])
            else:
                logprobs = logprobs_of_labels(logits[:, :-1, :], all_tokens[:, 1:])
                ref_logprobs = logprobs_of_labels(ref_logits[:, :-1, :], all_tokens[:, 1:])

            n_samples: int = samples.shape[0]
            logprobs = logprobs.cpu()
            ref_logprobs = ref_logprobs.cpu()
            prompt_tensors = prompt_tensors.cpu()
            sample_outputs = sample_outputs.cpu()

            # Estimate the KL divergence between the model and reference model
            if self.trainer.config.model.model_arch_type == "seq2seq":
                # Skip the beginning of sequence token
                start = 1

                # Get the number of non-padding tokens for each sample
                # This assumes all padding is on the right side
                padding_token: int = 0
                ends = (sample_outputs[:, start:] != padding_token).sum(1)

                # Get the logprobs and values, for tokens that are not padding
                # or beginning of sequences tokens. These are from the model
                # (not the reference model)
                all_logprobs = [logprobs[ix, start : ends[ix]] for ix in range(n_samples)]
                all_values = [values[ix, start - 1 : ends[ix] - 1] for ix in range(n_samples)]

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
                values = values.cpu()[:, :-1]
                start = prompt_tensors.shape[1] - 1
                ends = start + attention_mask[:, start:].sum(1)
                all_values = [values[ix, start : ends[ix]] for ix in range(n_samples)]
                all_logprobs = [logprobs[ix, start : ends[ix]] for ix in range(n_samples)]

                kl_divergence_estimate = -self.trainer.kl_ctl.value * (logprobs - ref_logprobs)
                kl_divergence_estimate = [rs[start : ends[ix]] for ix, rs in enumerate(kl_divergence_estimate)]

            # Compute rewards
            all_rewards = []

            rollout_count = 0

            for sample_idx in range(n_samples):
                sample_kl_divergence_estimate = kl_divergence_estimate[sample_idx]

                if len(sample_kl_divergence_estimate) == 0 or len(all_logprobs[sample_idx]) == 0:
                    continue

                sample_kl_divergence_estimate[-1] += scores[sample_idx].cpu()
                all_rewards.append(sample_kl_divergence_estimate)

                ppo_rl_elements.append(
                    PPORLElement(
                        query_tensor=prompt_tensors[i],
                        response_tensor=sample_outputs[i],
                        logprobs=all_logprobs[i],
                        values=all_values[i],
                        rewards=all_rewards[i],
                    )
                )

                rollout_count += 1
            exp_time = clock.tick()
            tbar.set_description(f"[rollout {len(ppo_rl_elements)} / {num_rollouts}]")
            tbar.update(min(rollout_count, num_rollouts))
        tbar.close()

        stats["kl_ctl_value"] = self.trainer.kl_ctl.value
        stats["time/exp"] = exp_time

        if not ray.is_initialized():
            self.trainer.accelerator.log(stats, step=iter_count)

        # Push samples and rewards to trainer's rollout storage
        self.trainer.push_to_store(ppo_rl_elements)
