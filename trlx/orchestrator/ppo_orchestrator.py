from time import time

import ray
import torch
import torch.nn.functional as F

from trlx.data.accelerate_base_datatypes import PromptBatch
from trlx.data.ppo_types import PPORLElement
from trlx.orchestrator import Orchestrator, register_orchestrator
from trlx.pipeline import BasePipeline
from trlx.trainer import BaseRLTrainer
from trlx.utils import Clock
from trlx.utils.modeling import RunningMoments, logprobs_from_logits


@register_orchestrator
class PPOOrchestrator(Orchestrator):
    """
    Orchestrator prepares data for PPO training.
    Transforms samples from `pipeline` into `PPOBatch` and pushes them into trainer's `store`
    """

    def __init__(
        self,
        trainer: BaseRLTrainer,
        pipeline: BasePipeline,
        chunk_size: int = 512,
    ):
        self.pipeline = pipeline
        self.trainer = trainer
        self.chunk_size = chunk_size

        self.pipeline_loader = self.pipeline.create_loader(self.chunk_size, shuffle=True)
        self.pipeline_loader = self.trainer.accelerator.prepare(self.pipeline_loader)
        self.pipeline_iterator = iter(self.pipeline_loader)

        if not hasattr(self.trainer.model, "frozen_head"):
            self.ref_model = self.trainer.get_arch(self.trainer.config)
            self.ref_model.to(self.trainer.accelerator.device)

        self.trainer.orch = self

        self.running = RunningMoments()
        self.ref_mean = self.trainer.config.method.ref_mean
        self.ref_std = self.trainer.config.method.ref_std

    def score(self, samples):
        """
        Batched scoring function taking text and generating scalar
        """
        return self.trainer.reward_fn(samples)

    def make_experience(self, num_rollouts: int = 1024, iter_count: int = 0):  # noqa:
        """
        Takes `num_rollouts` prompts from `pipeline`, samples model and computes the
        KL againts a reference model. It then appends PPOElements to trainer's `store`
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
            device = samples.device
            str_samples, str_prompts, str_outputs = self.trainer.decode(query_tensors, samples)

            # Convert trimmed samples back into tensors for another head pass
            # This can be defered, instead letting the pass to made over the original samples
            # after unbinding and truncating operations lower are fixed
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
            response_tensors = torch.vstack(outputs).to(device)

            exp_score_time = time()

            scores = torch.tensor(
                self.trainer.reward_fn(
                    samples=str_samples,
                    prompts=str_prompts,
                    outputs=str_outputs,
                ),
                dtype=float,
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
                query_tensors = batch.input_ids.to(device)
                with torch.no_grad():
                    outputs = self.trainer.model(
                        input_ids=query_tensors,
                        attention_mask=attention_mask,
                        decoder_input_ids=response_tensors,
                    )
                    logits = outputs.logits
                    values = outputs.value
                    if hasattr(self.trainer.model, "frozen_head"):
                        ref_logits = self.trainer.model.forward_hydra(
                            input_ids=query_tensors,
                            attention_mask=attention_mask,
                            decoder_input_ids=response_tensors,
                        )
                    else:
                        ref_logits = self.ref_model(
                            input_ids=query_tensors,
                            attention_mask=attention_mask,
                            decoder_input_ids=response_tensors,
                        ).logits
            else:
                all_tokens = torch.cat((query_tensors.to(device), response_tensors), dim=1)
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
                logprobs = logprobs_from_logits(logits[:, :-1, :], response_tensors[:, 1:])
                ref_logprobs = logprobs_from_logits(ref_logits[:, :-1, :], response_tensors[:, 1:])
            else:
                logprobs = logprobs_from_logits(logits[:, :-1, :], all_tokens[:, 1:])
                ref_logprobs = logprobs_from_logits(ref_logits[:, :-1, :], all_tokens[:, 1:])

            n = samples.shape[0]
            logprobs = logprobs.cpu()
            ref_logprobs = ref_logprobs.cpu()
            query_tensors = query_tensors.cpu()
            response_tensors = response_tensors.cpu()
            if self.trainer.config.model.model_arch_type == "seq2seq":
                start = 1  # skip the <s> token
                ends = (response_tensors[:, start:] != 0).sum(1)
                all_logprobs = [logprobs[ix, start : ends[ix]] for ix in range(n)]
                all_values = [values[ix, start - 1 : ends[ix] - 1] for ix in range(n)]
                rewards = [
                    -self.trainer.kl_ctl.value * (logprobs[ix, start : ends[ix]] - ref_logprobs[ix, start : ends[ix]])
                    for ix in range(n)
                ]
            else:
                values = values.cpu()[:, :-1]
                start = query_tensors.shape[1] - 1
                ends = start + attention_mask[:, start:].sum(1)
                all_values = [values[ix, start : ends[ix]] for ix in range(n)]
                all_logprobs = [logprobs[ix, start : ends[ix]] for ix in range(n)]

                rewards = -self.trainer.kl_ctl.value * (logprobs - ref_logprobs)
                rewards = [rs[start : ends[ix]] for ix, rs in enumerate(rewards)]

            for ix in range(n):
                if len(rewards[ix]) == 0:
                    continue

                rewards[ix][-1] += scores[ix].cpu()

                ppo_rl_elements.append(
                    PPORLElement(
                        query_tensor=query_tensors[ix],
                        response_tensor=response_tensors[ix],
                        logprobs=all_logprobs[ix],
                        values=all_values[ix],
                        rewards=rewards[ix],
                    )
                )

            exp_time = clock.tick()

        stats["kl_ctl_value"] = self.trainer.kl_ctl.value
        stats["time/exp"] = exp_time

        if not ray.is_initialized():
            self.trainer.accelerator.log(stats, step=iter_count)

        # Push samples and rewards to trainer's rollout storage
        self.trainer.push_to_store(ppo_rl_elements)
