import json
import os
import uuid
from time import time
from typing import Callable, List

import numpy as np
import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import trlx.utils.logging as logging
from trlx.data.accelerate_base_datatypes import PromptBatch
from trlx.data.configs import TRLConfig
from trlx.data.ppo_types import PPORLBatch, PPORLElement
from trlx.models.modeling_ppo import (
    AdaptiveKLController,
    AutoModelForCausalLMWithHydraValueHead,
    AutoModelForSeq2SeqLMWithHydraValueHead,
    FixedKLController,
)
from trlx.pipeline.offline_pipeline import PromptPipeline
from trlx.pipeline.ppo_pipeline import PPORolloutStorage
from trlx.trainer import register_trainer
from trlx.trainer.accelerate_base_trainer import AccelerateRLTrainer
from trlx.utils import Clock, infinite_dataloader
from trlx.utils.modeling import RunningMoments, gather_dict, logprobs_of_labels

logger = logging.get_logger(__name__)


@register_trainer
class AcceleratePPOTrainer(AccelerateRLTrainer):
    """PPO Accelerate Trainer"""

    reward_fn: Callable[[List[str], List[str], List[str]], List[float]]
    tokenizer: AutoTokenizer

    def __init__(self, config: TRLConfig, **kwargs):
        """PPO Accelerate Trainer initialization

        Args:
            config: Config
        """
        super().__init__(config, **kwargs)

        # Setup rollout logging
        if config.train.rollout_logging_dir is not None:
            self.log_rollouts = True
            self.setup_rollout_logging(config)
        else:
            self.log_rollouts = False

        # Setup the rollout store
        # Rollouts contain the prompt & response, log probs, values and rewards - from each rollout
        self.store = PPORolloutStorage(self.tokenizer.pad_token_id, self.tokenizer.padding_side)

        # Create the rollout store dataloader (for batching up rollouts)
        # TODO (jon-tow): This is only used to satisfy to `accelerator.prepare` call constraint below - remove in future
        rollout_loader: DataLoader = self.store.create_loader(self.config.train.batch_size, shuffle=True)

        # Prepare multi-GPU acceleration
        self.model, self.opt, self.scheduler, rollout_loader = self.accelerator.prepare(
            self.model, self.opt, self.scheduler, rollout_loader
        )

        self.store.clear_history()  # Clear the rollout store

        # Setup a reference model when hydra heads and distributed ref model are not used
        self.dist_ref_model = config.method.dist_ref_model
        if not hasattr(self.model, "frozen_head") and not self.model.peft_type and not self.dist_ref_model:
            self.ref_model = self.get_arch(self.config)
            self.ref_model.to(self.accelerator.device)
            self.ref_model.eval()

        # Set up the KL controller
        # This helps prevent large divergences in the controller (policy)
        if config.method.target is not None:
            self.kl_ctl = AdaptiveKLController(config.method.init_kl_coef, config.method.target, config.method.horizon)
        else:
            self.kl_ctl = FixedKLController(config.method.init_kl_coef)

        # Create the parameters for the Hugging Face language model's generator
        # method (that generates new tokens from a prompt).
        # https://huggingface.co/docs/transformers/v4.25.1/en/main_classes/text_generation#transformers.GenerationMixin.generate
        generate_kwargs = dict(
            do_sample=True,
            use_cache=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        self.generate_kwargs = {**generate_kwargs, **config.method.gen_kwargs}

        if config.method.gen_experience_kwargs is not None:
            self.generate_experience_kwargs = {**generate_kwargs, **config.method.gen_experience_kwargs}
        else:
            self.generate_experience_kwargs = None

        # Setup stats tracker
        self.running_moments = RunningMoments()
        self.ref_mean = self.config.method.ref_mean
        self.ref_std = self.config.method.ref_std

    def get_arch(self, config: TRLConfig):
        """Get the model"""
        model_class = AutoModelForCausalLMWithHydraValueHead
        if config.model.model_arch_type == "seq2seq":
            model_class = AutoModelForSeq2SeqLMWithHydraValueHead

        from_fn = model_class.from_pretrained
        # backward-compat: Try to create a randomly initialized architecture from a config
        if issubclass(type(config.model.model_path), transformers.PretrainedConfig):
            from_fn = model_class.from_config

        return from_fn(
            config.model.model_path,
            num_layers_unfrozen=config.model.num_layers_unfrozen,
            peft_config=self.config.model.peft_config,
        )

    def loss(self, batch: PPORLBatch):
        """Forward pass & loss

        Args:
            batch: Previous batch of episodes
        """
        # Move `batch` data to `accelerator` device
        query_tensors = batch.query_tensors.to(self.accelerator.device)
        response_tensors = batch.response_tensors.to(self.accelerator.device)
        old_logprobs = batch.logprobs.to(self.accelerator.device)
        old_values = batch.values.to(self.accelerator.device)
        old_rewards = batch.rewards.to(self.accelerator.device)
        response_length = old_rewards.shape[1]

        advantages, returns = self.config.method.get_advantages_and_returns(old_values, old_rewards, response_length)

        if self.config.model.model_arch_type == "seq2seq":
            input_ids = query_tensors
            decoder_input_ids = response_tensors
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long().to(self.accelerator.device)
            decoder_attention_mask = (
                decoder_input_ids.ne(self.tokenizer.pad_token_id).long().to(self.accelerator.device)
            )
            decoder_attention_mask[:, 0] = 1

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
            )

            logits = outputs.logits
            values_pred = outputs.value
            logprobs = logprobs_of_labels(logits[:, :-1, :], decoder_input_ids[:, 1:])
            mask = decoder_input_ids.ne(self.tokenizer.pad_token_id).long().to(self.accelerator.device)
            start = 0
            end = start + response_length
            logprobs, values_pred, mask = (
                logprobs[:, start:end],
                values_pred[:, start:end],
                mask[:, start + 1 : end + 1],
            )
        else:
            tokens = torch.cat((query_tensors, response_tensors), dim=1)
            attention_mask = tokens.not_equal(self.tokenizer.pad_token_id).long().to(tokens.device)
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            outputs = self.model(tokens, attention_mask, return_dict=True, position_ids=position_ids)
            logits = outputs.logits
            values_pred = outputs.value
            values_pred = values_pred[:, :-1]
            logprobs = logprobs_of_labels(logits[:, :-1, :], tokens[:, 1:])

            start = query_tensors.shape[1] - 1
            end = start + response_length
            logprobs, values_pred, mask = (
                logprobs[:, start:end],
                values_pred[:, start:end],
                attention_mask[:, start + 1 : end + 1],
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

        return loss, stats

    def setup_rollout_logging(self, config):
        # Make rollout logging dir for this run and store config
        exists = os.path.exists(config.train.rollout_logging_dir)
        isdir = os.path.isdir(config.train.rollout_logging_dir)
        assert exists and isdir

        self.run_id = f"run-{uuid.uuid4()}"
        self.rollout_logging_dir = os.path.join(config.train.rollout_logging_dir, self.run_id)
        os.mkdir(self.rollout_logging_dir)

        with open(os.path.join(self.rollout_logging_dir, "config.json"), "w") as f:
            f.write(json.dumps(config.to_dict(), indent=2))

    def post_epoch_callback(self):
        """Post epoch callback

        Clears the store and creates `num_rollouts` new episodes.
        """
        if self.log_rollouts:
            self.store.export_history(location=self.rollout_logging_dir)
        self.store.clear_history()
        # Collect more rollouts for training
        self.make_experience(self.config.method.num_rollouts, self.iter_count)

    def post_backward_callback(self):
        self.kl_ctl.update(self.mean_kl, n_steps=self.config.train.batch_size)

    def prepare_learning(self):
        eval_dataloader = self.eval_pipeline.create_loader(self.config.method.chunk_size)
        self.eval_dataloader = self.accelerator.prepare_data_loader(eval_dataloader)

        self.make_experience(self.config.method.num_rollouts)

        self.train_dataloader = self.store.create_loader(self.config.train.batch_size, shuffle=False)

        self.make_experience(self.config.method.num_rollouts)

        self.n_updates_per_batch = self.config.method.ppo_epochs
        self.total_steps = self.config.train.epochs * self.n_updates_per_batch * len(self.train_dataloader)
        self.total_steps = min(self.total_steps, self.config.train.total_steps)

    def add_prompt_pipeline(self, pipeline: PromptPipeline):
        """Add a prompt pipeline dataloader to a trainer instance for the `make_experience` stage"""
        prompt_dataloader = pipeline.create_loader(self.config.method.chunk_size, shuffle=False)
        prompt_dataloader = self.accelerator.prepare_data_loader(prompt_dataloader)
        self.prompt_iterator = infinite_dataloader(prompt_dataloader)

    def make_experience(self, num_rollouts: int = 1024, iter_count: int = 0):  # noqa:
        """Make experiences

        Takes `chunk_size` number of prompts from `prompt_iterator`, samples
        from the model and then computes the KL against a reference model. Finally it
        then appends PPOElements to trainer's `store`.

        Args:
            num_rollouts: Number of rollouts to generate
            iter_count: Total number of updates run (i.e. number of updates run for all batches & epochs)
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

        clock = Clock()
        ppo_rl_elements = []
        accumulated_stats = []

        # Require chunk_size * num_train_sequences divides num_rollouts
        assert num_rollouts % (self.config.method.chunk_size * self.config.method.num_train_sequences) == 0

        while len(ppo_rl_elements) < num_rollouts:
            stats = {}
            # Get next batch in prompt dataset
            batch: PromptBatch = next(self.prompt_iterator)

            rollout_generate_time = time()

            # Generate samples from the language model (similar to using HuggingFace `generate` method)
            samples = self.generate(
                batch["input_ids"],
                batch["attention_mask"],
                chunk_size=self.config.method.chunk_size,
                **self.generate_experience_kwargs,
            )
            stats["time/rollout_generate"] = time() - rollout_generate_time

            num_return_sequences = (
                self.generate_experience_kwargs["num_return_sequences"]
                if self.generate_experience_kwargs.get("num_return_sequences") is not None
                else 1
            )
            prompt_tensors = batch.input_ids.repeat_interleave(num_return_sequences, dim=0)
            device = samples.device

            prompt_sizes = torch.tensor([prompt_tensors.shape[1]] * len(prompt_tensors), device=device)
            padded_samples = self.accelerator.pad_across_processes(
                samples, dim=1, pad_index=self.tokenizer.eos_token_id, pad_first=False
            )
            padded_prompts = self.accelerator.pad_across_processes(
                prompt_tensors, dim=1, pad_index=self.tokenizer.eos_token_id, pad_first=False
            )
            gathered_samples = self.accelerator.gather(padded_samples)
            gathered_prompts = self.accelerator.gather(padded_prompts)
            gathered_prompt_sizes = self.accelerator.gather(prompt_sizes)
            metadata = gather_dict(
                {
                    k: self.repeat_interleave(v, num_return_sequences)
                    for k, v in batch.items()
                    if k != "input_ids" and k != "attention_mask"
                }
            )

            if self.accelerator.is_main_process:
                (
                    all_str_samples,
                    all_str_prompts,
                    all_str_outputs,
                    all_tok_samples,
                    all_tok_prompts,
                    all_tok_outputs,
                ) = self.decode(gathered_prompts, gathered_samples, gathered_prompt_sizes, append_eos_token=True)

                rollout_score_time = time()
                # reward_fn should return list of rewards at each token per sample
                # NOTE: all_scores[0][i] is the reward due to token (action) i in prompt + response (b/c of how kl is computed)
                # NOTE: reward_fn can optionally also compute the ref_logits.
                # In this case size will be [batch_size, response_length, 2]
                all_scores = self.reward_fn(
                    samples=all_str_samples,
                    prompts=all_str_prompts,
                    outputs=all_str_outputs,
                    tok_samples=all_tok_samples,
                    tok_prompts=all_tok_prompts,
                    tok_outputs=all_tok_outputs,
                    model_tok=self.tokenizer,
                    **metadata,
                )
                all_scores = [
                    torch.tensor(score, dtype=torch.float, device=device).view(
                        -1,
                    )
                    for score in all_scores
                ]
                # Pad -np.inf reward on the ends
                all_scores = pad_sequence(all_scores, batch_first=True, padding_value=-np.inf)
                max_len = torch.tensor(
                    len(all_scores[0]) / (1 + int(self.dist_ref_model)), dtype=torch.long, device=device
                )

                stats["time/rollout_score"] = time() - rollout_score_time

                all_scores = list(
                    all_scores.reshape(self.accelerator.num_processes, len(samples), max_len, -1).unbind()
                )
            else:
                all_scores = None
                max_len = torch.tensor(0, dtype=torch.long, device=device)

            if torch.distributed.is_initialized():
                torch.distributed.broadcast(max_len, 0)
                # Allocate extra space if scores include ref_logits
                scores = torch.empty((len(samples), max_len, 1 + int(self.dist_ref_model)), device=device)
                torch.distributed.scatter(scores, all_scores)
            else:
                scores = all_scores[0].clone().detach()

            # Remove ref_logits from scores if present
            if self.dist_ref_model:
                all_ref_logprobs = scores[:, :, 1]
                scores = scores[:, :, 0]
            else:
                all_ref_logprobs = None
                scores = scores.squeeze(-1)
            scores_mask = scores != -np.inf

            # Remove infs so mask can be used
            if self.config.method.cliprange_reward:
                scores = torch.clip(scores, -self.config.method.cliprange_reward, self.config.method.cliprange_reward)

            # Best-of-N Sampling.
            train_indices = self.get_topk_indices(
                input_tensor=scores_mask * scores,
                window_size=num_return_sequences,
                k=self.config.method.num_train_sequences,
                device=device,
            )
            scores = scores[train_indices]
            scores_mask = scores_mask[train_indices]
            samples = samples[train_indices]
            prompt_tensors = prompt_tensors[train_indices]
            if all_ref_logprobs is not None:
                all_ref_logprobs = all_ref_logprobs[train_indices]

            # store statistics of the initial rollout as reference
            if self.ref_mean is None:
                self.ref_mean, self.ref_std = (scores * scores_mask).sum(dim=1).mean(), (scores * scores_mask).sum(
                    dim=1
                ).std()
            all_scores_mean, all_scores_std = self.running_moments.update(scores, scores_mask)
            stats["rollout_scores/mean"] = all_scores_mean.item()
            stats["rollout_scores/std"] = all_scores_std.item()
            stats["rollout_scores/running_mean"] = self.running_moments.mean.item()
            stats["rollout_scores/running_std"] = self.running_moments.std.item()

            if self.config.method.scale_reward == "running":
                scores /= self.running_moments.std
            elif self.config.method.scale_reward == "ref":
                scores /= self.ref_std

            # Only use these samples, prompts, outputs to compute ppo stats
            _, _, _, tok_samples, tok_prompts, tok_outputs = self.decode(prompt_tensors, samples, append_eos_token=True)

            # Pad the sample outputs
            # outputs = self.tokenizer(str_outputs).input_ids
            # TODO: Why is this here? Should this be a sep token?
            if self.config.model.model_arch_type == "seq2seq":
                # add <pad> to the start of the output
                for i in range(len(tok_outputs)):
                    tok_outputs[i] = [self.tokenizer.pad_token_id] + outputs[i].tolist()

            tok_prompts = torch.stack(tok_prompts, dim=0)
            padded_tok_samples = pad_sequence(tok_samples, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            attention_mask = padded_tok_samples.not_equal(self.tokenizer.pad_token_id).long()

            if self.config.model.model_arch_type == "seq2seq":
                attention_mask = sample_outputs != self.tokenizer.pad_token_id
                start = 0
            else:
                # NOTE: -1 because kl[prompt_tensors.shape[1]] is kl of the second token in the response
                start = tok_prompts.shape[1] - 1

            # Precompute logprobs, values
            # TODO: Come back to seq2seq
            if self.config.model.model_arch_type == "seq2seq":
                attention_mask = batch.attention_mask.to(device)
                prompt_tensors = batch.input_ids.to(device)
                decoder_attention_mask = sample_outputs.not_equal(self.tokenizer.pad_token_id)
                decoder_attention_mask[:, 0] = 1
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=prompt_tensors,
                        attention_mask=attention_mask,
                        decoder_input_ids=sample_outputs,
                        decoder_attention_mask=decoder_attention_mask,
                    )
                    logits = outputs.logits
                    values = outputs.value
                    if hasattr(self.model, "frozen_head") or self.model.peft_type:
                        ref_logits = self.model.forward_hydra(
                            input_ids=prompt_tensors,
                            attention_mask=attention_mask,
                            decoder_input_ids=sample_outputs,
                            decoder_attention_mask=decoder_attention_mask,
                            return_dict=True,
                        ).logits
                    else:
                        ref_logits = self.ref_model(
                            input_ids=prompt_tensors,
                            attention_mask=attention_mask,
                            decoder_input_ids=sample_outputs,
                            decoder_attention_mask=decoder_attention_mask,
                            return_dict=True,
                        ).logits
            else:
                values_chunks = []
                log_probs_chunks = []
                ref_logprobs_chunks = []
                all_tokens = torch.cat((prompt_tensors.to(device), sample_outputs), dim=1)
                attention_mask = all_tokens.not_equal(self.tokenizer.pad_token_id).long().to(device)
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                all_tokens_chunks = torch.chunk(all_tokens, chunks=self.config.method.gen_chunk_size, dim=0)
                attention_mask_chunks = torch.chunk(attention_mask, chunks=self.config.method.gen_chunk_size, dim=0)
                position_ids_chunks = torch.chunk(position_ids, chunks=self.config.method.gen_chunk_size, dim=0)
                for all_tokens_chunk, attention_mask_chunk, position_ids_chunk in zip(
                    all_tokens_chunks, attention_mask_chunks, position_ids_chunks
                ):
                    all_tokens_chunk = all_tokens_chunk.to(device)
                    attention_mask_chunk = attention_mask_chunk.to(device)
                    position_ids_chunk = position_ids_chunk.to(device)
                    with torch.no_grad():
                        logits, *_, values = self.model(
                            all_tokens_chunk,
                            attention_mask=attention_mask_chunk,
                            position_ids=position_ids_chunk,
                        )
                        # If all_ref_logits is not None they have already been generated during call to reward_fn
                        if all_ref_logprobs is None:
                            if hasattr(self.model, "frozen_head"):
                                ref_logits = self.model.forward_hydra(
                                    all_tokens_chunk,
                                    attention_mask=attention_mask_chunk,
                                    position_ids=position_ids_chunk,
                                    return_dict=True,
                                ).logits
                            elif hasattr(self, "ref_model"):
                                ref_logits = self.ref_model(
                                    all_tokens_chunk,
                                    attention_mask=attention_mask_chunk,
                                    position_ids=position_ids_chunk,
                                    return_dict=True,
                                ).logits
                                ref_logits = ref_logits.to(device)
                            # If no ref model is provided then we compute no kl penalty
                            else:
                                ref_logits = logits.clone()

                    if self.config.model.model_arch_type == "seq2seq":
                        logprobs = logprobs_of_labels(logits[:, start:-1, :], all_tokens_chunk[:, start + 1 :])
                        ref_logprobs = logprobs_of_labels(ref_logits[:, start:-1, :], all_tokens_chunk[:, start + 1 :])
                    else:
                        # NOTE: logprob[i] is (log)prob at which all_token[i+1] was sampled
                        # So need to index at start = prompt_tensors.shape[1] - 1 which is
                        # the logprob corresponding to the first sampled token
                        # Indexing ends at -1 because the last logprob corresponds to an unsampled token
                        logprobs = logprobs_of_labels(logits[:, start:-1, :], all_tokens_chunk[:, start + 1 :])
                        if all_ref_logprobs is None:
                            ref_logprobs = logprobs_of_labels(
                                ref_logits[:, start:-1, :], all_tokens_chunk[:, start + 1 :]
                            )

                    values_chunks.append(values.cpu())
                    log_probs_chunks.append(logprobs.cpu())
                    if all_ref_logprobs is None:
                        ref_logprobs_chunks.append(ref_logprobs.cpu())

            # Remove values before v[start] (this is the value of the state before any tokens are sampled)
            # and remove the last value v[-1] (this is a terminal state after all tokens have been generated with value 0)
            values = torch.cat(values_chunks, dim=0)[:, start:-1]
            logprobs = torch.cat(log_probs_chunks, dim=0)
            attention_mask = attention_mask[:, start:].cpu()

            if all_ref_logprobs is None:
                ref_logprobs = torch.cat(ref_logprobs_chunks, dim=0)
            # all_ref_logprobs returned from reward already has prompt prefix removed
            else:
                # Remove (some) padding from distributed communication
                # So arithmetic with logprobs can be done
                ref_logprobs = all_ref_logprobs[:, : logprobs.shape[1]].cpu()

            # Estimate the KL divergence between the model and reference model
            # NOTE: nan is interfering with kl estimates since 0 * nan = 0
            # Convert inf padding terms in ref_logprobs to number removable with attention mask mult
            log_ratio = (logprobs - torch.nan_to_num(ref_logprobs)) * attention_mask[:, :-1]
            kl = log_ratio.exp() - 1 - log_ratio
            mean_kl_per_token = kl.mean()
            mean_kl = kl.sum(1).mean()
            kl_penalties = self.kl_ctl.value * -log_ratio.cpu()

            n_samples = padded_tok_samples.shape[0]
            rollout_count = 0

            # Get the logprobs and values, for tokens that are not padding,
            # from the end of the prompt up to the <eos> token, while also including the latter
            # (these are taken from the student model and not the reference model)
            # NOTE: Why are we summing including a token from the prompt?
            # In our case it's ok because we then subtract -1 from resulting end index
            ends = attention_mask.sum(1) + 1
            for sample_idx in range(n_samples):
                value = values[sample_idx, : ends[sample_idx] - 1]
                logprob = logprobs[sample_idx, : ends[sample_idx] - 1]
                kl_penalty = kl_penalties[sample_idx, : ends[sample_idx] - 1]
                query_tensor = tok_prompts[sample_idx]
                response_tensor = tok_outputs[sample_idx]
                if (
                    len(value) != len(logprob)
                    or len(logprob) != len(kl_penalty)
                    or len(kl_penalty) != len(response_tensor)
                ):
                    raise ValueError(
                        f"Length mismatch between value, logprob, kl, and response_tensor:\n\
                                        Value: {value.shape}, {value}\n\
                                        Logprob: {logprob.shape}, {logprob}\n\
                                        KL: {kl_penalty.shape}, {kl_penalty}\n\
                                        Response: {response_tensor.shape}, {response_tensor}, \
                                        {self.tokenizer.decode(response_tensor)}\n"
                    )

                # Then add in rewards
                if scores.shape[1] == 1:
                    # NOTE: Final reward given at EOS token following HHH practice
                    score = scores[sample_idx][0].cpu()
                    kl_penalty[-1] += score
                    rewards = kl_penalty
                else:
                    score = scores[sample_idx]
                    score_right_padding = torch.sum(scores_mask[sample_idx])
                    score = score[:score_right_padding].cpu()
                    if len(score) != len(kl_penalty):
                        raise ValueError(
                            f"Length mismatch between score and kl penalty:\n\
                                            Logprob: {logprob.shape}, {logprob}\n\
                                            kl_penalty: {kl_penalty.shape}, {kl_penalty}\n\
                                            Score: {score.shape}, {score}"
                        )
                    rewards = kl_penalty + score

                if kl_penalty.isnan().any() or score.isnan().any():
                    raise ValueError(
                        f"nan in tensor:\n\
                                        KL: {kl_penalty}\n\
                                        Score: {score}\n\
                                        logprob: {logprob}\n\
                                        ref logprob: {ref_logprobs[sample_idx][:ends[sample_idx]-1]}\n\
                                        mask: {attention_mask[sample_idx]}\n\
                                        kl ctl: {self.kl_ctl.value}"
                    )

                ppo_rl_elements.append(
                    PPORLElement(
                        query_tensor=query_tensor,
                        response_tensor=response_tensor,
                        logprobs=logprob,
                        values=value,
                        rewards=rewards,
                    )
                )

                rollout_count += 1

            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(mean_kl.to(self.accelerator.device), torch.distributed.ReduceOp.AVG)

            stats["time/rollout_time"] = clock.tick()
            stats["policy/sqrt_kl"] = torch.sqrt(mean_kl).item()
            stats["policy/kl_per_token"] = torch.sqrt(mean_kl_per_token).item()
            accumulated_stats.append(stats)

            tbar.set_description(f"[rollout {len(ppo_rl_elements)} / {num_rollouts}]")
            tbar.update(min(rollout_count, num_rollouts))
        tbar.close()

        stats = {k: sum([xs[k] for xs in accumulated_stats]) / len(accumulated_stats) for k in stats}
        stats["kl_ctl_value"] = self.kl_ctl.value
        self.mean_kl = stats["policy/sqrt_kl"] ** 2
        self.accelerator.log(stats, step=iter_count)

        # Push samples and rewards to trainer's rollout storage
        self.push_to_store(ppo_rl_elements)

    @staticmethod
    def get_topk_indices(input_tensor, window_size: int, k: int, device):
        # Sum the scores along dim 1
        input_tensor = input_tensor.sum(1).unsqueeze(1)
        # Use unfold to create the sliding windows
        unfolded = input_tensor.unfold(0, window_size, window_size)
        # Find the topk values and indices along the unfolded dimension
        _, indices = torch.topk(unfolded, k, dim=2)
        # Adjust indices to be relative to original tensor
        indices = indices.squeeze(1) + torch.arange(0, input_tensor.size(0) - window_size + 1, window_size).to(
            device
        ).unsqueeze(1)
        return indices.reshape(-1)
