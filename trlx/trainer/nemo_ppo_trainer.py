from math import ceil
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, List, Optional, Tuple, Union, cast

import torch
import transformers
from apex.transformer import parallel_state
from nemo.utils import logging
from omegaconf.omegaconf import OmegaConf
from torch.utils.data import DistributedSampler

from trlx.data.accelerate_base_datatypes import PromptBatch
from trlx.data.configs import TRLConfig
from trlx.data.ilql_types import flatten_dataclass
from trlx.data.ppo_types import PPORLBatch, PPORLElement
from trlx.models.modeling_nemo_ppo import PPOGPT
from trlx.models.modeling_ppo import PPOConfig
from trlx.pipeline.offline_pipeline import PromptPipeline
from trlx.pipeline.ppo_pipeline import ppo_collate_fn
from trlx.trainer import BaseRLTrainer, register_trainer
from trlx.trainer.nemo_ilql_trainer import ShuffledCyclicSequence, megatron_trainer
from trlx.utils import infinite_dataloader
from trlx.utils.modeling import logprobs_of_labels


@register_trainer
class NeMoPPOTrainer(BaseRLTrainer):
    def __init__(
        self,
        config: TRLConfig,
        metric_fn: Optional[Callable[[List[str]], Any]] = None,
        megatron_cfg: Optional[Union[str, dict]] = None,
        pretrained_model: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(config, metric_fn=metric_fn, **kwargs)

        if not isinstance(config.method, PPOConfig):
            raise ValueError("config.method must be PPOConfig")

        self.ppo_config: PPOConfig = cast(PPOConfig, config.method)
        if isinstance(megatron_cfg, str):
            cfg_path = Path(__file__).parent.parent.parent / "configs" / "nemo_configs" / megatron_cfg
            logging.info(f"Loading NeMo config from {cfg_path=}")
            megatron_cfg = OmegaConf.load(cfg_path)
        elif megatron_cfg is None:
            raise ValueError("megatron_cfg must be a path or a config")

        train_samples = self.ppo_config.num_rollouts * self.ppo_config.ppo_epochs
        if (train_samples % megatron_cfg.model.global_batch_size) != 0:
            train_samples = (
                ceil(train_samples / megatron_cfg.model.global_batch_size) * megatron_cfg.model.global_batch_size
            )
            print("Rounding up (num_rollouts * ppo_epochs) to", train_samples)

        megatron_cfg.model.train_steps = train_samples // megatron_cfg.model.global_batch_size

        self.train_samples = train_samples

        self.trainer = megatron_trainer(megatron_cfg)
        self.model = PPOGPT(
            ppo_config=self.ppo_config,
            cfg=megatron_cfg.model,
            trainer=self.trainer,
            metric_fn=self.metric_fn,
        )

        if pretrained_model is not None:
            self.model.load_from_pretrained(pretrained_model)

        self.batch_size = megatron_cfg.model.global_batch_size
        self.tokenizer = self.model.tokenizer.tokenizer
        self.tokenizer.truncation_side = config.tokenizer.truncation_side
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = megatron_cfg.model.encoder_seq_length

    def add_prompt_pipeline(self, pipeline: PromptPipeline):
        self.prompt_pipeline = pipeline

    def make_experience(self, prompt_iterator: Iterator, num_rollouts: int = 1024, iter_count: int = 0):
        ppo_rl_elements = []
        stats = {}

        device = self.model.device
        group = parallel_state.get_data_parallel_group()

        while num_rollouts > 0:
            batch: PromptBatch = next(prompt_iterator)

            lengths = batch.attention_mask.sum(dim=1)

            max_new_tokens = self.ppo_config.method.gen_kwargs.get("max_new_tokens", 128)
            if self.ppo_config.method.gen_experience_kwargs is not None:
                max_new_tokens = self.ppo_config.method.gen_experience_kwargs.get("max_new_tokens", max_new_tokens)

            samples = self.model.generate((batch["input_ids"], lengths), dict(max_length=max_new_tokens, min_length=1))

            scores = torch.tensor(self.reward_fn(samples["sentences"]), device=device)

            scores = torch.clip(scores, -self.config.method.cliprange_reward, self.config.method.cliprange_reward)

            all_tokens = torch.cat((batch["input_ids"], samples["tokens"]), dim=1)
            attention_mask = all_tokens.ne(self.tokenizer.pad_token_id).long().to(device)

            model_output = self.model.infer_logits_and_values(all_tokens, attention_mask)

            # Model output is None on intermediate pipeline stages
            if model_output is None:
                num_elements_added = torch.tensor(0, device=device)
                torch.distributed.all_reduce(num_elements_added, op=torch.distributed.ReduceOp.SUM, group=group)
                num_elements_added = num_elements_added.item()
                num_rollouts -= num_elements_added

            logits, ref_logits, values = model_output["logits"], model_output["ref_logits"], model_output["values"]

            logprobs = logprobs_of_labels(logits[:, :-1, :], all_tokens[:, 1:])
            ref_logprobs = logprobs_of_labels(ref_logits[:, :-1, :], all_tokens[:, 1:])

            start = batch["input_ids"].shape[1] - 1

            logprobs = logprobs.cpu()
            ref_logprobs = ref_logprobs.cpu()
            prompt_tensors = batch["input_ids"].cpu()
            sample_outputs = samples["tokens"].cpu()
            values = values.cpu()[:, :-1]

            n_samples = values.shape[0]

            ends = start + attention_mask[:, start:].sum(dim=1)
            all_values = [values[ix, start : ends[ix]] for ix in range(n_samples)]
            all_logprobs = [logprobs[ix, start : ends[ix]] for ix in range(n_samples)]

            log_ratio = (logprobs - ref_logprobs) * attention_mask[:, :-1]

            all_kl_penalty = self.kl_ctl.value * -log_ratio.cpu()
            all_kl_penalty = [xs[start : ends[ix]] for ix, xs in enumerate(kl_penalty)]

            num_elements_added = torch.tensor(0, device=device, requires_grad=False)
            for query_tensor, response_tensor, logprobs, values, kl_penalty, score in zip(
                prompt_tensors, sample_outputs, all_logprobs, all_values, all_kl_penalty, scores
            ):
                rewards = kl_penalty[:]
                rewards[-1] += score.cpu()
                ppo_rl_elements.append(
                    PPORLElement(
                        query_tensor=query_tensor,
                        response_tensor=response_tensor,
                        logprobs=logprobs,
                        values=values,
                        rewards=rewards,
                    )
                )
                num_elements_added += 1

            torch.distributed.all_reduce(num_elements_added, op=torch.distributed.ReduceOp.SUM, group=group)
            num_elements_added = num_elements_added.item()
            num_rollouts -= num_elements_added

        return ppo_rl_elements, stats

    def learn(self):
        def add_special_token_ids(input_ids: List[int], add_bos: bool, add_eos: bool):
            if add_bos:
                input_ids = [self.tokenizer.bos_token_id] + input_ids
            if add_eos:
                input_ids = input_ids + [self.tokenizer.eos_token_id]
            if len(input_ids) > self.max_length:
                input_ids = input_ids[: self.max_length]
            return input_ids

        def collate_fn(elems: Iterable[PPORLElement]):
            batch = ppo_collate_fn(elems)
            return flatten_dataclass(PPORLBatch)(batch)

        def generate_collate(elems):
            context_tokens = [
                add_special_token_ids(e["input_ids"], add_bos=self.model.cfg.data.get("add_bos", False), add_eos=False)
                for e in elems
            ]
            max_new_tokens = self.ppo_config.gen_kwargs.get("max_new_tokens", 64)

            context_lengths = [len(x) for x in context_tokens]
            max_context = max(context_lengths)

            pad_id = self.tokenizer.eos_token_id
            padded = [x + [pad_id] * (max_context + max_new_tokens - len(x)) for x in context_tokens]

            return [
                torch.as_tensor(padded, device="cpu"),
                torch.as_tensor(context_lengths, device="cpu"),
            ]

        parallel_state.initialize_model_parallel(
            self.model.cfg.tensor_model_parallel_size, self.model.cfg.pipeline_model_parallel_size, None
        )

        prompt_dataset = ShuffledCyclicSequence(
            new_length=self.config.train.total_steps * self.batch_size,
            data=self.prompt_pipeline,
            seed=self.config.train.seed,
        )

        sampler = DistributedSampler(self.prompt_pipeline, rank=parallel_state.get_data_parallel_rank(), shuffle=True)
        prompt_dataloader = DataLoader(
            self.prompt_pipeline,
            batch_size=self.batch_size,
            collate_fn=generate_collate,
            num_workers=0,
            pin_memory=True,
            sampler=sampler,
        )

        prompt_dataloader = infinite_dataloader(prompt_dataloader, sampler=sampler)
        for global_step in range(config.train.total_steps, self.ppo_config.num_rollouts * self.ppo_config.ppo_epochs):
            ppo_rl_rollouts, stats = self.make_experience(
                iter(prompt_dataloader), num_rollouts=self.ppo_config.num_rollouts * self.ppo_config.ppo_epochs
            )

            rollout_dataset = ShuffledCyclicSequence(
                new_length=self.train_samples,
                data=ppo_rl_rollouts,
                seed=self.config.train.seed,
            )
            self.model.set_train_dataset(rollout_dataset, collate_fn=collate_fn)
            self.trainer.fit(self.model)

        eval_iters = (max_train_steps // self.trainer.val_check_interval + 1) * self.trainer.limit_val_batches
        eval_samples = eval_iters * self.model.cfg.global_batch_size

        eval_dataset = ShuffledCyclicSequence(
            new_length=eval_samples,
            data=self.eval_pipeline,
            seed=self.config.train.seed,
        )

        self.model.set_valid_dataset(eval_dataset, collate_fn=eval_collate)

        torch.set_float32_matmul_precision("medium")
        self.trainer.validate(self.model)
