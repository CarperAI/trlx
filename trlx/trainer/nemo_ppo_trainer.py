from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union, cast

import torch
import transformers
from nemo.utils import logging
from omegaconf.omegaconf import OmegaConf

from trlx.data.configs import TRLConfig
from trlx.data.ppo_types import PPORLElement
from trlx.models.modeling_nemo_ppo import PPOGPT
from trlx.models.modeling_ppo import PPOConfig
from trlx.trainer import BaseRLTrainer, register_trainer
from trlx.trainer.nemo_ilql_trainer import ShuffledCyclicSequence, megatron_trainer
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

        self.trainer = megatron_trainer(megatron_cfg)
        self.model = PPOGPT(
            sft_config=self.sft_config,
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

    def make_experience(self, num_rollouts: int = 1024, iter_count: int = 0):
        ppo_rl_elements = []
        stats = {}

        device = self.model.device
        group = parallel_state.get_data_parallel_group()

        while num_rollouts > 0:
            batch: PromptBatch = next(self.prompt_iterator)

            samples = self.model.generate(batch["input_ids"], batch["attention_mask"])

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

            num_elements_added = torch.tensor(0, device=device)
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

    def learn(self):
        def add_special_token_ids(input_ids: List[int], add_bos: bool, add_eos: bool):
            if add_bos:
                input_ids = [self.tokenizer.bos_token_id] + input_ids
            if add_eos:
                input_ids = input_ids + [self.tokenizer.eos_token_id]
            if len(input_ids) > self.max_length:
                input_ids = input_ids[: self.max_length]
            return input_ids

        def pad_batch_and_build_loss_mask(
            input_ids: List[List[int]], batch_max_length: int
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            batch_loss_masks = []
            padded_input_ids = []
            for ids in input_ids:
                input_length = len(ids)
                padding_length = batch_max_length - input_length
                padded_input_ids.append(ids + [self.tokenizer.pad_token_id] * padding_length)
                loss_mask = [1.0] * input_length + [0.0] * padding_length
                batch_loss_masks.append(torch.tensor(loss_mask, dtype=torch.float))
            padded_input_ids = torch.as_tensor(padded_input_ids, dtype=torch.long)
            batch_loss_masks = torch.stack(batch_loss_masks, dim=0)
            # NOTE: Un-build the loss mask if we're not going to mask eod tokens
            if self.model.cfg.data.get("eod_mask_loss", False) is False:
                loss_mask = torch.ones_like(loss_mask)
            return padded_input_ids, batch_loss_masks

        def collate_fn(elems: List[transformers.BatchEncoding]):
            context_tokens = [
                add_special_token_ids(
                    e["input_ids"],
                    self.model.cfg.data.get("add_bos", False),
                    self.model.cfg.data.get("add_eos", True),
                )
                for e in elems
            ]
            input_ids, loss_mask = pad_batch_and_build_loss_mask(context_tokens, self.max_length)
            return input_ids, loss_mask

        train_samples = self.model.cfg.global_batch_size * self.trainer.max_steps
        train_dataset = ShuffledCyclicSequence(train_samples, self.store, self.config.train.seed)
        self.model.set_train_dataset(train_dataset, collate_fn=collate_fn)

        def eval_collate(elems):
            context_tokens = [
                add_special_token_ids(e["input_ids"], add_bos=self.model.cfg.data.get("add_bos", False), add_eos=False)
                for e in elems
            ]
            max_new_tokens = self.sft_config.gen_kwargs.get("max_new_tokens", 64)

            context_lengths = [len(x) for x in context_tokens]
            max_context = max(context_lengths)

            pad_id = self.tokenizer.eos_token_id
            padded = [x + [pad_id] * (max_context + max_new_tokens - len(x)) for x in context_tokens]

            return [
                torch.as_tensor(padded, device="cpu"),
                torch.as_tensor(context_lengths, device="cpu"),
            ]

        max_train_steps = self.trainer.max_steps
        eval_iters = (max_train_steps // self.trainer.val_check_interval + 1) * self.trainer.limit_val_batches
        eval_samples = eval_iters * self.model.cfg.global_batch_size

        eval_dataset = ShuffledCyclicSequence(
            new_length=eval_samples,
            data=self.eval_pipeline,
            seed=self.config.train.seed,
        )

        self.model.set_valid_dataset(eval_dataset, collate_fn=eval_collate)

        torch.set_float32_matmul_precision("medium")
        self.trainer.fit(self.model)
