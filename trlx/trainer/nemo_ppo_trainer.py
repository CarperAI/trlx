import os
import sys
from logging import getLogger
from math import ceil
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, List, Optional, Union, cast

import torch
from apex.transformer import parallel_state
from omegaconf.omegaconf import OmegaConf
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

import wandb
from trlx.data.accelerate_base_datatypes import PromptBatch
from trlx.data.configs import TRLConfig
from trlx.data.ilql_types import flatten_dataclass
from trlx.data.ppo_types import PPORLBatch, PPORLElement
from trlx.models.modeling_nemo_ppo import PPOGPT
from trlx.models.modeling_ppo import AdaptiveKLController, FixedKLController, PPOConfig
from trlx.pipeline.offline_pipeline import PromptPipeline
from trlx.pipeline.ppo_pipeline import ppo_collate_fn
from trlx.trainer import BaseRLTrainer, register_trainer
from trlx.trainer.nemo_ilql_trainer import megatron_trainer
from trlx.utils import get_git_tag, infinite_dataloader

logging = getLogger(__name__)


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

        if metric_fn is None:

            def metric_fn(*args, **kwargs):
                return {"reward": self.reward_fn(*args, **kwargs)}

            self.metric_fn = metric_fn

        if not isinstance(config.method, PPOConfig):
            raise ValueError("config.method must be PPOConfig")

        self.ppo_config: PPOConfig = cast(PPOConfig, config.method)
        if isinstance(megatron_cfg, str):
            cfg_path = Path(__file__).parent.parent.parent / "configs" / "nemo_configs" / megatron_cfg
            logging.info(f"Loading NeMo config from {cfg_path=}")
            megatron_cfg = OmegaConf.load(cfg_path)
        elif megatron_cfg is None:
            raise ValueError("megatron_cfg must be a path or a config")

        megatron_cfg.trlx = config.to_dict()

        megatron_cfg.model.global_batch_size = config.train.batch_size
        megatron_cfg.model.seed = config.train.seed

        world_size = megatron_cfg.trainer.num_nodes * megatron_cfg.trainer.devices
        dp_world = world_size // (
            megatron_cfg.model.tensor_model_parallel_size * megatron_cfg.model.pipeline_model_parallel_size
        )
        rank_batch_size = megatron_cfg.model.global_batch_size // dp_world

        if (self.ppo_config.chunk_size % rank_batch_size) != 0:
            self.ppo_config.chunk_size = rank_batch_size * ceil(self.ppo_config.chunk_size / rank_batch_size)
            print("Rounding chunk size to", self.ppo_config.chunk_size)

        train_samples = dp_world * self.ppo_config.num_rollouts * self.ppo_config.ppo_epochs
        # Disable validation within nemo, run it ourselves
        self.limit_val_batches = megatron_cfg.trainer.limit_val_batches
        self.val_check_interval = config.train.eval_interval

        megatron_cfg.trainer.limit_val_batches = 0.0
        megatron_cfg.trainer.val_check_interval = None
        self.train_samples = train_samples
        megatron_cfg.trainer.max_steps = config.train.epochs * (train_samples // megatron_cfg.model.global_batch_size)

        self.trainer = megatron_trainer(megatron_cfg)

        self.model = PPOGPT(
            ppo_config=self.ppo_config,
            cfg=megatron_cfg.model,
            trainer=self.trainer,
            metric_fn=self.metric_fn,
            stop_sequences=self.stop_sequences,
            num_layers_unfrozen=config.model.num_layers_unfrozen,
        )
        self.megatron_cfg = megatron_cfg

        if pretrained_model is not None:
            self.model.load_from_pretrained(pretrained_model)

        self.batch_size = megatron_cfg.model.global_batch_size
        self.tokenizer = self.model.tokenizer.tokenizer
        self.tokenizer.truncation_side = config.tokenizer.truncation_side
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_length = megatron_cfg.model.encoder_seq_length

        if self.ppo_config.target is not None:
            self.kl_ctl = AdaptiveKLController(
                self.ppo_config.init_kl_coef, self.ppo_config.target, self.ppo_config.horizon
            )
        else:
            self.kl_ctl = FixedKLController(self.ppo_config.init_kl_coef)

    def add_prompt_pipeline(self, pipeline: PromptPipeline):
        self.prompt_pipeline = pipeline

    def make_experience(self, prompt_iterator: Iterator, num_rollouts: int = 1024, dp_world: int = 1):
        ppo_rl_elements = []

        device = torch.device("cuda")

        if torch.distributed.get_rank() == 0:
            tbar = tqdm(total=num_rollouts * dp_world, desc="Generating experience")

        self.model.offload_reference_model()

        inputs_samples = []

        while num_rollouts > 0:
            batch: PromptBatch = next(prompt_iterator)
            lengths = batch.attention_mask.sum(dim=1)

            max_new_tokens = self.ppo_config.gen_kwargs.get("max_new_tokens", 128)
            if self.ppo_config.gen_experience_kwargs is not None:
                max_new_tokens = self.ppo_config.gen_experience_kwargs.get("max_new_tokens", max_new_tokens)

            input_ids = batch["input_ids"].to(device)
            lengths = lengths.to(device)
            pad_to = lengths.max().item() + max_new_tokens

            pad_to = ceil(pad_to / 8) * 8
            input_ids = torch.nn.functional.pad(
                input_ids, (0, pad_to - input_ids.shape[1]), value=self.tokenizer.pad_token_id
            )

            samples = self.model.generate((input_ids, lengths), dict(max_length=max_new_tokens, min_length=1))

            inputs_samples.append((input_ids.cpu(), lengths.cpu(), samples))

            if torch.distributed.get_rank() == 0:
                tbar.update(len(input_ids) * dp_world)
            num_rollouts = num_rollouts - len(input_ids)

        self.model.free_kv_cache()

        all_sents = [sentence for _, _, samples in inputs_samples for sentence in samples["sentences"]]
        all_prompts = [prompt for _, _, samples in inputs_samples for prompt in samples["prompts"]]
        all_responses = [response for _, _, samples in inputs_samples for response in samples["responses"]]
        scores = torch.tensor(
            self.reward_fn(samples=all_sents, prompts=all_prompts, outputs=all_responses), device=device
        )
        scores = torch.clip(scores, -self.ppo_config.cliprange_reward, self.ppo_config.cliprange_reward)
        chunk_size = self.ppo_config.chunk_size
        scores = [scores[i : i + chunk_size] for i in range(0, len(scores), chunk_size)]

        if torch.distributed.get_rank() == 0:
            tbar.close()
            lps_tbar = tqdm(total=num_rollouts * dp_world, desc="Computing logprobs")

        for (_, lengths, samples), scores in zip(inputs_samples, scores):
            output_tokens = samples["token_ids"]

            max_length = max(len(x) for x in output_tokens)

            output_tokens = [
                x + [self.tokenizer.pad_token_id] * ((ceil(max_length / 8) * 8) - len(x)) for x in output_tokens
            ]

            all_tokens = torch.tensor(output_tokens, device=device)
            attention_mask = all_tokens.ne(self.tokenizer.pad_token_id).long().to(device)

            model_output = self.model.infer_logprobs_and_values(all_tokens, attention_mask)

            # Model output is None on intermediate pipeline stages
            if model_output is None:
                continue

            logprobs, ref_logprobs, values = (
                model_output["logprobs"],
                model_output["ref_logprobs"],
                model_output["values"],
            )

            query_responses = [(t[:l], t[l:]) for t, l in zip(output_tokens, lengths)]
            query_tokens, response_tokens = zip(*query_responses)

            query_tensors = [torch.tensor(t, device="cpu") for t in query_tokens]
            response_tensors = [torch.tensor(t, device="cpu") for t in response_tokens]
            logprobs = logprobs.cpu()
            ref_logprobs = ref_logprobs.cpu()

            masks = attention_mask.cpu()
            log_ratio = (logprobs - ref_logprobs) * masks[:, :-1]

            for query_tensor, response_tensor, logps, vs, kl_penalty, score, start, mask in zip(
                query_tensors, response_tensors, logprobs, values, log_ratio, scores, lengths, masks
            ):
                response_end = mask[start:].sum()
                end = start + response_end
                rewards = self.kl_ctl.value * -kl_penalty[start:end].cpu()
                rewards[-1] += score.cpu()

                assert (
                    vs[:-1][start:end].shape[0] == rewards.shape[0]
                ), f"{vs[start:end].shape} != {rewards.shape} {kl_penalty[start:end].shape=} {values.shape=} {start=} {end=}"

                ppo_rl_elements.append(
                    PPORLElement(
                        query_tensor=query_tensor,
                        response_tensor=response_tensor[: response_end + 1],
                        logprobs=logps[start:end],
                        values=vs[:-1][start:end],
                        rewards=rewards,
                    )
                )

            if torch.distributed.get_rank() == 0:
                lps_tbar.update(len(query_tensors) * dp_world)

        return ppo_rl_elements

    def learn(self):  # noqa: C901
        def add_special_token_ids(input_ids: List[int], add_bos: bool, add_eos: bool):
            if add_bos:
                input_ids = [self.tokenizer.bos_token_id] + input_ids
            if add_eos:
                input_ids = input_ids + [self.tokenizer.eos_token_id]
            if len(input_ids) > self.max_length:
                input_ids = input_ids[: self.max_length]
            return input_ids

        def collate_fn(elems: Iterable[PPORLElement]):
            batch = ppo_collate_fn("left", self.tokenizer.eos_token_id, elems)
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

        global_rank = self.trainer.global_rank
        world_size = self.trainer.world_size
        if global_rank == 0:
            script = os.path.basename(sys.argv[0]).rsplit(".", 1)[0]
            cfg_name = self.config.train.trainer_kwargs["megatron_cfg"]
            if isinstance(cfg_name, str):
                config_name = cfg_name.rsplit(".", 1)[0]
            else:
                config_name = cfg_name.get("name", "unknown")
            branch = get_git_tag()[0]
            name = f"{script}/{config_name}/{world_size}gpus:{branch}"
            wandb.init(
                name=name,
                project=self.config.train.project_name,
                mode="online" if self.config.train.tracker == "wandb" else "disabled",
                group=self.config.train.group_name,
                entity=self.config.train.entity_name,
                config=OmegaConf.to_container(self.megatron_cfg, resolve=True),
            )
            print(OmegaConf.to_container(self.megatron_cfg, resolve=True))

        def dummy():
            return

        if self.trainer.strategy.launcher is not None:
            self.trainer.strategy.launcher.launch(dummy, trainer=self.trainer)
        self.trainer.strategy.setup_environment()

        dp_world = parallel_state.get_data_parallel_world_size()
        dp_rank = parallel_state.get_data_parallel_rank()

        sampler = DistributedSampler(
            self.prompt_pipeline, num_replicas=dp_world, rank=dp_rank, shuffle=True, drop_last=True
        )
        prompt_dataloader = infinite_dataloader(
            self.prompt_pipeline.create_loader(self.ppo_config.chunk_size, shuffle=False, sampler=sampler),
            sampler=sampler,
        )

        if self.model.cfg.get("transformer_engine", False):
            self.model.setup_transformer_engine_tp_groups()

        self.model.setup()
        self.trainer.strategy._lightning_module = self.model
        _, schedulers = self.model.configure_optimizers()
        scheduler = schedulers[0]["scheduler"]

        prompt_iter = iter(prompt_dataloader)
        local_batch_idx = 0

        rank_batch_size = self.batch_size // dp_world
        total_batches = self.config.train.epochs * (self.train_samples // self.batch_size)
        if global_rank == 0:
            train_tbar = tqdm(desc="Training", total=total_batches)

        metrics = None
        best_metric = None

        for epoch in range(self.config.train.epochs):
            ppo_rl_rollouts = self.make_experience(
                prompt_iter,
                num_rollouts=self.ppo_config.num_rollouts,
                dp_world=dp_world,
            )

            dataloader = DataLoader(ppo_rl_rollouts, batch_size=rank_batch_size, collate_fn=collate_fn, drop_last=True)
            self.model.offload_reference_model()
            self.model.train()

            for batch in dataloader:
                for _ in range(self.ppo_config.ppo_epochs):
                    self.model.training_step(batch, local_batch_idx)
                    if global_rank == 0:
                        train_tbar.update(1)
                    self.model._optimizer.step()
                    scheduler.step()

                    if local_batch_idx % self.val_check_interval == 0:
                        mbs = self.ppo_config.chunk_size
                        if global_rank == 0:

                            def tbar(x):
                                return tqdm(x, desc="Validation", total=len(self.eval_pipeline) // mbs)

                        else:

                            def tbar(x):
                                return x

                        val_loader = DataLoader(self.eval_pipeline, batch_size=mbs, collate_fn=generate_collate)
                        val_stats = [
                            self.model.validation_step(val_batch, local_batch_idx) for val_batch in tbar(val_loader)
                        ]
                        metrics = self.model.validation_epoch_end(val_stats, local_batch_idx)

                    if (local_batch_idx % self.config.train.checkpoint_interval) == 0:
                        if self.config.train.save_best and metrics is not None:
                            if best_metric is None or metrics["val_metrics/reward"] > best_metric:
                                best_metric = metrics["val_metrics/reward"]
                                self.model.save_pretrained(self.config.train.checkpoint_dir)
                        else:
                            self.model.save_pretrained(self.config.train.checkpoint_dir)

                    local_batch_idx += 1

            if local_batch_idx > self.config.train.total_steps:
                break

        mbs = self.ppo_config.chunk_size
        val_loader = DataLoader(self.eval_pipeline, batch_size=mbs, collate_fn=generate_collate)
        val_stats = [self.model.validation_step(val_batch, local_batch_idx) for val_batch in val_loader]
        self.model.validation_epoch_end(val_stats, local_batch_idx)
