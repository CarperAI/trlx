import gc
import os
import sys
from logging import getLogger
from math import ceil
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, List, Optional, Union, cast

import torch
import wandb
from apex.transformer import parallel_state
from omegaconf.omegaconf import OmegaConf
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import LlamaTokenizer

from trlx.data.accelerate_base_datatypes import PromptBatch
from trlx.data.configs import TRLConfig
from trlx.data.ilql_types import flatten_dataclass
from trlx.data.ppo_types import PPORLBatch, PPORLElement
from trlx.models.modeling_nemo_ppo import PPOGPT
from trlx.models.modeling_ppo import FixedKLController, PPOConfig
from trlx.pipeline.offline_pipeline import PromptPipeline
from trlx.pipeline.ppo_pipeline import ppo_collate_fn
from trlx.trainer import BaseRLTrainer, register_trainer
from trlx.trainer.nemo_ilql_trainer import megatron_trainer
from trlx.utils import get_git_tag, infinite_dataloader, tree_map
from trlx.utils.modeling import get_global_statistics, whiten

logging = getLogger(__name__)


def rank_0_tqdm(*args, **kwargs):
    return tqdm(*args, **kwargs, disable=torch.distributed.get_rank() != 0)


@register_trainer
class NeMoPPOTrainer(BaseRLTrainer):
    def __init__(  # noqa: C901
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

        world_size = megatron_cfg.trainer.num_nodes * megatron_cfg.trainer.devices
        dp_world = world_size // (
            megatron_cfg.model.tensor_model_parallel_size * megatron_cfg.model.pipeline_model_parallel_size
        )
        megatron_cfg.model.global_batch_size = config.train.batch_size * dp_world

        if config.train.minibatch_size is not None:
            megatron_cfg.model.micro_batch_size = config.train.minibatch_size
        else:
            megatron_cfg.model.micro_batch_size = config.train.batch_size

        megatron_cfg.model.seed = config.train.seed

        megatron_cfg.model.optim = {"name": config.optimizer.name, **config.optimizer.kwargs}

        megatron_cfg.model.optim.sched = {"name": config.scheduler.name, **config.scheduler.kwargs}

        rank_batch_size = config.train.batch_size

        if (self.ppo_config.chunk_size % rank_batch_size) != 0:
            self.ppo_config.chunk_size = rank_batch_size * ceil(self.ppo_config.chunk_size / rank_batch_size)
            logging.info(f"Rounding chunk size to {self.ppo_config.chunk_size}")

        train_samples = dp_world * self.ppo_config.num_rollouts * self.ppo_config.ppo_epochs
        # Disable validation within nemo, run it ourselves
        self.limit_val_batches = megatron_cfg.trainer.limit_val_batches
        self.val_check_interval = config.train.eval_interval

        megatron_cfg.trainer.limit_val_batches = 0.0
        megatron_cfg.trainer.val_check_interval = None
        self.train_samples = train_samples
        megatron_cfg.trainer.max_steps = config.train.epochs * (train_samples // megatron_cfg.model.global_batch_size)
        megatron_cfg.trainer.max_steps = min(megatron_cfg.trainer.max_steps, config.train.total_steps)

        if pretrained_model is not None and megatron_cfg.model.tokenizer.library == "sentencepiece":
            megatron_cfg.model.tokenizer.model = str(Path(pretrained_model) / megatron_cfg.model.tokenizer.model)
            megatron_cfg.model.tokenizer.tokenizer_model = str(
                Path(pretrained_model) / megatron_cfg.model.tokenizer.tokenizer_model
            )

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

        if megatron_cfg.model.tokenizer.library != "sentencepiece":
            self.tokenizer = self.model.tokenizer.tokenizer
        else:
            self.tokenizer = LlamaTokenizer(vocab_file=megatron_cfg.model.tokenizer.model)
        self.tokenizer.truncation_side = config.tokenizer.truncation_side

        self.tokenizer.pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_length = megatron_cfg.model.encoder_seq_length

        if self.ppo_config.target is not None:
            raise ValueError("AdaptiveKLController not implemented yet")
        else:
            self.kl_ctl = FixedKLController(self.ppo_config.init_kl_coef)

        self.ref_std = None

    def add_prompt_pipeline(self, pipeline: PromptPipeline):
        self.prompt_pipeline = pipeline

    def make_experience(self, prompt_iterator: Iterator, num_rollouts: int = 1024, dp_world: int = 1):  # noqa: C901
        ppo_rl_elements = []

        gc.collect()

        device = torch.device("cuda")

        tbar = rank_0_tqdm(total=num_rollouts * dp_world, desc="Generating experience")

        inputs_samples = []

        while num_rollouts > 0:
            batch: PromptBatch = next(prompt_iterator)
            lengths = batch.attention_mask.sum(dim=1)

            max_new_tokens = self.ppo_config.gen_kwargs.get("max_new_tokens", 128)
            min_new_tokens = self.ppo_config.gen_kwargs.get("min_new_tokens", 1)
            if self.ppo_config.gen_experience_kwargs is not None:
                max_new_tokens = self.ppo_config.gen_experience_kwargs.get("max_new_tokens", max_new_tokens)
                min_new_tokens = self.ppo_config.gen_experience_kwargs.get("min_new_tokens", 1)

            input_ids = batch["input_ids"].to(device)
            lengths = lengths.to(device)
            pad_to = lengths.max().item() + max_new_tokens

            pad_to = ceil(pad_to / 8) * 8
            input_ids = torch.nn.functional.pad(
                input_ids, (0, pad_to - input_ids.shape[1]), value=self.tokenizer.pad_token_id
            )

            samples = self.model.generate(
                (input_ids, lengths), dict(max_length=max_new_tokens, min_length=min_new_tokens)
            )

            inputs_samples.append((input_ids.cpu(), lengths.cpu(), samples))

            if torch.distributed.get_rank() == 0:
                tbar.update(len(input_ids) * dp_world)
            num_rollouts = num_rollouts - len(input_ids)

        self.model.free_kv_cache()
        gc.collect()

        all_sents = [sentence for _, _, samples in inputs_samples for sentence in samples["sentences"]]
        all_prompts = [prompt for _, _, samples in inputs_samples for prompt in samples["prompts"]]
        all_responses = [response for _, _, samples in inputs_samples for response in samples["responses"]]
        unnorm_scores = torch.tensor(
            self.reward_fn(samples=all_sents, prompts=all_prompts, outputs=all_responses), device=device
        )

        if self.ppo_config.scale_reward == "whiten":
            scores = whiten(unnorm_scores, shift_mean=False, group=parallel_state.get_data_parallel_group())
        elif self.ppo_config.scale_reward == "ref":
            if self.ref_std is None:
                _, variance, _ = get_global_statistics(unnorm_scores, group=parallel_state.get_data_parallel_group())
                self.ref_std = torch.sqrt(variance + 1e-8)
            scores = unnorm_scores / self.ref_std
        else:
            scores = unnorm_scores

        scores = torch.clip(scores, -self.ppo_config.cliprange_reward, self.ppo_config.cliprange_reward)

        tbar.close()
        lps_tbar = rank_0_tqdm(total=num_rollouts * dp_world, desc="Computing logprobs")

        stats = {"train/unnorm_scores": unnorm_scores, "train/scores": scores, "train/kl": [], "train/kl_penalty": []}

        chunk_tokens = [samples["token_ids"] for _, _, samples in inputs_samples]
        flattened_tokens = [x for chunk in chunk_tokens for x in chunk]
        flattened_tokens = [torch.tensor(x) for x in flattened_tokens]
        input_lengths = [length for _, lengths, _ in inputs_samples for length in lengths]
        max_length = max(len(x) for x in flattened_tokens)
        padded_tokens = [
            torch.nn.functional.pad(x, (0, (ceil(max_length / 8) * 8) - len(x)), value=self.tokenizer.pad_token_id)
            for x in flattened_tokens
        ]
        all_tokens = torch.stack(padded_tokens, dim=0).to(device)
        attention_mask = all_tokens.ne(self.tokenizer.pad_token_id).long().to(device)
        logprobs_and_values = self.model.infer_logprobs_and_values(all_tokens, attention_mask)
        logprobs_and_values = tree_map(lambda x: x.cpu(), logprobs_and_values)

        query_responses = [(t[:l], t[l:]) for t, l in zip(flattened_tokens, input_lengths)]
        query_tensors, response_tensors = zip(*query_responses)

        logprobs, ref_logprobs, values = (
            logprobs_and_values["logprobs"],
            logprobs_and_values["ref_logprobs"],
            logprobs_and_values["values"],
        )

        attention_mask = attention_mask.cpu()
        log_ratio = (logprobs - ref_logprobs) * attention_mask[:, :-1]
        for query_tensor, response_tensor, logps, ref_logps, vs, kl_penalty, score, start, mask in zip(
            query_tensors,
            response_tensors,
            logprobs,
            ref_logprobs,
            values,
            log_ratio,
            scores,
            input_lengths,
            attention_mask,
        ):
            response_end = mask[start:].sum()
            end = start + response_end
            rewards = self.kl_ctl.value * -kl_penalty[start:end].cpu()
            rewards[-1] += score.cpu()

            response_tensor = torch.nn.functional.pad(response_tensor, (0, 1), value=self.tokenizer.pad_token_id)
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

            ratio = kl_penalty[start:end]
            kl = ratio.exp() - 1 - ratio
            stats["train/kl_penalty"].append(self.kl_ctl.value * -ratio.mean(dim=0, keepdim=True))
            stats["train/kl"].append(kl.mean(dim=0, keepdim=True))

        if torch.distributed.get_rank() == 0:
            lps_tbar.update(len(query_tensors) * dp_world)

        for k, v in stats.items():
            v = torch.cat(v).cuda() if isinstance(v, list) else v.cuda()
            gathered = torch.empty((v.shape[0] * dp_world,), dtype=v.dtype).cuda()
            torch.distributed.all_gather_into_tensor(gathered, v, group=parallel_state.get_data_parallel_group())
            stats[k] = gathered.cpu().numpy()

        return ppo_rl_elements, stats

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

        # Init DDP
        def dummy():
            return

        if self.trainer.strategy.launcher is not None:
            self.trainer.strategy.launcher.launch(dummy, trainer=self.trainer)
        self.trainer.strategy.setup_environment()

        if self.model.cfg.get("transformer_engine", False):
            self.model.setup_transformer_engine_tp_groups()

        dp_world = parallel_state.get_data_parallel_world_size()
        dp_rank = parallel_state.get_data_parallel_rank()

        sampler = DistributedSampler(
            self.prompt_pipeline, num_replicas=dp_world, rank=dp_rank, shuffle=True, drop_last=True
        )
        prompt_dataloader = infinite_dataloader(
            self.prompt_pipeline.create_loader(
                self.ppo_config.chunk_size, shuffle=False, sampler=sampler, drop_last=True
            ),
            sampler=sampler,
        )

        self.model.setup()
        self.trainer.strategy._lightning_module = self.model

        _, schedulers = self.model.configure_optimizers()
        self.model._optimizer.zero_grad()

        scheduler = schedulers[0]["scheduler"]

        prompt_iter = iter(prompt_dataloader)
        local_batch_idx = 0

        rank_batch_size = self.batch_size // dp_world
        total_batches = self.config.train.epochs * (self.train_samples // self.batch_size)

        total_batches = min(total_batches, self.config.train.total_steps)
        train_tbar = rank_0_tqdm(desc="Training", total=total_batches)

        metrics = None
        best_metric = None

        for epoch in range(self.config.train.epochs):
            ppo_rl_rollouts, stats = self.make_experience(
                prompt_iter,
                num_rollouts=self.ppo_config.num_rollouts,
                dp_world=dp_world,
            )

            if torch.distributed.get_rank() == 0:
                hstats = {k: wandb.Histogram(v, num_bins=512) for k, v in stats.items()}
                wandb.log(
                    {**hstats, **{"train/mean_kl": stats["train/kl"].mean(), "trainer/global_step": local_batch_idx}},
                    step=local_batch_idx,
                )

            dataloader = DataLoader(ppo_rl_rollouts, batch_size=rank_batch_size, collate_fn=collate_fn, drop_last=True)
            self.model.offload_reference_model()
            self.model.train()

            for batch in dataloader:
                for _ in range(self.ppo_config.ppo_epochs):
                    self.model.training_step(batch, local_batch_idx)
                    train_tbar.update(1)
                    self.model._optimizer.step()
                    scheduler.step()

                    if local_batch_idx % self.val_check_interval == 0 and local_batch_idx > 0:
                        mbs = self.ppo_config.chunk_size
                        if (mbs * dp_world) > len(self.eval_pipeline):
                            mbs = len(self.eval_pipeline) // dp_world
                        sampler = DistributedSampler(
                            self.eval_pipeline, num_replicas=dp_world, rank=dp_rank, shuffle=False
                        )
                        val_loader = DataLoader(
                            self.eval_pipeline, batch_size=mbs, collate_fn=generate_collate, sampler=sampler
                        )
                        val_stats = [
                            self.model.validation_step(val_batch, local_batch_idx)
                            for val_batch in rank_0_tqdm(val_loader)
                        ]
                        metrics = self.model.validation_epoch_end(val_stats, local_batch_idx)

                    if (local_batch_idx % self.config.train.checkpoint_interval) == 0 and local_batch_idx > 0:
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
