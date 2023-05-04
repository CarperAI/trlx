from logging import getLogger
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
from trlx.models.modeling_ppo import AdaptiveKLController, FixedKLController, PPOConfig
from trlx.pipeline.offline_pipeline import PromptPipeline
from trlx.pipeline.ppo_pipeline import ppo_collate_fn
from trlx.trainer import BaseRLTrainer, register_trainer
from trlx.trainer.nemo_ilql_trainer import ShuffledCyclicSequence, megatron_trainer
from trlx.utils import infinite_dataloader
from trlx.utils.modeling import logprobs_of_labels

logging = getLogger(__name__)


def fake_initialize_model_parallel(
    world_size,
    rank,
    tensor_model_parallel_size_,
    pipeline_model_parallel_size_,
    pipeline_model_parallel_split_rank_=None,
    virtual_pipeline_model_parallel_size_=None,
):
    """
    Fake initialize model data parallel groups so that we can instantiate model parallel models before DDP is initialized.
    This is needed because PTL execution flow is init model, init trainer -> call trainer.fit(model). DDP is initialized during .fit.
    This function is taken from megatron.core.parallel_state and modified so that the distributed groups are not created.
    Arguments:
        tensor_model_parallel_size: number of GPUs used to parallelize model tensor.
        pipeline_model_parallel_size: number of GPUs used to parallelize model pipeline.
    Let's say we have a total of 16 GPUs denoted by g0 ... g15 and we
    use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
    the model pipeline. The present function will
    create 8 tensor model-parallel groups, 4 pipeline model-parallel groups
    and 8 data-parallel groups as:
        8 data_parallel groups:
            [g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15]
        8 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7], [g8, g9], [g10, g11], [g12, g13], [g14, g15]
        4 pipeline model-parallel groups:
            [g0, g4, g8, g12], [g1, g5, g9, g13], [g2, g6, g10, g14], [g3, g7, g11, g15]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """

    # Get world size and rank. Ensure some consistencies.
    tensor_model_parallel_size = min(tensor_model_parallel_size_, world_size)
    pipeline_model_parallel_size = min(pipeline_model_parallel_size_, world_size)
    model_parallel_size = tensor_model_parallel_size * pipeline_model_parallel_size

    assert (
        world_size % tensor_model_parallel_size * pipeline_model_parallel_size == 0
    ), f"world_size: {world_size} must be divisible by tensor_model_parallel_size: {tensor_model_parallel_size} times pipeline_model_parallel_size {pipeline_model_parallel_size}"
    data_parallel_size = world_size // (tensor_model_parallel_size * pipeline_model_parallel_size)

    num_tensor_model_parallel_groups = world_size // tensor_model_parallel_size
    num_pipeline_model_parallel_groups = world_size // pipeline_model_parallel_size

    virtual_pipeline_model_parallel_rank = None
    if virtual_pipeline_model_parallel_size_ is not None:
        virtual_pipeline_model_parallel_rank = 0

    # Build the data-parallel groups.
    all_data_parallel_group_ranks = []
    for i in range(pipeline_model_parallel_size):
        start_rank = i * num_pipeline_model_parallel_groups
        end_rank = (i + 1) * num_pipeline_model_parallel_groups
        for j in range(tensor_model_parallel_size):
            ranks = range(start_rank + j, end_rank, tensor_model_parallel_size)
            all_data_parallel_group_ranks.append(list(ranks))
            if rank in ranks:
                data_parallel_group = list(ranks)
                logging.info(f"Rank {rank} has data parallel group: {data_parallel_group}")

    data_parallel_rank = data_parallel_group.index(rank)
    logging.info(f"All data parallel group ranks: {all_data_parallel_group_ranks}")
    logging.info(f"Ranks {rank} has data parallel rank: {data_parallel_rank}")

    # Build the model-parallel groups.
    all_model_parallel_group_ranks = []
    for i in range(data_parallel_size):
        ranks = [data_parallel_group_ranks[i] for data_parallel_group_ranks in all_data_parallel_group_ranks]
        all_model_parallel_group_ranks.append(ranks)
        if rank in ranks:
            logging.info(f"Rank {rank} has model parallel group: {list(ranks)}")
    logging.info(f"All model parallel group ranks: {all_model_parallel_group_ranks}")

    # Build the tensor model-parallel groups.
    all_tensor_model_parallel_group_ranks = []
    tensor_model_parallel_group = None
    for i in range(num_tensor_model_parallel_groups):
        ranks = range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
        all_tensor_model_parallel_group_ranks.append(list(ranks))
        if rank in ranks:
            tensor_model_parallel_group = list(ranks)
            logging.info(f"Rank {rank} has tensor model parallel group: {tensor_model_parallel_group}")

    tensor_model_parallel_rank = tensor_model_parallel_group.index(rank)

    logging.info(f"All tensor model parallel group ranks: {all_tensor_model_parallel_group_ranks}")
    logging.info(f"Rank {rank} has tensor model parallel rank: {tensor_model_parallel_rank}")

    # Build the pipeline model-parallel groups and embedding groups
    # (first and last rank in each pipeline model-parallel group).
    all_pipeline_model_parallel_group_ranks = []
    all_embedding_group_ranks = []
    pipeline_model_parallel_group = None
    embedding_group = None
    embedding_rank = None
    for i in range(num_pipeline_model_parallel_groups):
        ranks = range(i, world_size, num_pipeline_model_parallel_groups)
        all_pipeline_model_parallel_group_ranks.append(list(ranks))
        if rank in ranks:
            pipeline_model_parallel_group = list(ranks)
            logging.info(f"Rank {rank} has pipeline model parallel group: {pipeline_model_parallel_group}")

        # Setup embedding group (to exchange gradients between
        # first and last stages).
        if len(ranks) > 1:
            embedding_ranks = [ranks[0], ranks[-1]]
            all_embedding_group_ranks.append(embedding_ranks)
        else:
            embedding_ranks = ranks
            all_embedding_group_ranks.append(list(embedding_ranks))
        if rank in embedding_ranks:
            embedding_group = list(embedding_ranks)
            logging.info(f"Rank {rank} has embedding group: {embedding_group}")

    pipeline_model_parallel_rank = pipeline_model_parallel_group.index(rank)
    if embedding_group is not None:
        embedding_rank = embedding_group.index(rank)

    logging.info(f"All pipeline model parallel group ranks: {all_pipeline_model_parallel_group_ranks}")
    logging.info(f"Rank {rank} has pipeline model parallel rank {pipeline_model_parallel_rank}")
    logging.info(f"All embedding group ranks: {all_pipeline_model_parallel_group_ranks}")
    logging.info(f"Rank {rank} has embedding rank: {embedding_rank}")

    return (
        tensor_model_parallel_rank,
        pipeline_model_parallel_rank,
        data_parallel_rank,
        model_parallel_size,
        data_parallel_size,
        pipeline_model_parallel_split_rank_,
        virtual_pipeline_model_parallel_rank,
    )


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

        # Disable validation within nemo, run it ourselves
        megatron_cfg.trainer.limit_val_batches = 0.0
        megatron_cfg.trainer.val_check_interval = None
        self.train_samples = train_samples

        self.trainer = megatron_trainer(megatron_cfg)
        self.model = PPOGPT(
            ppo_config=self.ppo_config,
            cfg=megatron_cfg.model,
            trainer=self.trainer,
            metric_fn=self.metric_fn,
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

    def make_experience(
        self, prompt_iterator: Iterator, num_rollouts: int = 1024, iter_count: int = 0, data_parallel_size=1
    ):
        ppo_rl_elements = []
        stats = {}

        device = torch.device("cuda")
        rank_rollouts = num_rollouts // data_parallel_size
        while rank_rollouts > 0:
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

            print("generating...")
            samples = self.model.generate((input_ids, lengths), dict(max_length=max_new_tokens, min_length=1))

            scores = torch.tensor(self.reward_fn(samples["sentences"]), device=device)

            scores = torch.clip(scores, -self.ppo_config.cliprange_reward, self.ppo_config.cliprange_reward)
            print(samples.keys())

            output_tokens = samples["token_ids"]

            output_tokens = [
                x + [self.tokenizer.pad_token_id] * ((ceil(len(x) / 8) * 8) - len(x)) for x in output_tokens
            ]
            print([len(t) for t in output_tokens])

            all_tokens = torch.tensor(output_tokens, device=device)
            attention_mask = all_tokens.ne(self.tokenizer.pad_token_id).long().to(device)
            print(f"{all_tokens.shape=}")
            model_output = self.model.infer_logits_and_values(all_tokens, attention_mask)

            # Model output is None on intermediate pipeline stages
            if model_output is None:
                continue

            logits, ref_logits, values = model_output["logits"], model_output["ref_logits"], model_output["values"]

            logprobs = logprobs_of_labels(logits[:, :-1, :], all_tokens[:, 1:])
            ref_logprobs = logprobs_of_labels(ref_logits[:, :-1, :], all_tokens[:, 1:])

            query_responses = [(t[:l], t[l:]) for t, l in zip(output_tokens, lengths)]
            query_tokens, response_tokens = zip(*query_responses)

            query_tensors = [torch.tensor(t, device="cpu") for t in query_tokens]
            response_tensors = [torch.tensor(t, device="cpu") for t in response_tokens]
            logprobs = logprobs.cpu()
            ref_logprobs = ref_logprobs.cpu()

            values = values.cpu()[:, :-1]

            masks = attention_mask.cpu()
            log_ratio = (logprobs - ref_logprobs) * masks[:, :-1]

            for query_tensor, response_tensor, logps, vs, kl_penalty, score, start, mask in zip(
                query_tensors, response_tensors, logprobs, values, log_ratio, scores, lengths, masks
            ):
                end = mask.sum()
                rewards = self.kl_ctl.value * -kl_penalty[start:end].cpu()
                rewards[-1] += score.cpu()
                print(f"{rewards.shape=} {vs.shape=} {logps.shape=} {start=} {end=} {score=}")
                ppo_rl_elements.append(
                    PPORLElement(
                        query_tensor=query_tensor,
                        response_tensor=response_tensor,
                        logprobs=logps[start:end],
                        values=vs[start:end],
                        rewards=rewards,
                    )
                )
                rank_rollouts -= 1

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
            batch = ppo_collate_fn(self.tokenizer.eos_token_id, elems)
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

        world_size = self.trainer.world_size
        global_rank = self.trainer.global_rank
        local_rank = self.trainer.local_rank

        tp_rank, pp_rank, dp_rank, mp_world, dp_world, *_ = fake_initialize_model_parallel(
            world_size,
            global_rank,
            self.model.cfg.tensor_model_parallel_size,
            self.model.cfg.pipeline_model_parallel_size,
        )

        prompt_dataset = ShuffledCyclicSequence(
            new_length=self.config.train.total_steps * self.batch_size,
            data=self.prompt_pipeline,
            seed=self.config.train.seed,
        )

        prompt_dataloader = infinite_dataloader(
            self.prompt_pipeline.create_loader(self.ppo_config.chunk_size, shuffle=True)
        )
        prompt_iter = iter(prompt_dataloader)
        for global_step in range(
            self.config.train.total_steps, self.ppo_config.num_rollouts * self.ppo_config.ppo_epochs
        ):
            ppo_rl_rollouts, stats = self.make_experience(
                prompt_iter,
                num_rollouts=self.ppo_config.num_rollouts * self.ppo_config.ppo_epochs,
                data_parallel_size=dp_world,
            )

            rollout_dataset = ShuffledCyclicSequence(
                new_length=self.train_samples,
                data=ppo_rl_rollouts,
                seed=self.config.train.seed,
            )
            # Dataloader must be initialized inside fit since PTL hasn't initalized DDP yet
            self.model.set_train_dataset(rollout_dataset, collate_fn=collate_fn)
            self.trainer.fit(self.model)
            print("fitted!")

        eval_iters = (max_train_steps // self.trainer.val_check_interval + 1) * self.trainer.limit_val_batches
        eval_samples = eval_iters * self.model.cfg.global_batch_size

        eval_dataset = ShuffledCyclicSequence(
            new_length=eval_samples,
            data=self.eval_pipeline,
            seed=self.config.train.seed,
        )

        self.model.set_valid_dataset(eval_dataset, collate_fn=generate_collate)

        torch.set_float32_matmul_precision("medium")
        self.trainer.validate(self.model)
