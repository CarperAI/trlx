import contextlib
import json
import os
import sys
from abc import abstractmethod
from contextlib import contextmanager
from time import time
from typing import Dict, List, Optional, Tuple

import ray
import torch
from accelerate import Accelerator  # type: ignore
from ray.air import session
from rich.console import Console
from rich.table import Table
from transformers import AutoTokenizer

import trlx.utils.logging as logging
from trlx.data.configs import TRLConfig
from trlx.pipeline import MiniBatchIterator
from trlx.trainer import BaseRLTrainer, register_trainer
from trlx.utils import (
    filter_non_scalars,
    get_distributed_config,
    get_git_tag,
    get_optimizer_class,
    get_scheduler_class,
    significant,
)
from trlx.utils.modeling import (
    flatten_dict,
    freeze_bottom_causal_layers,
    freeze_bottom_seq2seq_layers,
    gather_dict,
    get_delta_model_class,
    parse_delta_kwargs,
)

logger = logging.get_logger(__name__)


@register_trainer
class AccelerateRLTrainer(BaseRLTrainer):
    """
    RL model trainer with an `accelerate` based backend
    """

    def __init__(self, config, **kwargs):  # noqa: C901
        super().__init__(config, **kwargs)
        self.max_length = config.train.seq_length
        if config.train.minibatch_size:
            assert config.train.batch_size % config.train.minibatch_size == 0, "Minibatch size must divide batch size"
            self.mb_size = config.train.minibatch_size
        else:
            self.mb_size = config.train.batch_size
        self.num_mb = config.train.batch_size // self.mb_size
        self.mb_count = 0
        self.accelerator = Accelerator(log_with=config.train.tracker, logging_dir=config.train.logging_dir)

        if self.accelerator.state.deepspeed_plugin is not None:
            # by accelerate's default, arguments in `model.forward` would be casted to half
            if "fp16" in self.accelerator.state.deepspeed_plugin.deepspeed_config:
                self.accelerator.state.deepspeed_plugin.deepspeed_config["fp16"]["auto_cast"] = False

        if int(os.environ.get("WORLD_SIZE", 1)) > 1:
            torch.distributed.barrier(device_ids=[int(os.environ.get("LOCAL_RANK", 0))])

        self.model = self.setup_model()
        self.opt = self.setup_optimizer()
        self.scheduler = self.setup_scheduler()

        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_path)
        self.tokenizer.padding_side = config.tokenizer.padding_side
        self.tokenizer.truncation_side = config.tokenizer.truncation_side
        self.tokenizer.sep_token = "<sep>"
        if config.model.model_arch_type != "seq2seq":
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        script_name = os.path.basename(sys.argv[0]).rsplit(".", 1)[0]
        if not isinstance(config.model.model_path, str):
            model_name = str(config.model.model_path).split()[0]
        else:
            model_name = config.model.model_path.split("/")[-1]

        if self.accelerator.num_processes == 1:
            num_gpus = "1gpu"
        else:
            num_gpus = f"{self.accelerator.num_processes}gpus"
        branch = get_git_tag()[0]

        run_name = "/".join([script_name, model_name, num_gpus]) + f":{branch}"

        if self.accelerator.is_main_process:
            config_dict = self.config.to_dict()
            dist_config = get_distributed_config(self.accelerator)
            config_dict["distributed"] = dist_config
            init_trackers_kwargs = {}

            if config.train.tracker == "wandb":
                init_trackers_kwargs["wandb"] = {
                    "name": run_name,
                    "entity": self.config.train.entity_name,
                    "group": self.config.train.group_name,
                    "tags": self.config.train.tags + ["/".join(get_git_tag())],
                    "mode": "disabled" if os.environ.get("debug", False) else "online",
                }

                self.accelerator.init_trackers(
                    project_name=self.config.train.project_name,
                    config=config_dict,
                    init_kwargs=init_trackers_kwargs,
                )
            elif config.train.tracker == "tensorboard":
                # flatten config for tensorboard, split list in hparams into flatten config
                config_dict_flat = flatten_dict(config_dict)
                config_dict_flat["optimizer/kwargs/beta_1"] = config_dict_flat["optimizer/kwargs/betas"][0]
                config_dict_flat["optimizer/kwargs/beta_2"] = config_dict_flat["optimizer/kwargs/betas"][1]
                config_dict_flat.pop("optimizer/kwargs/betas", None)
                self.accelerator.init_trackers(
                    project_name=self.config.train.project_name,
                    config=config_dict_flat,
                )
            elif config.train.tracker is None:
                self.accelerator.init_trackers(project_name=self.config.train.project_name)
            else:
                raise ValueError(
                    f"Only supported trackers are `wandb` and `tensorboard`. Got: `{config.train.tracker}`. "
                    "Set `tracker` to `None` to disable tracking."
                )

    def setup_model(self):
        """
        Returns a model derived from an instance's TRLConfig
        """
        logger.info(f"Initializing model: {self.config.model.model_path}")

        # Retrieves model equipped for ppo, ilql, etc
        model = self.get_arch(self.config)
        if self.config.model.model_arch_type == "seq2seq":
            freeze_bottom_seq2seq_layers(model.base_model, self.config.model.num_layers_unfrozen)
        else:
            freeze_bottom_causal_layers(model.base_model, self.config.model.num_layers_unfrozen)
        # Set the delta tuning strategies
        if self.config.model.delta_kwargs is not None:
            delta_type, delta_kwargs = parse_delta_kwargs(
                model.base_model.config,
                self.config.model.delta_kwargs,
                self.config.model.num_layers_unfrozen,
            )
            delta_model_class = get_delta_model_class(delta_type)
            delta_model = delta_model_class(model.base_model, **delta_kwargs)
            delta_model.freeze_module(exclude=["deltas"], set_state_dict=True)
            if self.accelerator.is_main_process:
                delta_model.log()
        return model

    def setup_optimizer(self):
        """
        Returns an optimizer derived from an instance's TRLConfig
        """
        optimizer_class = get_optimizer_class(self.config.optimizer.name)
        optimizer = optimizer_class(
            self.model.parameters(),
            **self.config.optimizer.kwargs,
        )

        if "bitsandbytes" in optimizer.__class__.__module__:
            # Force 32-bit `nn.Embedding` weights for stability. See discussion:
            # https://github.com/huggingface/transformers/issues/14819#issuecomment-1016017746
            from bitsandbytes.optim import GlobalOptimManager

            manager = GlobalOptimManager.get_instance()
            for module in self.model.modules():
                if isinstance(module, torch.nn.Embedding):
                    manager.register_module_override(module, "weight", {"optim_bits": 32})

        return optimizer

    def setup_scheduler(self):
        """
        Returns a learning rate scheduler derived from an instance's TRLConfig
        """
        scheduler_class = get_scheduler_class(self.config.scheduler.name)
        scheduler = scheduler_class(self.opt, **self.config.scheduler.kwargs)
        return scheduler

    def decode(
        self,
        prompts: List[torch.LongTensor],
        samples: List[torch.LongTensor],
        prompt_sizes: torch.LongTensor = None,
        append_eos_token: bool = False,
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Decode tensor generations into lists of strings (`samples`: List[str], `prompts`: List[str], `outputs`: List[str])
        """
        if prompt_sizes is None:
            # Assuming prompts were left-padded
            prompt_sizes = [prompts.shape[1]] * len(prompts)

        str_samples, str_prompts, str_outputs = [], [], []
        for prompt, sample, prompt_size in zip(prompts, samples, prompt_sizes):
            if self.config.model.model_arch_type == "seq2seq":
                output_start_ix = 0
            else:
                output_start_ix = prompt_size

            str_prompt = self.tokenizer.decode(prompt[:prompt_size], skip_special_tokens=True)
            str_output = self.tokenizer.decode(sample[output_start_ix:], skip_special_tokens=True)
            # Trim outputs up to `self.stop_sequences` if any are present
            trimmed = False
            if self.stop_sequences:
                for stop in self.stop_sequences:
                    stop_ix = str_output.find(stop)
                    if stop_ix >= 0:
                        str_output = str_output[:stop_ix].rstrip()
                        trimmed = True

            # Recover the last <eos> if it was present in the original sample
            # or add one if it was trimmed with `self.stop_sequences`.
            # Only in cases when a generation ended due to `max_new_tokens` exhaustion,
            # <eos> token would not be present in the original sample
            if append_eos_token and (trimmed or sample[-1] == self.tokenizer.eos_token_id):
                str_output += self.tokenizer.eos_token

            str_prompts.append(str_prompt)
            str_outputs.append(str_output)

            if self.config.model.model_arch_type == "seq2seq":
                sample = str_prompt + self.tokenizer.sep_token + str_output
            else:
                sample = str_prompt + str_output

            str_samples.append(sample)

        return str_samples, str_prompts, str_outputs

    def generate(self, input_ids, attention_mask=None, **kwargs):
        """Wraps hf's `generate` adding some specific method's defaults"""
        input_ids = input_ids.to(self.accelerator.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.accelerator.device)
        if self.generate_experience_kwargs is not None:
            kwargs = dict(self.generate_experience_kwargs, **kwargs)
        else:
            kwargs = dict(self.generate_kwargs, **kwargs)

        with torch.no_grad():
            return self.accelerator.unwrap_model(self.model).generate(
                input_ids=input_ids, attention_mask=attention_mask, **kwargs
            )

    def generate_eval(self, input_ids, attention_mask=None, **kwargs):
        """Wraps hf's `generate` adding some specific method's defaults"""
        input_ids = input_ids.to(self.accelerator.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.accelerator.device)

        kwargs = dict(self.generate_kwargs, **kwargs)

        with torch.no_grad():
            return self.accelerator.unwrap_model(self.model).generate(
                input_ids=input_ids, attention_mask=attention_mask, **kwargs
            )

    def save_pretrained(self, directory: Optional[str] = None, **kwargs):
        """Save the underlying Hugging Face model, tokenizer, and configuration files to a directory for
        later use.

        Args:
            directory (str, *optional*): The directory to save the trainer files to.
                NOTE: If not specified, the model will be saved to a directory named `hf_model` in the
                checkpoint directory as specified by the Trainer's config.
            **kwargs: Additional keyword arguments passed to the underlying Hugging Face model's
                `save_pretrained` method.
        """
        if directory is None:
            directory = os.path.join(self.config.train.checkpoint_dir, "hf_model")

        self.accelerator.wait_for_everyone()
        self.accelerator.unwrap_model(self.model).save_pretrained(
            directory,
            save_function=self.accelerator.save,
            is_main_process=self.accelerator.is_main_process,
            state_dict=self.accelerator.get_state_dict(self.model),
            **kwargs,
        )

        if self.accelerator.is_main_process:
            self.tokenizer.save_pretrained(directory)

    def save(self, directory: Optional[str] = None, **kwargs):
        """Creates a checkpoint of the optimizer, scheduler and model"""
        self.accelerator.save_state(directory or self.config.train.checkpoint_dir, **kwargs)

    def load(self, directory: Optional[str] = None, **kwargs):
        """Load checkpoint of optimizer, scheduler and a model"""
        self.accelerator.load_state(directory or self.config.train.checkpoint_dir, **kwargs)

    def add_eval_pipeline(self, eval_pipeline):
        """Adds pipeline from with validation prompts"""
        self.eval_pipeline = eval_pipeline

    def evaluate(self):  # noqa: C901
        """Samples model on `eval_prompts`, logs stats with `reward_fn` or `metric_fn` if provided"""
        logger.info("Evaluating model")

        # Do multiple evaluations over a single list in `gen_kwargs` if present
        if self.generate_sweep_kwarg is not None:
            gen_sweep_arg, gen_sweep_values = self.generate_sweep_kwarg
        else:
            gen_sweep_values = [None]

        desc = [
            f"generation sweep 0/{len(gen_sweep_values)}",
            f"eval batch 0/{len(self.eval_dataloader)}",
        ]
        tbar = logging.tqdm(
            total=len(self.eval_dataloader) * len(gen_sweep_values),
            desc=f"[{' | '.join(desc)}]",
            disable=not self.accelerator.is_main_process,
            position=0,
            leave=True,
        )

        stats = {}
        table = []

        for i_sweep, gen_sweep_value in enumerate(gen_sweep_values):
            # A dedicated suffix for wandb logging
            if gen_sweep_value is not None:
                sweep_suffix = f"@{gen_sweep_arg}={gen_sweep_value}"
            else:
                sweep_suffix = ""

            all_samples = []
            all_prompts = []
            all_prompt_sizes = []
            all_metadata = []
            generate_time = time()
            for i_prompt, prompts in enumerate(self.eval_dataloader):
                metadata = {k: v for k, v in prompts.items() if k != "input_ids" and k != "attention_mask"}
                if self.generate_sweep_kwarg:
                    samples = self.generate_eval(
                        prompts["input_ids"], prompts["attention_mask"], **{gen_sweep_arg: gen_sweep_value}
                    )
                else:
                    samples = self.generate_eval(prompts["input_ids"], prompts["attention_mask"])

                # TODO(reciprocated): this should be moved into `decode`
                # but that needs to be synced with indexing in `make_experience`
                if self.config.model.model_arch_type == "seq2seq":
                    samples = samples[:, 1:].contiguous()

                prompt_sizes = torch.tensor(prompts.input_ids.shape[1]).repeat(len(prompts.input_ids))
                prompts, samples, prompt_sizes = self.accelerator.gather_for_metrics(
                    self.accelerator.pad_across_processes(
                        [prompts.input_ids, samples, prompt_sizes.to(samples.device)],
                        dim=1,
                        pad_index=self.tokenizer.pad_token_id,
                    )
                )
                all_samples.extend(samples.tolist())
                all_prompts.extend(prompts.tolist())
                all_prompt_sizes.extend(prompt_sizes.tolist())

                metadata = gather_dict(metadata, self.accelerator.gradient_state)
                all_metadata.append(metadata)

                desc = [
                    f"generation sweep {i_sweep + 1}/{len(gen_sweep_values)}",
                    f"eval batch {i_prompt + 1}/{len(self.eval_dataloader)}",
                ]
                tbar.set_description(f"[{' | '.join(desc)}]")
                tbar.update()
            tbar.close()

            stats["time/generate"] = time() - generate_time

            if self.accelerator.is_main_process:
                str_samples, str_prompts, str_outputs = self.decode(all_prompts, all_samples, all_prompt_sizes)

                columns = ["prompt", "output"]
                columns_data = [str_prompts, str_outputs]

                metadata, *xs = all_metadata
                for k in metadata:
                    for x in xs:
                        metadata[k].extend(x[k])

                # in online setting, compute the reward for validation
                if self.reward_fn:
                    logger.info("Computing rewards")
                    rewards = torch.tensor(
                        self.reward_fn(samples=str_samples, prompts=str_prompts, outputs=str_outputs, **metadata),
                        dtype=float,
                    )
                    mean_reward = rewards.mean().item()
                    columns.append("reward")
                    if not isinstance(rewards, list):
                        rewards = rewards.tolist()
                    columns_data.append(rewards)
                    stats[f"reward/mean{sweep_suffix}"] = mean_reward

                # additionally log any other metrics
                if self.metric_fn:
                    logger.info("Computing metrics")
                    metric_time = time()
                    metrics = self.metric_fn(samples=str_samples, prompts=str_prompts, outputs=str_outputs, **metadata)
                    stats["time/metric"] = time() - metric_time

                    mean_metrics = {
                        f"metrics/{k}{sweep_suffix}": torch.as_tensor(xs).mean(-1).item() for k, xs in metrics.items()
                    }

                    stats.update(mean_metrics)

                    for metric, values in metrics.items():
                        # Skip metrics that are scalers since they represent aggregated values
                        if isinstance(values, float):
                            continue
                        columns.append(metric)
                        if not isinstance(values, list):
                            values = values.tolist()
                        columns_data.append(values)

                # Prepend the sweep argument along with samples
                if self.generate_sweep_kwarg:
                    columns.insert(0, gen_sweep_arg)
                    columns_data.insert(0, [gen_sweep_value] * len(samples))

                table.append(list(zip(*columns_data)))

        # Log and display evaluation metrics
        logger.info("Summarizing evaluation")
        if self.accelerator.is_main_process:
            rows = sum(list(map(list, zip(*table))), [])

            # Add metrics/rewards to the table's title
            table_title = f"Evaluation #{self.nth_evaluation}"
            for k, x in stats.items():
                if k.startswith("reward") or k.startswith("metrics"):
                    table_title += f" {k}: {significant(x)}"

            rich_table = Table(*columns, title=table_title, show_lines=True)
            for ix in range(max(min(3, len(rows)), len(gen_sweep_values))):
                rich_table.add_row(*[str(significant(x)) for x in rows[ix]])
            Console().print(rich_table)

            if self.config.train.tracker == "wandb":
                import wandb

                stats["samples"] = wandb.Table(columns, rows)

        self.nth_evaluation += 1
        return stats

    @contextmanager
    def _accumulate(self):
        # We can't use accelerator.accumulate() since that checks if the dataloader is exhausted
        # and we do exhaust the eval dataloader right before each training loop
        self.mb_count += 1
        assert self.mb_count // self.num_mb <= self.config.train.total_steps, "Beyond total steps, something is wrong"
        if (
            self.mb_count % self.accelerator.gradient_accumulation_steps == 0
            or self.mb_count // self.num_mb >= self.config.train.total_steps
        ):
            context = contextlib.nullcontext
        else:
            context = self.accelerator.no_sync
        with context(self.model):
            yield

    def learn(self):  # noqa: C901
        """
        Samples batches from `self.store`, updates model and periodically evaluates it on `self.eval_dataloader`
        """
        logger.info("Starting training")

        self.generate_sweep_kwarg = None
        for k, v in self.config.method.gen_kwargs.items():
            if isinstance(v, list):
                if self.generate_sweep_kwarg is not None:
                    logger.info("Only a single sweep is allowed, {k} is going to be set to {v[0]}")
                    self.generate_kwargs[k] = v[0]
                else:
                    self.generate_sweep_kwarg = (k, v)

        self.prepare_learning()
        self.iter_count = 0
        self.nth_evaluation = 0

        if ray.is_initialized():
            checkpoint = session.get_checkpoint()
            if checkpoint:
                with checkpoint.as_directory() as dir:
                    self.accelerator.load_state(dir)

                    with open(os.path.join(dir, "state.json")) as f:
                        state = json.load(f)
                        self.iter_count = state["iter_count"]
        else:
            results = self.evaluate()
            self.accelerator.log(results, step=self.iter_count)

        tbar = logging.tqdm(
            initial=self.iter_count,
            total=self.total_steps,
            disable=not self.accelerator.is_local_main_process,
            position=0,
            leave=True,
        )

        best_reward = -float("inf")

        # For each epoch
        for _ in range(self.config.train.epochs):
            # For each batch
            for mbs in MiniBatchIterator(self.train_dataloader, self.mb_size, self.num_mb):
                # For each update per batch
                for _ in range(self.n_updates_per_batch):
                    # Note that whereas standard policy gradient methods perform one
                    # gradient update per batch, PPO for example commonly performs
                    # multiple gradient updates on the same batch of data.
                    # https://arxiv.org/pdf/1707.06347.pdf
                    forward_time = 0
                    backward_time = 0
                    stats_accum = []
                    for mb in mbs:
                        with self._accumulate():
                            forward_time -= time()
                            loss, stats = self.loss(mb)
                            forward_time += time()
                            backward_time -= time()
                            self.accelerator.backward(loss)
                            backward_time += time()
                            stats_accum.append(stats)

                    forward_time /= self.num_mb
                    backward_time /= self.num_mb
                    # TODO(Dahoas): Best way to combine stats between mbs?
                    # How does accelerate do it?
                    stats = {key: sum([stats[key] for stats in stats_accum]) / self.num_mb for key in stats_accum[0]}

                    self.opt.step()
                    self.opt.zero_grad()
                    self.scheduler.step()
                    self.iter_count += 1

                    if (
                        self.iter_count % self.config.train.checkpoint_interval == 0
                        or self.iter_count >= self.total_steps
                    ):
                        subfolder = f"checkpoint_{self.iter_count:0{len(str(self.total_steps))}d}"
                        directory = os.path.join(self.config.train.checkpoint_dir, subfolder)
                        logger.info(f"Saving intermediate checkpoint into {directory}")
                        if self.config.train.save_optimizer:
                            self.save(directory)
                        else:
                            self.save_pretrained(directory)

                    stats["time/forward"] = forward_time
                    stats["time/backward"] = backward_time
                    for group_number, lr in enumerate(self.scheduler.get_last_lr()):
                        stats[f"learning_rate_group_{group_number}"] = lr

                    if self.iter_count % self.config.train.eval_interval == 0 or self.iter_count >= self.total_steps:
                        results = self.evaluate()
                        stats.update(results)
                        if ray.is_initialized():
                            session.report(filter_non_scalars(stats), checkpoint=checkpoint)

                        # always save checkpoint with the greatest mean reward
                        if self.config.train.save_best:
                            if stats.get("reward/mean", -float("inf")) > best_reward:
                                best_reward = stats.get("reward/mean")
                                do_save = True
                            # in case ILQL reports reward estimate as one of its metrics
                            elif stats.get("metrics/reward", -float("inf")) > best_reward:
                                best_reward = stats.get("metrics/reward")
                                do_save = True
                            else:
                                do_save = False
                            do_save = torch.tensor(do_save, device=self.accelerator.device)
                            if torch.distributed.is_initialized():
                                torch.distributed.all_reduce(do_save, torch.distributed.ReduceOp.MAX)
                            if do_save:
                                directory = os.path.join(self.config.train.checkpoint_dir, "best_checkpoint")
                                logger.info(f"Saving the best state so far into {directory}")
                                if self.config.train.save_optimizer:
                                    self.save(directory)
                                else:
                                    self.save_pretrained(directory)

                    desc = " | ".join(f"{k}: {v:.2f}" for k, v in stats.items() if k.startswith("loss"))
                    tbar.set_description(f"[{desc}]")
                    tbar.update()

                    self.accelerator.log(stats, step=self.iter_count)

                    if self.iter_count >= self.total_steps:
                        return results

                self.post_backward_callback()

            self.post_epoch_callback()
        tbar.close()

    @abstractmethod
    def get_arch(self, config: TRLConfig):
        """Returns a specific wrapper of the decoder architecture"""
        pass

    @abstractmethod
    def loss(self, batch) -> Tuple[float, Dict]:
        """Compute loss on a batch from `store` and return some statistics"""
        pass

    @abstractmethod
    def post_backward_callback(self):
        """Do something after model update"""
        pass

    @abstractmethod
    def post_epoch_callback(self):
        """Do something after exhausting/single pass over `self.store`"""
        pass
