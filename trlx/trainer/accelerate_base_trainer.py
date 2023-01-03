import importlib
import json
import os
import sys
from abc import abstractmethod
from time import time
from typing import Dict, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from accelerate import Accelerator  # type: ignore
from transformers import AutoTokenizer

if importlib.util.find_spec("rich") is not None:
    from tqdm.rich import tqdm
else:
    from tqdm import tqdm

import ray
from ray.air import session
from ray.air.checkpoint import Checkpoint

from trlx.data.configs import TRLConfig
from trlx.trainer import BaseRLTrainer, register_trainer
from trlx.utils import (
    filter_non_scalars,
    get_distributed_config,
    get_git_tag,
    get_optimizer_class,
    get_scheduler_class,
)
from trlx.utils.modeling import freeze_bottom_causal_layers


@register_trainer
class AccelerateRLTrainer(BaseRLTrainer):
    """
    RL model trainer with an `accelerate` based backend
    """

    def __init__(self, config, train_mode=True):
        super().__init__(config, train_mode)

        self.accelerator = Accelerator(log_with=config.train.trackers)

        if int(os.environ.get("WORLD_SIZE", 1)) > 1:
            torch.distributed.barrier(device_ids=[int(os.environ.get("LOCAL_RANK", 0))])

        self.max_length = config.train.seq_length

        # Retrieves model equipped for ppo, ilql, etc
        self.model = self.get_arch(self.config)
        freeze_bottom_causal_layers(
            self.model.base_model, self.config.model.num_layers_unfrozen
        )

        if config.model.tokenizer_path:
            self.tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer_path)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
        else:
            self.tokenizer = None

        script_name = os.path.basename(sys.argv[0]).rsplit(".", 1)[0]
        if not isinstance(config.model.model_path, str):
            model_name = str(config.model.model_path).split()[0]
        else:
            model_name = config.model.model_path.split("/")[-1]
        run_name = f"{script_name}/{model_name}"

        if self.accelerator.is_main_process and not ray.is_initialized():
            config_dict = self.config.to_dict()
            dist_config = get_distributed_config(self.accelerator)
            config_dict["distributed"] = dist_config
            init_trackers_kwargs = {}
            if "wandb" in config.train.trackers:
                init_trackers_kwargs["wandb"] = {
                    "name": run_name,
                    "entity": self.config.train.entity_name,
                    "tags": [get_git_tag()],
                    "mode": "disabled" if os.environ.get("debug", False) else "online",
                }
            self.accelerator.init_trackers(
                project_name=self.config.train.project_name,
                config=config_dict,
                init_kwargs=init_trackers_kwargs,
            )

        self.opt = get_optimizer_class(config.optimizer.name)(
            self.model.parameters(),
            **config.optimizer.kwargs,
        )

        self.scheduler = get_scheduler_class(config.scheduler.name)(
            self.opt,
            **config.scheduler.kwargs,
        )

    def tokenize(self, text: Union[Sequence[str], Sequence[torch.LongTensor]]):
        """
        Tokenize a batch of text after adding bos token to each of the samples
        """
        if isinstance(text[0], torch.LongTensor):
            return text

        text = [self.tokenizer.bos_token + txt for txt in text]
        return self.tokenizer(
            text,
            truncation=True,
            max_length=self.config.seq_length,
            return_tensors="pt",
            # NOTE: We manually add special tokens (bos) above so we set this False
            # to avoid models that automatically add special tokens (e.g. OPT)
            # adding them twice more.
            add_special_tokens=False,
        )

    def generate(self, input_ids, attention_mask=None, **kwargs):
        """Wraps hf's `generate` adding some specific method's defaults"""
        input_ids = input_ids.to(self.accelerator.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.accelerator.device)

        kwargs = dict(self.generate_kwargs, **kwargs)

        with torch.no_grad():
            return self.accelerator.unwrap_model(self.model).generate(
                input_ids=input_ids, attention_mask=attention_mask, **kwargs
            )

    def save(self, directory=None):
        """Creates checkpoint of optimizer, scheduler and a model"""
        self.accelerator.save_state(directory or self.config.train.checkpoint_dir)

    def load(self, directory=None):
        """Load checkpoint of optimizer, scheduler and a model"""
        self.accelerator.load_state(directory or self.config.train.checkpoint_dir)

    def add_eval_pipeline(self, eval_pipeline):
        """Adds pipeline from with validation prompts"""
        self.eval_pipeline = eval_pipeline

    def evaluate(self):  # noqa: C901
        """Samples model on `eval_prompts`, logs stats with `reward_fn` or `metric_fn` if provided"""
        stats = {}
        all_samples = []
        prompts_sizes = []
        generate_time = time()
        for prompts in self.eval_dataloader:
            if isinstance(prompts, torch.Tensor):
                samples = self.generate(prompts)
            else:
                samples = self.generate(**prompts)

            if isinstance(samples, tuple):
                samples, *_ = samples

            pad_token = self.tokenizer.eos_token_id if self.tokenizer else 0
            all_samples.append(
                F.pad(
                    samples,
                    (0, self.max_length - samples.shape[1]),
                    value=pad_token,
                )
            )
            sizes = torch.tensor(prompts.input_ids.shape[1]).repeat(
                len(prompts.input_ids)
            )
            prompts_sizes.append(sizes.to(samples.device))

        stats["time/generate"] = time() - generate_time

        samples = self.accelerator.gather(torch.vstack(all_samples))
        prompts_sizes = self.accelerator.gather(torch.hstack(prompts_sizes))

        if self.accelerator.is_main_process:
            if self.tokenizer:
                str_samples = self.tokenizer.batch_decode(
                    samples, skip_special_tokens=True
                )

                prompts, responses = [], []
                for sample, prompt_size in zip(samples, prompts_sizes):
                    prompts.append(sample[:prompt_size])
                    responses.append(sample[prompt_size:])

                str_prompts = self.tokenizer.batch_decode(
                    prompts, skip_special_tokens=True
                )
                str_responses = self.tokenizer.batch_decode(
                    responses, skip_special_tokens=True
                )

            if isinstance(str_samples[0], str):
                columns_data = [str_prompts, str_responses]
            else:
                columns_data = [samples.tolist()]
            columns = ["prompt", "response"]

            # in online setting, compute the reward for validation
            if self.reward_fn:
                rewards = torch.tensor(self.reward_fn(str_samples), dtype=torch.float)
                mean_reward = rewards.mean()
                columns.append("reward")
                columns_data.append(rewards)
                stats["reward/mean"] = mean_reward
                print(f"{mean_reward=}")

            # additionally log any other metrics
            if self.metric_fn:
                metric_time = time()
                metrics = self.metric_fn(str_samples)
                stats["time/metric"] = time() - metric_time

                mean_metrics = {
                    f"metrics/{k}": torch.as_tensor(xs).mean(-1)
                    for k, xs in metrics.items()
                }

                stats.update(mean_metrics)

                for metric, values in metrics.items():
                    columns.append(metric)
                    columns_data.append(values)

            rows = list(zip(*columns_data))
            print(rows[0])
            if not ray.is_initialized():
                if "wandb" in self.config.train.trackers:
                    import wandb

                    stats["samples"] = wandb.Table(columns=columns, rows=rows)

        return stats

    def learn(self):  # noqa: C901
        """
        Samples batches from `self.store`, updates model and periodically evaluates it on `self.eval_dataloader`
        """

        self.prepare_learning()
        self.iter_count = 0

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

        tbar = tqdm(
            initial=self.iter_count,
            total=self.total_steps,
            disable=not self.accelerator.is_local_main_process,
        )

        for _ in range(self.config.train.epochs):
            for batch in self.train_dataloader:
                for _ in range(self.n_updates_per_batch):
                    forward_time = time()
                    loss, stats = self.loss(batch)
                    forward_time = time() - forward_time

                    backward_time = time()
                    self.accelerator.backward(loss)
                    backward_time = time() - backward_time

                    self.opt.step()
                    self.opt.zero_grad()
                    self.scheduler.step()
                    self.iter_count += 1

                    if self.iter_count % self.config.train.checkpoint_interval == 0:
                        self.save()

                    stats["time/forward"] = forward_time
                    stats["time/backward"] = backward_time
                    for group_number, lr in enumerate(self.scheduler.get_last_lr()):
                        stats[f"learning_rate_group_{group_number}"] = lr

                    if self.iter_count % self.config.train.eval_interval == 0:
                        results = self.evaluate()
                        stats.update(results)

                        # Report the metrics to Ray Tune.
                        if ray.is_initialized():
                            self.save("state")
                            with open("state/state.json", "w") as f:
                                json.dump(dict(iter_count=self.iter_count), f)
                            checkpoint = Checkpoint.from_directory("state")
                            session.report(
                                filter_non_scalars(stats), checkpoint=checkpoint
                            )

                    if not ray.is_initialized():
                        self.accelerator.log(stats, step=self.iter_count)

                    desc = ", ".join(
                        f"{k}: {v:.2f}"
                        for k, v in stats.items()
                        if k.startswith("loss")
                    )
                    tbar.set_description(desc)
                    tbar.update()

                    if self.iter_count >= self.total_steps:
                        self.save()
                        return self.evaluate()

                self.post_backward_callback()

            self.post_epoch_callback()

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
