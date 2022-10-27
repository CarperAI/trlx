import importlib
import os
from abc import abstractmethod
from time import time
from typing import Any, Dict, Iterable, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
import sys

sys.path.append("/fsx/home-uwu/gpt-neox")
import megatron  # type: ignore

from transformers import AutoTokenizer

import wandb
from trlx.data.configs import TRLConfig
from trlx.model import BaseRLModel, register_model
from trlx.model.nn.ilql_models import GPTNeoXWithValueHeads

if importlib.util.find_spec("rich") is not None:
    from tqdm.rich import tqdm
else:
    from tqdm import tqdm


@register_model
class NeoXRLModel(BaseRLModel):
    """
    RL Model that uses NeoX for training
    """

    def __init__(
        self, config: TRLConfig, neox_args: megatron.NeoXArgs, train_mode=True
    ):
        super().__init__(config, train_mode)
        neox_args.is_pipe_parallel = True

        megatron.utils.init_wandb(neox_args=neox_args)
        megatron.initialize.initialize_megatron(neox_args)
        # Retrieves model equipped for ppo, ilql, etc
        self.neox_args = neox_args
        self.tokenizer = neox_args.tokenizer
        self.model = GPTNeoXWithValueHeads(config=neox_args, ilql_config=config.method)
        self.optimizer = megatron.training.get_optimizer(self.model, neox_args)
        self.lr_scheduler = megatron.training.get_learning_rate_scheduler(
            optimizer=self.optimizer, neox_args=neox_args
        )

    # def tokenize(self, text: Union[Sequence[str], Sequence[torch.LongTensor]]):
    #     """
    #     Tokenize a batch of text after adding bos token to each of the samples
    #     """
    #     if isinstance(text[0], torch.LongTensor):
    #         return text

    #     text = [self.tokenizer.bos_token + txt for txt in text]
    #     return self.tokenizer(
    #         text,
    #         truncation=True,
    #         max_length=self.config.seq_length,
    #         return_tensors="pt",
    #     )

    # def generate(self, input_ids, attention_mask=None, **kwargs):
    #     """Wraps hf's `generate` adding some specific method's defaults"""
    #     input_ids = input_ids.to(self.accelerator.device)
    #     if attention_mask is not None:
    #         attention_mask = attention_mask.to(self.accelerator.device)

    #     kwargs = dict(self.generate_kwargs, **kwargs)

    #     with torch.no_grad():
    #         return self.accelerator.unwrap_model(self.model).generate(
    #             input_ids=input_ids, attention_mask=attention_mask, **kwargs
    #         )

    # def get_components(self) -> Dict[str, Any]:
    #     components = (
    #         {"model": self.model, "opt": self.opt, "scheduler": self.scheduler}
    #         if self.train_mode
    #         else {"model": self.model}
    #     )
    #     return components

    # def save(self, directory=None):
    #     """Creates checkpoint of optimizer, scheduler and a model"""
    #     self.accelerator.save_state(directory or self.config.train.checkpoint_dir)

    # def add_eval_pipeline(self, eval_pipeline):
    #     """Adds pipeline from with validation prompts"""
    #     self.eval_pipeline = eval_pipeline

    # def evaluate(self):
    #     """Samples model on `eval_prompts`, logs stats with `reward_fn` or `metric_fn` if provided"""
    #     stats = {}
    #     all_samples = []
    #     generate_time = time()
    #     for prompts in self.eval_dataloader:
    #         if isinstance(prompts, torch.Tensor):
    #             samples = self.generate(prompts)
    #         else:
    #             samples = self.generate(**prompts)

    #         if isinstance(samples, tuple):
    #             samples, *_ = samples

    #         pad_token = self.tokenizer.eos_token_id if self.tokenizer else 0
    #         all_samples.append(
    #             F.pad(
    #                 samples,
    #                 (0, self.max_length - samples.shape[1]),
    #                 value=pad_token,
    #             )
    #         )
    #     stats["generate_time"] = time() - generate_time

    #     samples = self.accelerator.gather(torch.vstack(all_samples))

    #     if self.accelerator.is_main_process:
    #         if self.tokenizer:
    #             samples = self.tokenizer.batch_decode(samples, skip_special_tokens=True)

    #         if isinstance(samples[0], str):
    #             columns_data = [samples]
    #         else:
    #             columns_data = [samples.tolist()]
    #         columns = ["samples"]

    #         # in online setting, compute the reward for validation
    #         if self.reward_fn:
    #             rewards = torch.as_tensor(self.reward_fn(samples), dtype=torch.float)
    #             mean_reward = rewards.mean()
    #             columns.append("reward")
    #             columns_data.append(rewards)
    #             stats["mean_reward"] = mean_reward
    #             print(f"{mean_reward=}")

    #         # additionally log any other metrics
    #         if self.metric_fn:
    #             metric_time = time()
    #             metrics = self.metric_fn(samples)
    #             stats["metric_time"] = time() - metric_time

    #             mean_metrics = {
    #                 f"metrics/{k}": torch.as_tensor(xs).mean(-1)
    #                 for k, xs in metrics.items()
    #             }

    #             stats.update(mean_metrics)

    #             for metric, values in metrics.items():
    #                 columns.append(metric)
    #                 columns_data.append(values)

    #         rows = list(zip(*columns_data))
    #         stats["samples"] = wandb.Table(columns=columns, rows=rows)

    #         print(rows[0])

    #     return stats

    def learn(self):
        """
        Samples batches from `self.store`, updates model and periodically evaluates it on `self.eval_dataloader`
        """
        train_dataloader = self.store.create_loader(self.config.train.batch_size)
        eval_dataloader = self.eval_pipeline.create_loader(self.config.train.batch_size)

        megatron.training.train(
            self.neox_args,
            self.timers,
            self.model,
            self.optimizer,
            self.lr_scheduler,
            train_dataloader,
            eval_dataloader,
        )

    # @abstractmethod
    # def get_arch(self, config: TRLConfig):
    #     """Returns a specific wrapper of the decoder architecture"""
    #     pass

    # @abstractmethod
    # def loss(self, batch) -> Tuple[float, Dict]:
    #     """Compute loss on a batch from `store` and return some statistics"""
    #     pass

    # @abstractmethod
    # def post_backward_callback(self):
    #     """Do something after model update"""
    #     pass

    # @abstractmethod
    # def post_epoch_callback(self):
    #     """Do something after exhausting/single pass over `self.store`"""
    #     pass
