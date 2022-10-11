import os
from typing import Dict, Iterable, Union

from time import time
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer

from trlx.model import BaseRLModel, register_model
from trlx.model.nn.ilql_models import CausalLMWithValueHeads
from trlx.pipeline.offline_pipeline import (OfflinePipeline,
                                            OfflineRolloutStorage)
from trlx.utils import Clock, rampup_decay, safe_mkdir, topk_mask


@register_model
class ILQLModel(BaseRLModel):
    def __init__(
        self,
        config,
        logit_mask=None,
        train_mode=True,
    ):
        super().__init__(config, train_mode)

        self.model = CausalLMWithValueHeads(
            config.model.model_path,
            params=config.method,
            num_layers_unfrozen=config.model.num_layers_unfrozen,
        )
        self.max_length = config.train.gen_size

        if config.model.tokenizer_path:
            self.tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer_path)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = None

        self.logit_mask = logit_mask
        self.accelerator = Accelerator(log_with="wandb")

        if int(os.environ.get("WORLD_SIZE", 1)) > 1:
            torch.distributed.barrier(device_ids=[int(os.environ.get("LOCAL_RANK", 0))])
        else:
            torch.random.manual_seed(1000)

        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                project_name=config.train.project_name, config=config.to_dict()
            )

        if self.train_mode:
            self.opt = torch.optim.AdamW(
                self.model.parameters(), lr=self.config.train.learning_rate_init
            )
            self.scheduler = rampup_decay(
                self.config.train.lr_ramp_steps,
                self.config.train.lr_decay_steps,
                self.config.train.learning_rate_target
                / self.config.train.learning_rate_init,
                self.opt,
            )

    def tokenize(self, texts: Union[Iterable[str], Iterable[torch.LongTensor]]):
        if isinstance(texts[0], torch.LongTensor):
            return texts

        tokenized = self.tokenizer(
            [self.tokenizer.bos_token + x + self.tokenizer.eos_token for x in texts],
            max_length=self.max_length,
            truncation=True,
        )

        input_ids = list(map(torch.as_tensor, tokenized["input_ids"]))
        return input_ids

    def get_components(self) -> Dict[str, any]:
        components = (
            {"model": self.model, "opt": self.opt, "scheduler": self.scheduler}
            if self.train_mode
            else {"model": self.model}
        )
        return components

    def learn(self):
        train_dataloader = self.train_store.create_loader(self.config.train.batch_size)
        eval_dataloader = self.eval_pipeline.create_loader(self.config.train.batch_size)

        (
            self.model,
            self.opt,
            train_dataloader,
            eval_dataloader,
        ) = self.accelerator.prepare(
            self.model, self.opt, train_dataloader, eval_dataloader
        )

        opt_steps = 0
        for epoch in range(self.config.train.epochs):
            evals_stats = {}
            logs = {}
            for batch in train_dataloader:
                if opt_steps % self.config.train.eval_interval == 0:
                    generate_time = time()
                    self.model.eval()

                    for beta in self.config.method.betas:
                        all_samples = []
                        for prompts in eval_dataloader:
                            samples, tensor_stats = self.model.sample(
                                prompts,
                                beta=beta,
                                max_length=self.max_length,
                                logit_mask=self.logit_mask,
                            )

                            all_samples.append(samples)

                        samples = self.accelerator.gather(torch.vstack(all_samples))

                        if self.accelerator.is_main_process:
                            if self.tokenizer:
                                samples = self.tokenizer.batch_decode(
                                    samples, skip_special_tokens=True
                                )

                            metric_time = time()
                            metrics = self.metric_fn(samples)
                            metric_time = time() - metric_time
                            logs.update({'metric_time': metric_time})

                            mean_metrics = {
                                f"metrics/{k}/{beta}": torch.as_tensor(xs).mean(-1)
                                for k, xs in metrics.items()
                            }
                            logs.update(tensor_stats)
                            logs.update(mean_metrics)

                            rows = list(zip(samples, *metrics.values()))
                            logs[f"samples/{beta}"] = wandb.Table(
                                columns=["samples", *metrics.keys()], rows=rows
                            )
                            for row in rows[:4]:
                                print(row)

                    self.model.train()
                    generate_time = time() - generate_time

                forward_time = time()
                loss, stats = self.model.loss(batch)
                forward_time = time() - forward_time

                backward_time = time()
                self.accelerator.backward(loss)
                backward_time = time() - backward_time

                self.opt.step()
                self.opt.zero_grad()
                self.scheduler.step()

                if opt_steps % self.config.train.eval_interval == 0:
                    logs.update({
                        'forward_time': forward_time,
                        'generate_time': generate_time,
                        'backward_time': backward_time,
                    })
                    logs.update(stats)
                    self.accelerator.log(logs)

                if (opt_steps + 1) % self.config.method.steps_for_target_q_sync == 0:
                    self.model.sync_target_q_heads()

                opt_steps += 1
