import os
from typing import Dict, Iterable

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer

from trlx.model import BaseRLModel, register_model
from trlx.model.nn.ilql_models import CausalLMWithValueHeads
from trlx.pipeline.offline_pipeline import OfflinePipeline, OfflineRolloutStorage
from trlx.utils import Clock, rampup_decay, safe_mkdir, topk_mask

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))


@register_model
class ILQLModel(BaseRLModel):
    def __init__(
        self,
        config,
        tokenizer=None,
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

        self.logit_mask = logit_mask
        self.tokenizer = tokenizer

        self.accelerator = Accelerator(log_with="wandb")

        if WORLD_SIZE > 1:
            torch.distributed.barrier(device_ids=[LOCAL_RANK])
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

    def tokenize(self, texts):
        return self.tokenizer(
            [self.tokenizer.bos_token + x + self.tokenizer.eos_token for x in texts],
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

    def get_components(self) -> Dict[str, any]:
        components = (
            {"model": self.model, "opt": self.opt, "scheduler": self.scheduler}
            if self.train_mode
            else {"model": self.model}
        )
        return components

    def learn(self):
        timer = Clock()

        eos_token_id = self.tokenizer.eos_token_id if self.tokenizer else 0
        train_dataloader = self.train_store.create_loader(
            self.config.train.batch_size, eos_token_id=eos_token_id
        )
        eval_dataloader = self.eval_pipeline.create_loader(
            self.config.train.batch_size, shuffle=False
        )

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
                    self.model.eval()

                    all_samples = []
                    for prompts in eval_dataloader:
                        with torch.no_grad():
                            samples, _ = self.model.sample(
                                prompts,
                                beta=self.model.beta,
                                max_length=self.config.train.gen_size,
                                logit_mask=self.logit_mask,
                            )

                        all_samples.append(samples)

                    samples = self.accelerator.gather(torch.vstack(all_samples))

                    if self.accelerator.is_main_process:
                        rewards = torch.as_tensor(self.reward_fn(samples), dtype=float)
                        reward = rewards.mean()

                        if self.stats_fn:
                            eval_stats = self.stats_fn(samples)
                            logs.update(eval_stats)

                        if self.tokenizer:
                            texts = self.tokenizer.batch_decode(
                                samples, skip_special_tokens=True
                            )
                            pairs = list(zip(texts, rewards))
                            logs["samples"] = wandb.Table(
                                columns=["samples", "reward"], rows=pairs[:128]
                            )
                            if os.environ.get("DEBUG"):
                                print(
                                    f"\n".join(
                                        [
                                            f"[{reward:.2f}] {text}"
                                            for text, reward in pairs[:10]
                                        ]
                                    )
                                )
                        else:
                            if os.environ.get("DEBUG"):
                                print(samples)

                        logs["reward"] = reward

                    self.model.train()

                loss, stats = self.model.loss(batch)

                if opt_steps % self.config.train.eval_interval == 0:
                    logs.update(stats)
                    if self.accelerator.is_main_process:
                        self.accelerator.log(logs)
                        self.accelerator.print(
                            "Step: {}, loss_cql: {}, loss_v: {}, reward: {}".format(
                                opt_steps,
                                logs["loss_cql"],
                                logs["loss_v"],
                                logs["reward"],
                            )
                        )

                self.accelerator.backward(loss)
                self.opt.step()
                self.opt.zero_grad()
                self.scheduler.step()
                opt_steps += 1

                if opt_steps % self.config.method.steps_for_target_q_sync == 0:
                    self.model.sync_target_q_heads()
