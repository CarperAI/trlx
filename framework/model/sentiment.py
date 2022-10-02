from typing import Dict, Iterable

from framework.data import RLElement, BatchElement
from framework.data.sentiment import SentimentRLElement

from framework.model import BaseRLModel, register_model
from framework.pipeline.sentiment import SentimentRolloutStorage, SentimentPipeline

from framework.model.nn.ilql_models import QVModel
from framework.utils import rampup_decay, safe_mkdir, Clock, topk_mask

from transformers import AutoTokenizer, AutoConfig
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import numpy as np

from accelerate import Accelerator

@register_model
class SentimentILQLModel(BaseRLModel):
    def __init__(self, config, gpt_config_or_path, logit_mask, train_mode = True):
        super().__init__(config, train_mode)

        self.train_store = SentimentRolloutStorage()
        self.eval_pipeline = SentimentPipeline()
        self.logit_mask = logit_mask

        self.model = QVModel(gpt_config_or_path, config.method)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.train_mode:
            self.opt = torch.optim.AdamW(self.model.parameters(), lr = self.config.train.learning_rate_init)
            self.scheduler = rampup_decay(
                self.config.train.lr_ramp_steps,
                self.config.train.lr_decay_steps,
                self.config.train.learning_rate_target / self.config.train.learning_rate_init,
                self.opt
            )

    def tokenize(self, text):
        text = [self.tokenizer.bos_token + txt for txt in text]
        return self.tokenizer(
            text,
            truncation = True,
            padding = 'max_length',
            max_length = self.max_len,
            return_tensors = "pt",
        )

    def get_components(self) -> Dict[str, any]:
        components = {
            "model" : self.model,
            "opt" : self.opt,
            "scheduler" : self.scheduler} if self.train_mode else {"model" : self.model}
        return components

    def learn(self, log_fn = None, save_fn = None, eval_fn = None):
        # timer = Clock()

        # Make a prep function for loader that processes RLElement
        # Tokenizes text and converts to BatchElement
        def prep(elem : RLElement):
            if isinstance(elem.text[0], torch.Tensor):
                tokens = torch.vstack(elem.text)
                mask = tokens.ne(0).long()
                rewards = elem.score.view(-1, 1).repeat(1, tokens.shape[1])
                rewards[mask == 0] = 0
                rewards = rewards[:, :-1]

                return tokens, mask, rewards
            else:
                tok_out = self.tokenize(elem.text)
                be = BatchElement(tok_out["input_ids"], tok_out["attention_mask"])
                return be, elem.score

        train_dataloader = self.train_store.create_loader(self.config.train.batch_size, prep_fn=prep)
        eval_dataloader = self.eval_pipeline.create_loader(self.config.train.batch_size, shuffle=False)

        tbar = trange(self.config.train.epochs * len(train_dataloader))
        for epoch in range(self.config.train.epochs):
            for batch in train_dataloader:
                if tbar.n % self.config.train.eval_interval == 0:
                    self.model.eval()
                    beta = 1

                    all_samples = []
                    for tokens in eval_dataloader:
                        print(f'{tokens=}')
                        with torch.no_grad():
                            samples, stats = self.model.sample(
                                tokens,
                                beta=1,
                                max_length=self.config.train.gen_size,
                                logit_mask=self.logit_mask
                            )

                        all_samples.append(samples)
                        # logs.update(stats)

                    samples = torch.vstack(all_samples)
                    print(samples)
                    reward = np.mean(self.reward_fn(samples))

                    print(f'{reward=}')
                    self.model.train()

                # timer.tick()

                # tokens = batch.tokens.to(device)
                # masks = batch.masks.to(device)
                # reward = reward.to(device)
                loss, stats = self.model.loss(batch)

                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.grad_clip)
                self.opt.step()
                self.scheduler.step()

                tbar.set_description(f'{loss=:.2f}')
                tbar.update()

                if (tbar.n + 1) % self.config.method.steps_for_target_q_sync == 0:
                    self.model.sync_target_q_heads()

                # intervals = self.intervals(iter)
                # timer.tick(self.config.train.batch_size)

                # if intervals["do_log"]:
                #     total_epochs = self.config.train.epochs
                #     total_iters = len(loader)

                #     loss = loss.item()
                #     ar_loss = ar_loss.item()

                #     sec_per_1k = timer.get_stat(n_samp = 1000, reset = True)
                #     print(f"Epoch [{epoch}/{total_epochs}]: Batch [{iter}/{total_iters}]: " + \
                #         f"Loss {loss:.5f} (Time Per 1k: {sec_per_1k:.2f}s)")
                #     if log_fn is not None:
                #         log_fn({"Train Loss":loss, "Autoregressive CE Loss":ar_loss, "Seconds Per 1k Samples":sec_per_1k})
                # # if intervals["do_save"]:
                #     self.save("./")
                #     if save_fn is not None:
                        # save_fn(self.model)
