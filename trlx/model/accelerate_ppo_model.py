from abc import abstractmethod
from typing import Dict, Iterable, Tuple
from trlx.data.accelerate_base_datatypes import PromptBatch
from trlx.data.configs import TRLConfig

from trlx.model import BaseRLModel, register_model
from trlx.model.accelerate_base_model import AccelerateRLModel
from trlx.model.nn.ppo_models import GPT2HeadWithValueModel
from trlx.pipeline.ppo_pipeline import PPORolloutStorage

from trlx.utils import rampup_decay, safe_mkdir, Clock, topk_mask

from transformers import AutoTokenizer, AutoConfig
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from accelerate import Accelerator

from torchtyping import TensorType

from trlx.utils.modeling import clip_by_value, logprobs_from_logits, whiten
from tqdm import tqdm

@register_model
class AcceleratePPOModel(AccelerateRLModel):
    def __init__(self, config, train_mode = True):
        self.store = PPORolloutStorage()
        super().__init__(config, self.store)

    def get_arch(self, config: TRLConfig):
        # TODO(dahoas): Assumes model is gpt2 based
        return GPT2HeadWithValueModel.from_pretrained(self.config.model.model_path)

    def loss(self, query_tensors, response_tensors, all_logprobs, all_values, all_rewards):
        lastgaelam = 0
        advantages_reversed = []
        gen_len = response_tensors.shape[1]
        for t in reversed(range(gen_len)):
            nextvalues = all_values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = all_rewards[:, t] + self.config.method.gamma * nextvalues - all_values[:, t]
            lastgaelam = delta + self.config.method.gamma * self.config.method.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)

        returns = advantages + all_values
        advantages = whiten(advantages)
        advantages = advantages.detach()

        all_tokens = torch.cat((query_tensors, response_tensors), dim=1)
        logits, _, vpred = self.model(all_tokens)
        logprob = logprobs_from_logits(logits[:,:-1,:], all_tokens[:, 1:])

        #only the generation part of the values/logprobs is needed
        logprob, vpred = logprob[:, -gen_len:], vpred[:,-gen_len-1:-1]

        vpredclipped = clip_by_value(vpred,
                                    all_values - self.config.method.cliprange_value,
                                    all_values + self.config.method.cliprange_value)

        vf_losses1 = (vpred - returns)**2
        vf_losses2 = (vpredclipped - returns)**2
        vf_loss = .5 * torch.mean(torch.max(vf_losses1, vf_losses2))

        ratio = torch.exp(logprob - all_logprobs)

        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio,
                                            1.0 - self.config.method.cliprange,
                                            1.0 + self.config.method.cliprange)

        pg_loss = torch.mean(torch.max(pg_losses, pg_losses2))

        model_loss = pg_loss + self.config.method.vf_coef * vf_loss
        return model_loss

    def post_epoch_callback(self):
        #TODO(dahoas): are experiences being made for dataloaders on each process or same dataloader
        self.epoch += 1
        self.store.clear_history()
        self.orch.make_experience(self.config.method.num_rollouts, self.iter_count)  # Collect more rollouts for training

    def post_backward_callback(self, batch, rewards):
        pass

    def learn(self, log_fn = None, save_fn = None, eval_fn = None):

        rollout_loader = self.store.create_loader(self.config.train.batch_size, shuffle = True, prep_fn = None, num_workers = 2)
        rollout_loader = self.accelerator.prepare(rollout_loader)

        self.iter_count = 0
        self.epoch = 0
        while self.iter_count < self.config.train.total_steps or self.epoch <= self.config.train.epochs:
            for batch in rollout_loader:

                query_tensors = batch.query_tensors.to(self.accelerator.device)
                response_tensors = batch.response_tensors.to(self.accelerator.device)
                logprobs = batch.logprobs.to(self.accelerator.device)
                values = batch.values.to(self.accelerator.device)
                rewards = batch.rewards.to(self.accelerator.device)

                for _ in range(self.config.method.ppo_epochs):
                    loss = self.loss(query_tensors, response_tensors, logprobs, values, rewards)
                    self.opt.zero_grad()
                    self.accelerator.backward(loss)
                    self.opt.step()
                    self.scheduler.step()
                    self.iter_count += 1

                self.post_backward_callback(batch, rewards)

                self.accelerator.wait_for_everyone()

            self.post_epoch_callback()
            self.accelerator.wait_for_everyone()