from typing import Dict

from framework.data import RLElement
from framework.data.sentiment import SentimentRLElement

from framework.model import BaseRLModel, register_model
from framework.pipeline.sentiment import SentimentRolloutStorage

from framework.model.nn import QVModel
from framework.utils import rampup_decay, safe_mkdir

from transformers import AutoTokenizer, AutoConfig
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import einops as eo

@register_model
class SentimentILQLModel(BaseRLModel):
    def __init__(self, config, train_mode = False):
        super().__init__(config, train_mode)

        self.store = SentimentRolloutStorage()

        self.model = QVModel(self.config.model.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.model_path)
        self.model_config = AutoConfig.from_pretrained(self.config.model.model_path)

        # All parameters specific to ILQL, move to config later? Not sure how
        self.tau = 0.7
        self.cql_scale = 1.0e-4
        self.sync_every = 50
        self.max_len = 512

        if self.train_mode:
            self.opt = torch.optim.AdamW(self.model.parameters(), lr = self.config.train.learning_rate_init)
            self.scheduler = rampup_decay(
                self.config.train.lr_ramp_steps,
                self.config.train.lr_decay_steps,
                self.config.train.learning_rate_target / self.config.train.learning_rate_init,
                self.opt
            )
    
    def tokenize(self, text):
        return self.tokenizer(
            text,
            truncation = True,
            padding = True,
            max_length = self.max_len,
            return_tensors = "pt"
        )

    def act(self, data : SentimentRLElement) -> SentimentRLElement:
        tok_out = self.tokenize(data.text)
        model_out = self.model(input_ids = tok_out["input_ids"], attention_mask = tok_out["attention_mask"])
        action = model_out["q_out"].argmax(-1)
        data.action = action
        return data


    def get_components(self) -> Dict[str, any]:
        components = {
            "model" : self.model,
            "opt" : self.opt,
            "scheduler" : self.scheduler} if self.train_mode else {"model" : self.model}
        return components
    
    def learn(self, log_fn = None, save_fn = None, eval_fn = None):
        loader = self.store.create_loader(self.config.train.batch_size, shuffle = True)

        def get_loss(input_tokens, attn, rewards):
            # Rewards are [batch], wheras states are [batch, seq_len]
            # Need rewards to be [batch, seq_len - 1] to match value outputs
            # Rewards for non terminal token will be 0
            _rewards = torch.zeros(len(rewards), self.max_len - 1)
            _rewards[:,-1] = rewards
            rewards = _rewards

            model_out = self.model(input_ids = input_tokens, attention_mask = attn)

            logits = model_out["lm_out"]

            q_out = model_out["q_out"]
            Q = q_out[:,:-1,:].gather(-1, input[:,1:,None]).squeeze(-1)

            target_q_out = model_out["target_q_out"]
            targ_Q = target_q_out[:,:-1,:].gather(-1, input[:,1:,None]).squeeze(-1).detach()

            v_out = model_out["v_out"]
            V = v_out[:,1:].squeeze() * attn[:,1:]
            Q_star = rewards + V

            loss_q = (Q - Q_star.detach()).pow(2).mean() # MSE(Q,Q*)
            loss_v = (((targ_Q >= V).int() * self.tau * (targ_Q - V).pow(2) + \
                (targ_Q < V).int() * (1 - self.tau) * (targ_Q - V).pow(2)) * attn[:,1:]).mean()

            # CQL keeps Q output in line with expected LM output
            loss_cql = F.cross_entropy(
                q_out[:,:-1,:].reshape(-1, q_out.size(-1)),
                input_tokens[:,1:].reshape(-1)
            )

            # What "expected LM output" referred to above is being trained on
            loss_awac = F.cross_entropy(
                logits[:,:-1,:].reshape(-1, logits.size(-1)),
                input_tokens[:,1:].reshape(-1)
            )

            loss = loss_q + loss_v + loss_awac + self.cql_scale * loss_cql
            return loss 
        
        for epoch in range(self.config.train.epochs):
            for iter, batch in enumerate(loader):
                tok_out = self.tokenize(batch.text)

                loss = get_loss(tok_out["input_ids"], tok_out["attention_mask"], batch.score)

                # Likely where stuff has to change
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.grad_clip)
                self.opt.step()
                self.scheduler.step()

                if (iter + 1) % self.sync_every == 0:
                    self.model.sync_target(1)
                
                intervals = self.intervals(iter)
                if intervals["do_log"]:
                    print(f"Epoch [{epoch}/{self.config.train.epochs}]: Batch [{iter}/{len(loader)}]: Loss {loss.item()}")
                if intervals["do_save"]:
                    self.save("./")





