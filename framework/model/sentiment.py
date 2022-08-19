from typing import Dict, Iterable

from framework.data import RLElement, BatchElement
from framework.data.sentiment import SentimentRLElement

from framework.model import BaseRLModel, register_model
from framework.pipeline.sentiment import SentimentRolloutStorage

from framework.model.nn import QVModel
from framework.utils import rampup_decay, safe_mkdir, Clock, topk_mask

from transformers import AutoTokenizer, AutoConfig
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from accelerate import Accelerator

import einops as eo

@register_model
class SentimentILQLModel(BaseRLModel):
    def __init__(self, config, train_mode = True):
        super().__init__(config, train_mode)

        self.store = SentimentRolloutStorage()

        self.model = QVModel(self.config.model.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.model_path)

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'

        self.model_config = AutoConfig.from_pretrained(self.config.model.model_path)
        self.device = self.config.model.device

        self.model.to(self.device)

        # All parameters specific to ILQL, move to config later? Not sure how
        self.tau = 0.7
        self.beta = 1
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
        text = [self.tokenizer.bos_token + txt for txt in text]
        return self.tokenizer(
            text,
            truncation = True,
            padding = 'max_length',
            max_length = self.max_len,
            return_tensors = "pt",
        )

    def act(self, data : SentimentRLElement) -> SentimentRLElement:
        tok_out = self.tokenize(data.text)
        model_out = self.model(input_ids = tok_out["input_ids"], attention_mask = tok_out["attention_mask"])
        action = model_out["q_out"].argmax(-1)
        data.action = action
        return data

    @torch.inference_mode()
    def sample(self, prompts : Iterable[str] = None, length : int = 32, n_samples = 32) -> Iterable[str]:
        if prompts is None:
            prompts = [self.tokenizer.bos_token] * n_samples
        
        query = self.tokenize(prompts)["input_ids"].to(self.device) # [B, N]
        gen_tokens = torch.zeros(len(query), 0, dtype = torch.long, device = self.device) # generated tokens [B,0] at start

        for _ in range(length):
            # Get outputs for last token in sequence
            model_out = self.model(input_ids = query)
            logits, qs, vs = [model_out[k][:, -1, :] for k in ["lm_out", "q_out", "v_out"]]

            adv = qs - vs
            pi = F.log_softmax(logits, -1)
            modpi = topk_mask(pi + self.beta * adv, 10)
            ps = F.softmax(modpi, -1)

            tokens = torch.multinomial(ps, 1)
            query = torch.cat([query[:,1:], tokens], dim = -1) # [B, N]
            gen_tokens = torch.cat([gen_tokens, tokens], dim = -1)
        
        res = [prompt + self.tokenizer.decode(gen) for (prompt, gen) in zip(prompts, gen_tokens)]
        return res

    def get_components(self) -> Dict[str, any]:
        components = {
            "model" : self.model,
            "opt" : self.opt,
            "scheduler" : self.scheduler} if self.train_mode else {"model" : self.model}
        return components
    
    def learn(self, log_fn = None, save_fn = None, eval_fn = None):
        device = self.device
        timer = Clock()

        # Make a prep function for loader that processes RLElement
        # Tokenizes text and converts to BatchElement
        def prep(elem : RLElement):
            tok_out = self.tokenize(elem.text)
            be = BatchElement(tok_out["input_ids"], tok_out["attention_mask"])
            return be, elem.score

        loader = self.store.create_loader(self.config.train.batch_size, shuffle = True, prep_fn = prep, num_workers = 1)

        def get_loss(input_tokens, attn, rewards):
            # Rewards are [batch], wheras states are [batch, seq_len]
            # Need rewards to be [batch, seq_len - 1] to match value outputs
            # Rewards for non terminal token will be 0
            _rewards = torch.zeros(len(rewards), self.max_len - 1, device = rewards.device)
            _rewards[:,-1] = rewards
            rewards = _rewards

            model_out = self.model(input_ids = input_tokens, attention_mask = attn)

            logits = model_out["lm_out"]

            q_out = model_out["q_out"]
            Q = q_out[:,:-1,:].gather(-1, input_tokens[:,1:,None]).squeeze(-1)

            target_q_out = model_out["target_q_out"]
            targ_Q = target_q_out[:,:-1,:].gather(-1, input_tokens[:,1:,None]).squeeze(-1).detach()

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
            return loss, loss_awac # overall less and loss for the AR task
        
        for epoch in range(self.config.train.epochs):
            for iter, (batch, reward) in enumerate(loader):
                timer.tick()

                tokens = batch.tokens.to(device)
                masks = batch.masks.to(device)
                reward = reward.to(device)
                loss, ar_loss = get_loss(tokens, masks, reward)

                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.grad_clip)
                self.opt.step()
                self.scheduler.step()

                if (iter + 1) % self.sync_every == 0:
                    self.model.sync_target(1)
                
                intervals = self.intervals(iter)
                timer.tick(self.config.train.batch_size)

                if intervals["do_log"]:
                    total_epochs = self.config.train.epochs
                    total_iters = len(loader)

                    loss = loss.item()
                    ar_loss = ar_loss.item()

                    sec_per_1k = timer.get_stat(n_samp = 1000, reset = True)
                    print(f"Epoch [{epoch}/{total_epochs}]: Batch [{iter}/{total_iters}]: " + \
                        f"Loss {loss:.5f} (Time Per 1k: {sec_per_1k:.2f}s)")
                    if log_fn is not None:
                        log_fn({"Train Loss":loss, "Autoregressive CE Loss":ar_loss, "Seconds Per 1k Samples":sec_per_1k})
                if intervals["do_save"]:
                    self.save("./")
                    if save_fn is not None:
                        save_fn(self.model)
                if intervals["do_eval"]:
                    if eval_fn is not None:
                        eval_fn(self)






