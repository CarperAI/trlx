import os
from collections import defaultdict
from copy import deepcopy
from itertools import chain
from typing import NamedTuple, Tuple, Union

import accelerate
import deepspeed
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from accelerate.utils import compute_module_sizes
from torch import nn, tensor
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig
from transformers.tokenization_utils_base import BatchEncoding

import wandb


def topk_mask(xs: torch.FloatTensor, k: int):
    mintop = torch.topk(xs, k)[0][:, -1].unsqueeze(-1)
    return torch.where(xs < mintop, -np.inf * torch.ones_like(xs, dtype=xs.dtype), xs)


def make_head(n_embd: int, out: int):
    return nn.Sequential(
        nn.Linear(n_embd, n_embd * 2), nn.ReLU(), nn.Linear(n_embd * 2, out)
    )


class CausalLMWithValueHeads(nn.Module):
    """This is a wrapper around huggingface AutoModelForCausalLM with two additional scalar heads"""

    def __init__(
        self, config: Union[PretrainedConfig, str], params, num_layers_unfrozen=-1
    ):
        super().__init__()

        # enable zero3 init within from_pretrained
        if os.environ.get("DEEPSPEED_ZERO_STAGE", "0") == "3":
            config_path = os.environ.get("DEEPSPEED_CONFIG_FILE", "")
            if config_path:
                _hfconfig = transformers.deepspeed.HfDeepSpeedConfig(config_path)

        if isinstance(config, PretrainedConfig):
            self.gpt = AutoModelForCausalLM.from_config(config)
        else:
            self.gpt = AutoModelForCausalLM.from_pretrained(config)

        if hasattr(self.gpt, "gpt_neox"):
            self.gpt.transformer = self.gpt.gpt_neox
            self.gpt.lm_head = self.gpt_embed_out
            self.n_embd = self.gpt.config.hidden_size
            gpt_blocks = self.gpt.gpt_neox.layers
        else:
            self.n_embd = self.gpt.config.n_embd
            gpt_blocks = self.gpt.transformer.h

        if num_layers_unfrozen == 0:
            gpt_blocks_to_freeze = list(gpt_blocks)
        elif num_layers_unfrozen > 0:
            gpt_blocks_to_freeze = list(gpt_blocks)[:-num_layers_unfrozen]
        else:
            gpt_blocks_to_freeze = []

        for m in gpt_blocks_to_freeze:
            m.requires_grad_(False)

        self.vocab_size = self.gpt.config.vocab_size
        self.v_head = make_head(self.n_embd, 1)
        self.q1_head = make_head(self.n_embd, self.vocab_size)
        self.target_q1_head = deepcopy(self.q1_head)
        self.target_q1_head.requires_grad_(False)

        self.tau = params.tau
        self.alpha = params.alpha
        self.gamma = params.gamma
        self.awac_scale = params.awac_scale
        self.cql_scale = params.cql_scale
        self.two_qs = params.two_qs

        if self.two_qs:
            self.q2_head = make_head(self.n_embd, self.vocab_size)
            self.target_q2_head = deepcopy(self.q2_head)
            self.target_q2_head.requires_grad_(False)

    def forward(self, **x):
        out = self.gpt.transformer(**x)
        hs = out.last_hidden_state

        if self.two_qs:
            qs = (self.q1_head(hs), self.q2_head(hs))
            target_qs = (self.target_q1_head(hs), self.target_q2_head(hs))
        else:
            qs = self.q1_head(hs)
            target_qs = self.target_q1_head(hs)

        logits = self.gpt.lm_head(hs)
        vs = self.v_head(hs)

        return logits, qs, target_qs, vs, out.past_key_values

    def loss(self, batch):
        tokens = batch.input_ids.to(self.device)
        attn = batch.attention_mask.to(self.device)
        rewards = batch.rewards.to(self.device)

        actions = tokens[:, 1:, None]
        terminal_mask = attn[:, :-1]

        logits, qs, target_qs, vs, _ = self(input_ids=tokens, attention_mask=attn)
        bsize, ntokens, dsize = logits.shape

        if self.two_qs:
            Q1 = qs[0][:, :-1].gather(-1, actions).squeeze(-1)
            Q2 = qs[1][:, :-1].gather(-1, actions).squeeze(-1)

            targetQ1 = target_qs[0][:, :-1].gather(-1, actions).squeeze(-1).detach()
            targetQ2 = target_qs[1][:, :-1].gather(-1, actions).squeeze(-1).detach()
            targetQ = torch.minimum(targetQ1, targetQ2)
        else:
            Q = qs[:, :-1].gather(-1, actions).squeeze(-1)
            targetQ = target_qs[:, :-1].gather(-1, actions).squeeze(-1).detach()

        n_nonterminal = max(1, terminal_mask.sum())
        V = vs[:, 1:].squeeze() * terminal_mask
        Q_ = rewards + self.gamma * V

        if self.two_qs:
            loss_q1 = ((Q1 - Q_.detach()) * terminal_mask).pow(2).sum() / n_nonterminal
            loss_q2 = ((Q2 - Q_.detach()) * terminal_mask).pow(2).sum() / n_nonterminal
            loss_q = loss_q1 + loss_q2
        else:
            loss_q = ((Q - Q_.detach()) * terminal_mask).pow(2).sum() / n_nonterminal

        loss_v = (
            (
                (targetQ >= V).int() * self.tau * (targetQ - V).pow(2)
                + (targetQ < V).int() * (1 - self.tau) * (targetQ - V).pow(2)
            )
            * terminal_mask
        ).sum() / n_nonterminal

        if self.two_qs:
            loss_cql_q1 = (
                F.cross_entropy(
                    qs[0][:, :-1].reshape(-1, dsize),
                    actions.reshape(-1),
                    reduction="none",
                ).reshape(bsize, ntokens - 1)
                * terminal_mask
            ).sum() / n_nonterminal
            loss_cql_q2 = (
                F.cross_entropy(
                    qs[1][:, :-1].reshape(-1, dsize),
                    actions.reshape(-1),
                    reduction="none",
                ).reshape(bsize, ntokens - 1)
                * terminal_mask
            ).sum() / n_nonterminal
            loss_cql = loss_cql_q1 + loss_cql_q2
        else:
            loss_cql = (
                F.cross_entropy(
                    qs[:, :-1].reshape(-1, dsize), actions.reshape(-1), reduction="none"
                ).reshape(bsize, ntokens - 1)
                * terminal_mask
            ).sum() / n_nonterminal

        loss_awac = (
            F.cross_entropy(
                logits[:, :-1].reshape(-1, dsize), actions.reshape(-1), reduction="none"
            ).reshape(bsize, ntokens - 1)
            * terminal_mask
        ).sum() / n_nonterminal

        loss = loss_q + loss_v + self.cql_scale * loss_cql + self.awac_scale * loss_awac
        stats = {
            f"losses/{k}": v
            for k, v in locals().items()
            if k in ["loss", "loss_v", "loss_q", "loss_cql", "loss_awac"]
        }

        return loss, stats

    def _sync_target_q_heads(self, alpha):
        for target_param, copy_param in zip(
            self.target_q1_head.parameters(), self.q1_head.parameters()
        ):
            target_param.data.copy_(
                (alpha * copy_param.data) + (1.0 - alpha) * target_param.data
            )

        if self.two_qs:
            for target_param, copy_param in zip(
                self.target_q2_head.parameters(), self.q2_head.parameters()
            ):
                target_param.data.copy_(
                    (alpha * copy_param.data) + (1.0 - alpha) * target_param.data
                )

    def sync_target_q_heads(self):
        if os.environ.get("DEEPSPEED_ZERO_STAGE", "0") == "3":
            params = chain(
                self.q1_head.parameters(),
                self.target_q1_head.parameters(),
                self.q2_head.parameters() if self.two_qs else [],
                self.target_q2_head.parameters() if self.two_qs else [],
            )

            with deepspeed.zero.GatheredParameters(list(params), modifier_rank=0):
                if deepspeed.comm.get_rank() == 0:
                    self._sync_target_q_heads(self.alpha)
        else:
            self._sync_target_q_heads(self.alpha)

    @torch.inference_mode()
    def sample(
        self,
        prompts,
        beta=1,
        max_length=32,
        temperature=1,
        top_k=20,
        logit_mask=None,
        logs=True,
        eos_token_id=50256,
    ):
        if isinstance(prompts, (dict, BatchEncoding)):
            input_ids = prompts.get("input_ids")
            attention_mask = prompts.get("attention_mask", None)
        else:
            input_ids = prompts
            attention_mask = None

        if attention_mask is None:
            attention_mask = input_ids.not_equal(eos_token_id)

        samples = input_ids.clone()

        past_key_values = None
        tensors = defaultdict(list)
        n_new_tokens = max_length - input_ids.shape[1]

        finished = torch.zeros(
            input_ids.shape[0], 1, dtype=torch.long, device=input_ids.device
        )
        position_ids = attention_mask.cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask.eq(0), 0)

        for _ in range(n_new_tokens):
            out = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
            )

            logits, _, target_qs, vs, past_key_values = out
            if self.two_qs:
                qs = torch.minimum(target_qs[0][:, -1, :], target_qs[1][:, -1, :])
            else:
                qs = target_qs[:, -1, :]

            logits = logits[:, -1, :]
            vs = vs[:, -1, :]

            if logit_mask is not None:
                logits[torch.where(logit_mask[input_ids[:, -1].squeeze()])] = -np.inf

            adv = qs - vs
            pi_beta = F.log_softmax(logits, -1)
            pi_top_k = topk_mask(pi_beta + beta * adv, top_k)
            pi = F.softmax(pi_top_k / temperature, -1)

            input_ids = torch.multinomial(pi, num_samples=1)
            input_ids = (1 - finished) * input_ids + finished * eos_token_id
            finished = (input_ids == eos_token_id).long()

            samples = torch.hstack((samples, input_ids))
            attention_mask = torch.hstack(
                (attention_mask, (input_ids != eos_token_id).long())
            )
            position_ids = (position_ids[:, -1] + 1).view(-1, 1)

            if logs:
                tensors["qs"].append(qs)
                tensors["vs"].append(vs)
                tensors["adv"].append(adv)
                tensors["pi"].append(pi)
                tensors["pi_beta"].append(torch.exp(pi_beta))

            if torch.all(finished):
                break

        stats = {}
        for name, xs in tensors.items():
            xs = torch.vstack(xs)
            xs = torch.where(torch.isfinite(xs), xs, 0)

            stats.update(
                {
                    f"tensors/{name}/min/{beta}": xs.min(),
                    f"tensors/{name}/max/{beta}": xs.max(),
                    f"tensors/{name}/std/{beta}": xs.std(),
                    f"tensors/{name}/mean/{beta}": xs.mean(),
                    f"tensors/{name}/hist/{beta}": wandb.Histogram(
                        xs.cpu().float().view(-1)
                    ),
                }
            )

        return samples, stats

    @property
    def dummy_inputs(self):
        return {"input_ids": torch.ones(1, 1, device=self.gpt.device, dtype=torch.long)}

    @property
    def device(self):
        return self.gpt.device
