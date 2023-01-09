import os
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from functools import reduce
from itertools import chain
from typing import Union, Sequence, Any, TypeVar, Tuple

from trlx.data.ilql_types import ILQLBatch
from trlx.data.method_configs import register_method, MethodConfig


import deepspeed  # type: ignore
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from torch import nn
from transformers import AutoModelForCausalLM, PretrainedConfig

from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    get_all_params_for_weight_decay_optimization,
    get_params_for_weight_decay_optimization,
)

import wandb


def topk_mask(xs: torch.FloatTensor, k: int):
    if k > xs.shape[-1]:
        return xs
    mintop = torch.topk(xs, k)[0][:, -1].unsqueeze(-1)
    return torch.where(xs < mintop, -np.inf * torch.ones_like(xs, dtype=xs.dtype), xs)


def make_head(n_embd: int, out: int):
    return nn.Sequential(
        nn.Linear(n_embd, n_embd * 2), nn.ReLU(), nn.Linear(n_embd * 2, out)
    )


@dataclass
@register_method
class ILQLConfig(MethodConfig):
    tau: float
    gamma: float
    cql_scale: float
    awac_scale: float
    alpha: float
    steps_for_target_q_sync: float
    betas: Sequence[float]
    two_qs: bool

    def heads(self, hidden_size: int, vocab_size: int):
        return ILQLHeads(self, hidden_size, vocab_size)

    def loss(self, outputs, labels: ILQLBatch):
        logits, (qs, target_qs, vs) = outputs
        actions = (
            labels.input_ids[:, 1:]
            .gather(dim=1, index=labels.actions_ixs)
            .unsqueeze(-1)
        )
        bsize, _, dsize = logits.shape
        ntokens = labels.states_ixs.shape[1]
        Q = [q.gather(-1, actions).squeeze(-1) for q in qs]
        targetQs = [q.gather(-1, actions).squeeze(-1).detach() for q in target_qs]
        targetQ = reduce(torch.minimum, targetQs)
        terminal_mask = labels.dones[:, :-1]
        n_nonterminal = max(1, terminal_mask.sum())

        # values of current states
        V = vs[:, :-1].squeeze()
        # values of next states
        Vnext = vs[:, 1:].squeeze() * labels.dones[:, 1:]
        # target to fit Q
        Q_ = labels.rewards + self.gamma * Vnext.detach()

        loss_qs = [((Qi - Q_) * terminal_mask).pow(2).sum() / n_nonterminal for Qi in Q]
        loss_q = sum(loss_qs)

        targetQ = targetQ.detach()

        loss_v = (
            (
                (targetQ >= V).int() * self.tau * (targetQ - V).pow(2)
                + (targetQ < V).int() * (1 - self.tau) * (targetQ - V).pow(2)
            )
            * terminal_mask
        ).sum() / n_nonterminal

        nactions = qs[0].shape[1]

        def cql_loss(q):
            loss = F.cross_entropy(
                q.reshape(-1, dsize), actions.reshape(-1), reduction="none"
            )
            loss = loss.reshape(bsize, nactions) * terminal_mask
            loss = loss.sum() / n_nonterminal
            return loss

        loss_cql = sum(cql_loss(q) for q in qs)

        states_logits = logits.gather(
            1, index=labels.states_ixs.unsqueeze(-1).repeat(1, 1, logits.shape[-1])
        )
        input_states = labels.input_ids.gather(1, index=labels.states_ixs)

        loss_awac = (
            F.cross_entropy(
                states_logits[:, :-1, :].reshape(-1, dsize),
                input_states[:, 1:].reshape(-1),
                reduction="none",
            ).reshape(bsize, ntokens - 1)
            * labels.attention_mask[:, 1:]
        ).sum() / labels.attention_mask[:, 1:].sum()

        loss = loss_q + loss_v + self.cql_scale * loss_cql + self.awac_scale * loss_awac

        stats = {
            f"losses/{k}": v
            for k, v in locals().items()
            if k in ["loss", "loss_v", "loss_q", "loss_cql", "loss_awac"]
        }

        return loss, stats


class ILQLHeads(nn.Module):
    def __init__(self, config: ILQLConfig, hidden_size: int, vocab_size: int):
        super().__init__()

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.v_head = make_head(self.hidden_size, 1)
        self.config = config

        n_qs = 2 if self.config.two_qs else 1

        self.q_heads = nn.ModuleList(
            make_head(self.hidden_size, self.vocab_size) for _ in range(n_qs)
        )
        self.target_q_heads = nn.ModuleList(deepcopy(q_head) for q_head in self.q_heads)

        # for q_head in self.target_q_heads:
        #    q_head.requires_grad_(False)

    def forward(
        self,
        hs: torch.Tensor,
        states_ixs: torch.Tensor = None,
        actions_ixs: torch.Tensor = None,
        **kwargs,
    ):
        if states_ixs is not None:
            states_hs = hs.gather(
                dim=1, index=states_ixs.unsqueeze(-1).repeat(1, 1, hs.shape[-1])
            ).contiguous()
            actions_hs = hs.gather(
                dim=1, index=actions_ixs.unsqueeze(-1).repeat(1, 1, hs.shape[-1])
            ).contiguous()
        else:
            states_hs = actions_hs = hs

        qs = tuple(q_head(actions_hs) for q_head in self.q_heads)
        target_qs = tuple(q_head(actions_hs) for q_head in self.target_q_heads)
        vs = self.v_head(states_hs)

        return qs, target_qs, vs

    def _sync_target_q_heads(self, alpha):
        for target_q_head, q_head in zip(self.target_q_heads, self.q_heads):
            for target_param, copy_param in zip(
                target_q_head.parameters(), q_head.parameters()
            ):
                target_param.data.copy_(
                    (alpha * copy_param.data) + (1.0 - alpha) * target_param.data
                )

    def sync_target_q_heads(self):
        if os.environ.get("DEEPSPEED_ZERO_STAGE", "0") == "3":
            params = chain(
                chain(q_head.parameters() for q_head in self.q_heads),
                chain(q_head.parameters() for q_head in self.target_q_heads),
            )

            with deepspeed.zero.GatheredParameters(list(params), modifier_rank=0):
                if deepspeed.comm.get_rank() == 0:
                    self._sync_target_q_heads(self.config.alpha)
        else:
            self._sync_target_q_heads(self.config.alpha)


class CausalLMWithValueHeads(nn.Module):
    """This is a wrapper around huggingface AutoModelForCausalLM with two additional scalar heads"""

    def __init__(
        self,
        config: Union[PretrainedConfig, str],
        ilql_config: ILQLConfig,
        num_layers_unfrozen=-1,
    ):
        super().__init__()

        # enable zero3 init within from_pretrained
        if os.environ.get("DEEPSPEED_ZERO_STAGE", "0") == "3":
            config_path = os.environ.get("DEEPSPEED_CONFIG_FILE", "")
            if config_path:
                _hfconfig = transformers.deepspeed.HfDeepSpeedConfig(  # noqa: F841
                    config_path
                )

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

        self.ilql_heads = ilql_config.heads(self.n_embd, self.gpt.config.vocab_size)
        self.ilql_config = ilql_config

    def sync_target_q_heads(self):
        self.ilql_heads.sync_target_q_heads()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        actions_ixs=None,
        states_ixs=None,
    ):
        out = self.gpt.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        hs = out.last_hidden_state

        logits = self.gpt.lm_head(hs)
        qs, target_qs, vs = self.ilql_heads(
            hs, states_ixs=states_ixs, actions_ixs=actions_ixs
        )

        return logits, qs, target_qs, vs, out.past_key_values

    def generate(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        beta=1,
        max_length=32,
        temperature=1,
        top_k=20,
        logit_mask=None,
        pad_token_id=50256,
        eos_token_id=50256,
    ):
        """
        Generates samples akin to hf's `.generate` but with custom logp prepossessing: changing token probabilities as to how advantageous they would be according to value functions estimations.
        """
        if attention_mask is None:
            attention_mask = input_ids.not_equal(pad_token_id)

        if position_ids is None:
            position_ids = attention_mask.cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask.eq(0), 0)

        samples = input_ids.clone()
        tensors = defaultdict(list)
        n_new_tokens = max_length - input_ids.shape[1]

        finished = torch.zeros(
            input_ids.shape[0], 1, dtype=torch.long, device=input_ids.device
        )
        for _ in range(n_new_tokens):
            out = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
            )

            logits, _, target_qs, vs, past_key_values = out
            if self.ilql_config.two_qs:
                qs = torch.minimum(target_qs[0][:, -1, :], target_qs[1][:, -1, :])
            else:
                qs = target_qs[:, -1, :]

            logits = logits[:, -1, :]
            vs = vs[:, -1, :]

            if logit_mask is not None:
                mask = logit_mask[input_ids[:, -1].squeeze().to(logit_mask.device)]
                logits[torch.where(mask)] = -np.inf

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

            if torch.all(finished):
                break

        return samples

    @property
    def dummy_inputs(self):
        return {"input_ids": torch.ones(1, 1, device=self.gpt.device, dtype=torch.long)}

    @property
    def device(self):
        return self.gpt.device
