import inspect
import os
from copy import deepcopy
from dataclasses import dataclass
from functools import reduce
from itertools import chain
from typing import Any, Dict, Union

import deepspeed  # type: ignore
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from torch import nn

from trlx.data.ilql_types import ILQLBatch
from trlx.data.method_configs import MethodConfig, register_method
from trlx.utils.modeling import (
    freeze_bottom_causal_layers,
    hf_get_causal_base_model,
    hf_get_hidden_size,
    hf_get_lm_head,
    make_head,
)


def topk_mask(xs: torch.FloatTensor, k: int):
    if k > xs.shape[-1]:
        return xs
    mintop = torch.topk(xs, k)[0][:, -1].unsqueeze(-1)
    return torch.where(xs < mintop, -np.inf * torch.ones_like(xs, dtype=xs.dtype), xs)


@dataclass
@register_method
class ILQLConfig(MethodConfig):
    tau: float
    gamma: float
    cql_scale: float
    awac_scale: float
    alpha: float
    steps_for_target_q_sync: float
    two_qs: bool
    gen_kwargs: dict

    def heads(self, hidden_size: int, vocab_size: int):
        return ILQLHeads(self, hidden_size, vocab_size)

    def loss(self, outputs, labels: ILQLBatch):
        logits, (qs, target_qs, vs) = outputs
        actions = (
            labels.input_ids[:, 1:]
            .gather(dim=1, index=labels.actions_ixs)
            .unsqueeze(-1)
        )
        bsize, ntokens, dsize = logits.shape

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

        loss_awac = (
            F.cross_entropy(
                logits[:, :-1, :].reshape(-1, dsize),
                labels.input_ids[:, 1:].reshape(-1),
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

        for q_head in self.target_q_heads:
            q_head.requires_grad_(False)

    def forward(
        self,
        hs: torch.Tensor,
        states_ixs: torch.Tensor = None,
        actions_ixs: torch.Tensor = None,
    ):
        if states_ixs is not None:
            states_hs = hs.gather(
                dim=1, index=states_ixs.unsqueeze(-1).repeat(1, 1, hs.shape[-1])
            )
            actions_hs = hs.gather(
                dim=1, index=actions_ixs.unsqueeze(-1).repeat(1, 1, hs.shape[-1])
            )
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
        config: Union[transformers.PretrainedConfig, str],
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

        if isinstance(config, str):
            self.config = transformers.AutoConfig.from_pretrained(config)
            self.base_model = transformers.AutoModelForCausalLM.from_pretrained(config)
        else:
            self.config = config
            self.base_model = transformers.AutoModelForCausalLM.from_config(config)

        self.base_model.transformer = hf_get_causal_base_model(self.base_model)
        self.base_model.lm_head = hf_get_lm_head(self.base_model)
        freeze_bottom_causal_layers(self.base_model, num_layers_unfrozen)

        # Cache `transformer.forward` args for general use (avoids incompatible args across architectures)
        self.base_model_transformer_args = inspect.getfullargspec(
            self.base_model.transformer.forward
        ).args

        self.hidden_size = hf_get_hidden_size(self.config)
        self.ilql_heads = ilql_config.heads(self.hidden_size, self.config.vocab_size)
        self.ilql_config = ilql_config

    def _get_compatible_forward_kwargs(self, **kwargs) -> Dict[str, Any]:
        """Filter out arguments not supported by the specific instance of `base_model.transformer.forward`"""
        return {
            k: v for k, v in kwargs.items() if k in self.base_model_transformer_args
        }

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
        forward_kwargs = self._get_compatible_forward_kwargs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        out = self.base_model.transformer(**forward_kwargs)
        hs = out.last_hidden_state

        logits = self.base_model.lm_head(hs)
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
        max_new_tokens=32,
        max_length=1024,
        temperature=1,
        top_k=20,
        logit_mask=None,
        pad_token_id=None,
        eos_token_id=None,
    ):
        """
        Generates samples akin to hf's `.generate` but with custom logp prepossessing:
        changing token probabilities as to how advantageous they would be
        according to value functions estimations.
        """
        if attention_mask is None:
            attention_mask = input_ids.not_equal(pad_token_id)

        if position_ids is None:
            position_ids = attention_mask.cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask.eq(0), 0)

        samples = input_ids.clone()
        max_new_tokens = min(max_new_tokens, max_length - input_ids.shape[1])

        finished = torch.zeros(
            input_ids.shape[0], 1, dtype=torch.long, device=input_ids.device
        )
        for _ in range(max_new_tokens):
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
        return {
            "input_ids": torch.ones(
                1, 1, device=self.base_model.device, dtype=torch.long
            )
        }

    @property
    def device(self):
        return self.base_model.device
