import os
from collections import defaultdict
from copy import deepcopy
from itertools import chain
from typing import Union

import deepspeed
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from torch import nn
from transformers import AutoModelForCausalLM, PretrainedConfig

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

        if states_ixs is not None:
            states_hs = hs.gather(
                dim=1, index=states_ixs.unsqueeze(-1).repeat(1, 1, hs.shape[-1])
            )
            actions_hs = hs.gather(
                dim=1, index=actions_ixs.unsqueeze(-1).repeat(1, 1, hs.shape[-1])
            )
        else:
            states_hs = actions_hs = hs

        if self.two_qs:
            qs = (self.q1_head(actions_hs), self.q2_head(actions_hs))
            target_qs = (
                self.target_q1_head(actions_hs),
                self.target_q2_head(actions_hs),
            )
        else:
            qs = self.q1_head(actions_hs)
            target_qs = self.target_q1_head(actions_hs)

        logits = self.gpt.lm_head(hs)
        vs = self.v_head(states_hs)

        return logits, qs, target_qs, vs, out.past_key_values

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
        logs=True,
        pad_token_id=50256,
        eos_token_id=50256,
    ):
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

            if torch.all(finished):
                break

        stats = {}
        for name, xs in tensors.items():
            xs = torch.vstack(xs)
            xs = torch.where(torch.isfinite(xs), xs, 0)

            stats.update(
                {
                    f"tensors/{name}/{beta}": wandb.Histogram(
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
