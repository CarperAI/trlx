import os
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from itertools import chain
from typing import Union, Sequence, Any, TypeVar

from torchtyping import TensorType  # type: ignore
from trlx.data.ilql_types import ILQLBatch
from trlx.data.method_configs import register_method, MethodConfig


import deepspeed  # type: ignore
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from torch import nn
from transformers import AutoModelForCausalLM, PretrainedConfig

import wandb

import megatron

print(dir(megatron), megatron.__file__)
from megatron import print_rank_0, mpu

from attrs import define


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
        return ILQLHeads(hidden_size, vocab_size, self)

    def layer_spec(self, hidden_size: int, vocab_size: int):
        return LayerSpec(ILQLHeads, hidden_size, vocab_size, self)

    def loss(self, x, batch: ILQLBatch):
        logits, qs, target_qs, vs = x
        actions = (
            batch.input_ids[:, 1:].gather(dim=1, index=batch.actions_ixs).unsqueeze(-1)
        )
        bsize, ntokens, dsize = logits.shape

        # compute two separate q-value estimates, to then select minimum values from both
        if self.two_qs:
            Q1 = qs[0].gather(-1, actions).squeeze(-1)
            Q2 = qs[1].gather(-1, actions).squeeze(-1)

            targetQ1 = target_qs[0].gather(-1, actions).squeeze(-1).detach()
            targetQ2 = target_qs[1].gather(-1, actions).squeeze(-1).detach()
            targetQ = torch.minimum(targetQ1, targetQ2)
        else:
            Q = qs.gather(-1, actions).squeeze(-1)
            targetQ = target_qs.gather(-1, actions).squeeze(-1).detach()

        terminal_mask = batch.dones[:, :-1]
        n_nonterminal = max(1, terminal_mask.sum())

        # values of current states
        V = vs[:, :-1].squeeze()
        # values of next states
        Vnext = vs[:, 1:].squeeze() * batch.dones[:, 1:]
        # target to fit Q
        Q_ = batch.rewards + self.gamma * Vnext.detach()

        if self.two_qs:
            loss_q1 = ((Q1 - Q_) * terminal_mask).pow(2).sum() / n_nonterminal
            loss_q2 = ((Q2 - Q_) * terminal_mask).pow(2).sum() / n_nonterminal
            loss_q = loss_q1 + loss_q2
        else:
            loss_q = ((Q - Q_) * terminal_mask).pow(2).sum() / n_nonterminal

        targetQ = targetQ.detach()

        loss_v = (
            (
                (targetQ >= V).int() * self.tau * (targetQ - V).pow(2)
                + (targetQ < V).int() * (1 - self.tau) * (targetQ - V).pow(2)
            )
            * terminal_mask
        ).sum() / n_nonterminal

        if self.two_qs:
            nactions = qs[0].shape[1]
            loss_cql_q1 = (
                F.cross_entropy(
                    qs[0].reshape(-1, dsize),
                    actions.reshape(-1),
                    reduction="none",
                ).reshape(bsize, nactions)
                * terminal_mask
            ).sum() / n_nonterminal
            loss_cql_q2 = (
                F.cross_entropy(
                    qs[1].reshape(-1, dsize),
                    actions.reshape(-1),
                    reduction="none",
                ).reshape(bsize, nactions)
                * terminal_mask
            ).sum() / n_nonterminal
            loss_cql = loss_cql_q1 + loss_cql_q2
        else:
            nactions = qs.shape[1]
            loss_cql = (
                F.cross_entropy(
                    qs.reshape(-1, dsize), actions.reshape(-1), reduction="none"
                ).reshape(bsize, nactions)
                * terminal_mask
            ).sum() / n_nonterminal

        loss_awac = (
            F.cross_entropy(
                logits[:, :-1, :].reshape(-1, dsize),
                batch.input_ids[:, 1:].reshape(-1),
                reduction="none",
            ).reshape(bsize, ntokens - 1)
            * batch.attention_mask[:, 1:]
        ).sum() / batch.attention_mask[:, 1:].sum()

        loss = loss_q + loss_v + self.cql_scale * loss_cql + self.awac_scale * loss_awac
        stats = {
            f"losses/{k}": v
            for k, v in locals().items()
            if k in ["loss", "loss_v", "loss_q", "loss_cql", "loss_awac"]
        }

        return loss, stats


class ILQLHeads(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, config: ILQLConfig):
        super().__init__()

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.v_head = make_head(self.hidden_size, 1)
        self.q1_head = make_head(self.hidden_size, self.vocab_size)
        self.target_q1_head = deepcopy(self.q1_head)
        self.target_q1_head.requires_grad_(False)

        self.config = config

        if self.config.two_qs:
            self.q2_head = make_head(self.hidden_size, self.vocab_size)
            self.target_q2_head = deepcopy(self.q2_head)
            self.target_q2_head.requires_grad_(False)

    def forward(
        self,
        hs: TensorType["N", "T", "C"],  # type: ignore
        states_ixs=None,
        actions_ixs=None,
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

        if self.config.two_qs:
            qs = (self.q1_head(actions_hs), self.q2_head(actions_hs))
            target_qs = (
                self.target_q1_head(actions_hs),
                self.target_q2_head(actions_hs),
            )
        else:
            qs = self.q1_head(actions_hs)
            target_qs = self.target_q1_head(actions_hs)

        vs = self.v_head(states_hs)

        return qs, target_qs, vs

    def _sync_target_q_heads(self, alpha):
        for target_param, copy_param in zip(
            self.target_q1_head.parameters(), self.q1_head.parameters()
        ):
            target_param.data.copy_(
                (alpha * copy_param.data) + (1.0 - alpha) * target_param.data
            )

        if self.config.two_qs:
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
                self.q2_head.parameters() if self.config.two_qs else [],
                self.target_q2_head.parameters() if self.config.two_qs else [],
            )

            with deepspeed.zero.GatheredParameters(list(params), modifier_rank=0):
                if deepspeed.comm.get_rank() == 0:
                    self._sync_target_q_heads(self.config.alpha)
        else:
            self._sync_target_q_heads(self.config.alpha)


from deepspeed.pipe import PipelineModule, LayerSpec, TiedLayerSpec  # type: ignore


class HeadsLayerSpec(LayerSpec):
    def __init__(self, specs: Sequence[LayerSpec]):
        self.branches = specs

    def build(self):
        return Heads(heads=[m.build() for m in self.branches])


class Heads(nn.Module):
    def __init__(self, modules: Sequence[nn.Module]):
        self.branches = modules

    def forward(self, x: torch.Tensor):
        return [m(x) for m in self.branches]


from megatron.model import GPT2ModelPipe


class GPTNeoXWithValueHeads(GPT2ModelPipe):
    def __init__(
        self,
        config: megatron.NeoXArgs,
        ilql_config: ILQLConfig,
    ):
        super().__init__(
            neox_args=config,
            num_tokentypes=0,
            parallel_output=True,
            topology=mpu.get_topology(),
            use_cache=False,
        )

        self.loss_fn = ilql_config.loss
        embedding = self.specs[-1]
        self.specs[-1] = HeadsLayerSpec(
            specs=[
                embedding,
                ilql_config.layer_spec(self.hidden_size, config.padded_vocab_size),
            ],
        )
        PipelineModule.__init__(
            self,
            layers=self.specs,
            loss_fn=self.loss_fn,
            topology=self.__topology__,
            activation_checkpoint_interval=self.activation_checkpoint_interval,
            partition_method=config.pipe_partition_method,
            checkpointable_layers=["GMLPBlock", "ParallelTransformerLayerPipe"],
        )


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
            hs, actions_ixs=actions_ixs, states_ixs=states_ixs
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
        logs=True,
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
