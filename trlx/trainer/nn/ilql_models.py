import gc
import os
from copy import deepcopy
from dataclasses import dataclass
from functools import reduce
from itertools import chain

import deepspeed  # type: ignore
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from torch import nn

from trlx.data.ilql_types import ILQLBatch
from trlx.data.method_configs import MethodConfig, register_method
from trlx.utils.modeling import (
    PreTrainedModelWrapper,
    flatten_dict,
    get_tensor_stats,
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
    beta: float
    steps_for_target_q_sync: float
    two_qs: bool
    gen_kwargs: dict

    def loss(self, outputs, labels: ILQLBatch):
        logits, (qs, target_qs, vs) = outputs
        terminal_mask = labels.dones[:, :-1]
        n_nonterminal = max(1, terminal_mask.sum())

        actions = (
            labels.input_ids[:, 1:]
            .gather(dim=1, index=labels.actions_ixs)
            .unsqueeze(-1)
        )
        nactions = actions.shape[1]
        bsize, _, dsize = logits.shape

        Q = [q.gather(-1, actions).squeeze(-1) for q in qs]
        targetQs = [q.gather(-1, actions).squeeze(-1).detach() for q in target_qs]
        targetQ = reduce(torch.minimum, targetQs)

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

        def cql_loss(q):
            loss = F.cross_entropy(
                q.reshape(-1, dsize), actions.reshape(-1), reduction="none"
            )
            loss = loss.reshape(bsize, nactions) * terminal_mask
            loss = loss.sum() / n_nonterminal
            return loss

        loss_cql = sum(cql_loss(q) for q in qs)

        # select logits from continuations
        action_logits = logits.gather(
            dim=1, index=labels.actions_ixs.unsqueeze(-1).repeat(1, 1, dsize)
        )
        cross_entropy = F.cross_entropy(
            action_logits.reshape(-1, dsize),
            actions.reshape(-1),
            reduction="none",
        ).reshape(bsize, nactions)

        with torch.no_grad():
            awac_weight = torch.exp(self.beta * (targetQ - V))

        loss_awac = (
            torch.sum(cross_entropy * awac_weight * terminal_mask) / n_nonterminal
        )
        loss = loss_q + loss_v + self.cql_scale * loss_cql + self.awac_scale * loss_awac

        stats = dict(
            losses=dict(
                loss=loss.item(),
                loss_q=loss_q.item(),
                loss_v=loss_v.item(),
                loss_cql=loss_cql.item(),
                loss_awac=loss_awac.item(),
            ),
            values=get_tensor_stats(V, terminal_mask, n_nonterminal),
            qvalues={
                str(ix): get_tensor_stats(Q[ix], terminal_mask, n_nonterminal)
                for ix in range(len(Q))
            },
            awac_weight=get_tensor_stats(awac_weight, terminal_mask, n_nonterminal),
        )

        return loss, flatten_dict(stats)


class ILQLHeads(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        two_qs: bool,
        alpha: float,
        dtype: type,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.two_qs = two_qs
        self.alpha = alpha
        self.v_head = make_head(self.hidden_size, 1, dtype)

        n_qs = 2 if self.two_qs else 1
        self.q_heads = nn.ModuleList(
            make_head(self.hidden_size, self.vocab_size, dtype) for _ in range(n_qs)
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
                    self._sync_target_q_heads(self.alpha)
        else:
            self._sync_target_q_heads(self.alpha)


class AutoModelForCausalLMWithILQLHeads(PreTrainedModelWrapper):
    """An `AutoModel` class wrapper for `transformers` causal models wtih a language
    modeling head and ILQL heads.

    References:
        [1] Snell et al., "Offline RL for Natural Language Generation with Implicit Language Q Learning",
            https://arxiv.org/abs/2206.11871, 2022
    """

    _auto_model_parent_class = transformers.AutoModelForCausalLM
    _supported_modules = ["ilql_heads"]
    _supported_args = ["two_qs", "alpha"]

    def __init__(
        self,
        pretrained_model: transformers.PreTrainedModel,
        two_qs: bool = True,
        alpha: float = 0.99,
    ):
        super().__init__(pretrained_model)
        hidden_size = hf_get_hidden_size(self.pretrained_model.config)
        vocab_size = self.pretrained_model.config.vocab_size
        dtype = next(hf_get_lm_head(self.pretrained_model).parameters()).dtype
        self.two_qs = two_qs
        self.alpha = alpha
        self.ilql_heads = ILQLHeads(
            hidden_size, vocab_size, self.two_qs, self.alpha, dtype=dtype
        )

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        actions_ixs=None,
        states_ixs=None,
    ):
        forward_kwargs = self.get_compatible_forward_kwargs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        forward_kwargs["output_hidden_states"] = True

        outputs = self.pretrained_model(**forward_kwargs)
        qs, target_qs, vs = self.ilql_heads(
            outputs.hidden_states[-1], states_ixs=states_ixs, actions_ixs=actions_ixs
        )

        return outputs.logits, qs, target_qs, vs, outputs.past_key_values

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
        pad_token_id = (
            pad_token_id
            if pad_token_id is not None
            else self.pretrained_model.config.pad_token_id
        )
        eos_token_id = (
            eos_token_id
            if eos_token_id is not None
            else self.pretrained_model.config.eos_token_id
        )

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
            if self.two_qs:
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

    def sync_target_q_heads(self):
        self.ilql_heads.sync_target_q_heads()

    def state_dict(self, *args, **kwargs):
        """
        Returns the state dictionary of the model. We add the state dictionary of the ilql heads
        to the state dictionary of the wrapped model by prepending the key with `ilql_heads.`.
        """
        pretrained_model_state_dict = self.pretrained_model.state_dict(*args, **kwargs)
        ilql_heads_state_dict = self.ilql_heads.state_dict(*args, **kwargs)
        for k, v in ilql_heads_state_dict.items():
            pretrained_model_state_dict[f"ilql_heads.{k}"] = v
        return pretrained_model_state_dict

    def post_init(self, state_dict):
        """
        We add the state dictionary of the ilql heads to the state dictionary of the wrapped model
        by preprending the key with `ilql_heads.`. This function removes the `ilql_heads.` prefix from the
        keys of the value head state dictionary.
        """
        for k in list(state_dict.keys()):
            if "ilql_heads." in k:
                state_dict[k.replace("ilql_heads.", "")] = state_dict.pop(k)
        self.ilql_heads.load_state_dict(state_dict, strict=False)
        del state_dict
        gc.collect()
