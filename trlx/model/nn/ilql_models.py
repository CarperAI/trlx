import os
from collections import defaultdict
from copy import deepcopy
from itertools import chain
from typing import NamedTuple, Tuple, Union

import accelerate
import deepspeed
import numpy as np
import torch as th
import torch.nn.functional as F
import transformers
from accelerate.utils import compute_module_sizes
from torch import nn, tensor
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig


def topk_mask(xs: th.FloatTensor, k: int):
	mintop = th.topk(xs, k)[0][:, -1].unsqueeze(-1)
	return th.where(xs < mintop, -np.inf * th.ones_like(xs, dtype=xs.dtype), xs)


class QVOutput(Tuple):
	logits: th.FloatTensor
	qs: th.FloatTensor
	target_qs: th.FloatTensor
	vs: th.FloatTensor
	past_key_values: Tuple[th.FloatTensor]


def make_head(n_embd: int, out: int):
	return nn.Sequential(
		nn.Linear(n_embd, n_embd * 2), nn.ReLU(), nn.Linear(n_embd * 2, out)
	)


class QVModel(nn.Module):
	def __init__(self, config: Union[PretrainedConfig, str], params):
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

		for block in self.gpt.transformer.h:
			block.requires_grad_(False)

		if hasattr(self.gpt.config, "hidden_size"):
			self.n_embd = self.gpt.config.hidden_size
		else:
			self.n_embd = self.gpt.config.n_embd
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
		self.beta = params.beta

		if self.two_qs:
			self.q2_head = make_head(self.n_embd, self.vocab_size)
			self.target_q2_head = deepcopy(self.q2_head)
			self.target_q2_head.requires_grad_(False)

	def forward(self, **x):
		if hasattr(self.gpt, "gpt_neox"):
			out = self.gpt.gpt_neox(**x)
		else:
			out = self.gpt.transformer(**x)

		hs = out.last_hidden_state

		if self.two_qs:
			qs = (self.q1_head(hs), self.q2_head(hs))
			target_qs = (self.target_q1_head(hs), self.target_q2_head(hs))
		else:
			qs = self.q1_head(hs)
			target_qs = self.target_q1_head(hs)

		if hasattr(self.gpt, "gpt_neox"):
			logits = self.gpt.embed_out(hs)
		else:
			logits = self.gpt.lm_head(hs)

		return QVOutput((logits, qs, target_qs, self.v_head(hs), out.past_key_values))

	def loss(self, batch):
		tokens = batch.input_ids.to(self.device)
		attn = batch.attention_mask.to(self.device)
		rewards = batch.rewards.to(self.device)

		actions = tokens[:, 1:, None]
		isterminal = attn[:, :-1]

		logits, qs, target_qs, vs, _ = self(input_ids=tokens, attention_mask=attn)
		bsize, ntokens, dsize = logits.shape

		if self.two_qs:
			Q1 = qs[0][:, :-1].gather(-1, actions).squeeze(-1)
			Q2 = qs[1][:, :-1].gather(-1, actions).squeeze(-1)

			targetQ1 = target_qs[0][:, :-1].gather(-1, actions).squeeze(-1).detach()
			targetQ2 = target_qs[1][:, :-1].gather(-1, actions).squeeze(-1).detach()
			targetQ = th.minimum(targetQ1, targetQ2)
		else:
			Q = qs[:, :-1].gather(-1, actions).squeeze(-1)
			targetQ = target_qs[:, :-1].gather(-1, actions).squeeze(-1).detach()

		n_nonterminal = max(1, isterminal.sum())
		V = vs[:, 1:].squeeze() * isterminal
		Q_ = rewards + self.gamma * V

		if self.two_qs:
			loss_q1 = ((Q1 - Q_.detach()) * isterminal).pow(2).sum() / n_nonterminal
			loss_q2 = ((Q2 - Q_.detach()) * isterminal).pow(2).sum() / n_nonterminal
			loss_q = loss_q1 + loss_q2
		else:
			loss_q = ((Q - Q_.detach()) * isterminal).pow(2).sum() / n_nonterminal

		loss_v = (
			(
				(targetQ >= V).int() * self.tau * (targetQ - V).pow(2)
				+ (targetQ < V).int() * (1 - self.tau) * (targetQ - V).pow(2)
			)
			* isterminal
		).sum() / n_nonterminal

		if self.two_qs:
			loss_cql_q1 = (
				F.cross_entropy(
					qs[0][:, :-1].reshape(-1, dsize),
					actions.reshape(-1),
					reduction="none",
				).reshape(bsize, ntokens - 1)
				* isterminal
			).sum() / n_nonterminal
			loss_cql_q2 = (
				F.cross_entropy(
					qs[1][:, :-1].reshape(-1, dsize),
					actions.reshape(-1),
					reduction="none",
				).reshape(bsize, ntokens - 1)
				* isterminal
			).sum() / n_nonterminal
			loss_cql = loss_cql_q1 + loss_cql_q2
		else:
			loss_cql = (
				F.cross_entropy(
					qs[:, :-1].reshape(-1, dsize), actions.reshape(-1), reduction="none"
				).reshape(bsize, ntokens - 1)
				* isterminal
			).sum() / n_nonterminal

		loss_awac = (
			F.cross_entropy(
				logits[:, :-1].reshape(-1, dsize), actions.reshape(-1), reduction="none"
			).reshape(bsize, ntokens - 1)
			* isterminal
		).sum() / n_nonterminal

		loss = loss_q + loss_v + self.cql_scale * loss_cql + self.awac_scale * loss_awac
		stats = {
			k: v
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

	@th.inference_mode()
	def sample(
		self,
		query,
		beta=1,
		max_length=32,
		temperature=1,
		top_k=20,
		logit_mask=None,
		logs=True,
		eos_token_id=50256,
	):
		input = query.clone()
		past_key_values = None
		tensors = defaultdict(list)

		finished = th.zeros(input.shape[0], 1, dtype=th.long, device=query.device)

		for _ in range(max_length - 1):
			logits, _, target_qs, vs, past_key_values = self.forward(
				input_ids=input, past_key_values=past_key_values
			)

			if self.two_qs:
				qs = th.minimum(target_qs[0][:, -1], target_qs[1][:, -1])
			else:
				qs = target_qs[:, -1]

			logits = logits[:, -1]

			if logit_mask is not None:
				logits[th.where(logit_mask[input[:, -1]])] = -np.inf

			adv = qs - vs[:, -1, :]
			pi = F.log_softmax(logits, -1)
			modpi = topk_mask(pi + beta * adv, top_k)
			ps = F.softmax(modpi / temperature, -1)

			tokens = th.multinomial(ps, 1)
			tokens = (1 - finished) * tokens + finished * eos_token_id

			query = th.hstack((query, tokens))

			input = tokens
			finished = (tokens == eos_token_id).long()

			if logs:
				tensors["qs"].append(qs)
				tensors["vs"].append(vs)
				tensors["adv"].append(adv)

		stats = {}
		for name, xs in tensors.items():
			xs = th.vstack(xs)
			stats.update(
				{
					f"{name}-min": xs.min(),
					f"{name}-max": xs.max(),
					f"{name}-std": xs.std(),
					f"{name}-avg": xs.mean(),
				}
			)

		return query, stats

	@property
	def dummy_inputs(self):
		return {"input_ids": th.ones(1, 1, device=self.gpt.device, dtype=th.long)}

	@property
	def device(self):
		return self.gpt.device
