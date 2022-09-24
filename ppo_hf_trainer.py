from transformers import Trainer
import torch
import torch.nn.functional as F
import wandb
import numpy as np


def clip_by_value(x, tensor_min, tensor_max):
    """
    Tensor extenstion to torch.clamp
    https://github.com/pytorch/pytorch/issues/2793#issuecomment-428784713
    """
    clipped = torch.max(torch.min(x, tensor_max), tensor_min)
    return clipped

def whiten(values, shift_mean=True):
	"""Whiten values."""
	gen_size = values.size()[-1]
	mean, var = torch.mean(values, dim=1), torch.var(values, dim=1)
	whitened = (values - mean.repeat_interleave(gen_size).view(-1, gen_size)) * torch.rsqrt(var.repeat_interleave(gen_size).view(-1, gen_size) + 1e-8)
	if not shift_mean:
		whitened += mean
	return whitened


class AdaptiveKLController:
	"""
	Adaptive KL controller described in the paper:
	https://arxiv.org/pdf/1909.08593.pdf
	"""
	def __init__(self, init_kl_coef, target, horizon):
		self.value = init_kl_coef
		self.target = target
		self.horizon = horizon

	def update(self, current, n_steps):
		target = self.target
		proportional_error = np.clip(current / target - 1, -0.2, 0.2)
		mult = 1 + proportional_error * n_steps / self.horizon
		self.value *= mult


class PPOTrainer(Trainer):
	def init_ppo_params(self, ppo_params, sent_kwargs, gen_kwargs, ref_model, tokenizer, sentiment_pipe):
		self.ppo_params = ppo_params
		self.ref_model = ref_model
		self.sent_kwargs = sent_kwargs
		self.gen_kwargs = gen_kwargs
		self.tokenizer = tokenizer
		self.sentiment_pipe = sentiment_pipe

		# Set kl controller
		if self.ppo_params['adap_kl_ctrl']:
			self.kl_ctl = AdaptiveKLController(self.ppo_params['init_kl_coef'],
												self.ppo_params['target'],
												self.ppo_params['horizon'])
		else:
			self.kl_ctl = FixedKLController(self.ppo_params['init_kl_coef'])

	def logprobs_from_logits(self, logits, labels):
		"""
		See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
		"""
		logp = F.log_softmax(logits, dim=2)
		logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
		return logpy

	def batched_forward_pass(self, queries, responses, input_ids):
		with torch.no_grad():
			logits, _, v = self.model(input_ids)
			ref_logits, _, _ = self.ref_model(input_ids.cpu()) # TODO(dahoas): Need to make decision about what to do with ref model: keep on cpu?
			ref_logits = ref_logits.to(self.model.device)
		logprobs = self.logprobs_from_logits(logits[:,:-1,:], input_ids[:,1:])
		ref_logprobs = self.logprobs_from_logits(ref_logits[:,:-1,:], input_ids[:,1:])

		start = len(queries[0])-1  # TODO(dahoas): Do different queries have different lengths
		end = len(queries[0]) + len(responses[0]) - 1
		all_values = v[:, start-1:end-1]
		all_logprobs = logprobs[:, start:end]
		all_ref_logprobs = ref_logprobs[:, start:end]
		return all_logprobs, all_ref_logprobs, all_values

	def compute_rewards(self, scores, logprobs, ref_logprobs):
		"""Compute per token rewards from scores and KL-penalty."""
		kl = logprobs - ref_logprobs
		non_score_rewards = -self.kl_ctl.value * kl
		rewards = non_score_rewards.clone()
		rewards[:, -1] += scores
		return rewards, non_score_rewards

	def ppo_loss(self, queries, responses, scores):
		input_ids = torch.stack([torch.cat([q, r]) for q, r in zip(queries, responses)])  # TODO(dahoas): may be inefficient
		old_logprobs, ref_logprobs, values = self.batched_forward_pass(queries, responses, input_ids)
		rewards, non_score_reward = self.compute_rewards(scores, old_logprobs, ref_logprobs)

		lastgaelam = 0
		advantages_reversed = []
		gen_len = self.gen_kwargs['max_length']

		for t in reversed(range(gen_len)):
			nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
			delta = rewards[:, t] + self.ppo_params['gamma'] * nextvalues - values[:, t]
			lastgaelam = delta + self.ppo_params['gamma'] * self.ppo_params['lam'] * lastgaelam
			advantages_reversed.append(lastgaelam)
		advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)

		returns = advantages + values
		advantages = whiten(advantages)
		advantages = advantages.detach()

		logits, _, vpreds = self.model(input_ids)
		logprobs = self.logprobs_from_logits(logits[:,:-1,:], input_ids[:, 1:])
		# Only the generation part of the values/logprobs is needed
		logprobs, vpreds = logprobs[:, -gen_len:], vpreds[:,-gen_len-1:-1]

		# PPO loss computation
		vpredsclipped = clip_by_value(vpreds,
									 values - self.ppo_params["cliprange_value"],
									 values + self.ppo_params["cliprange_value"])

		vf_losses1 = (vpreds - returns)**2
		vf_losses2 = (vpredsclipped - returns)**2
		vf_loss = .5 * torch.mean(torch.max(vf_losses1, vf_losses2))
		vf_clipfrac =  torch.mean(torch.gt(vf_losses2, vf_losses1).double())

		ratio = torch.exp(logprobs - old_logprobs)

		pg_losses = -advantages * ratio
		pg_losses2 = -advantages * torch.clamp(ratio,
											   1.0 - self.ppo_params['cliprange'],
											   1.0 + self.ppo_params['cliprange'])

		pg_loss = torch.mean(torch.max(pg_losses, pg_losses2))
		pg_clipfrac = torch.mean(torch.gt(pg_losses2, pg_losses).double())

		loss = pg_loss + self.ppo_params['vf_coef'] * vf_loss

		kl = old_logprobs - ref_logprobs
		mean_kl = torch.mean(torch.sum(kl, dim=1)).item()
		self.kl_ctl.update(mean_kl, input_ids.size()[0])
		return loss

	def compute_loss(self, model, inputs):
		batch_input_tokens = inputs.get('input_ids')
		with torch.no_grad():
			response_tensors = model.generate(batch_input_tokens, **self.gen_kwargs)  # I can batch generate
		responses = [self.tokenizer.decode(r.squeeze()) for r in response_tensors]  # Only scoring responses, not prefix
		pipe_outputs = self.sentiment_pipe(responses, **self.sent_kwargs)  # TODO(dahoas): set up sentiment pipe
		try:
			rewards = torch.tensor([output[1]["score"] for output in pipe_outputs]).to(self.model.device)  # TODO(dahoas): Correct device?
		except:
			print(pipe_outputs)
			exit()
		# Log rewards
		wandb.log({'mean_rewards': torch.mean(rewards).item()})
		loss = self.ppo_loss(batch_input_tokens, response_tensors, rewards)
		return loss  # Negate loss to try maximizing reward