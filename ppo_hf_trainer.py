from transformers import Trainer
import torch

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
	def init_ppo_params(self, ppo_params, sent_kwargs, gen_kwargs, ref_model):
		self.ppo_params = ppo_params
		self.ref_model = ref_model
		self.sent_kwargs = sent_kwargs
		self.gen_kwargs = gen_kwargs

		# Set kl controller
		if self.ppo_params['adap_kl_ctrl']:
			self.kl_ctl = AdaptiveKLController(self.ppo_params['init_kl_coef'],
												self.ppo_params['target'],
												self.ppo_params['horizon'])
		else:
			self.kl_ctl = FixedKLController(self.ppo_params['init_kl_coef'])

	def batched_forward_pass(self, queries, responses):
		input_ids = self.data_collator([torch.cat([q, r]) for q, r in zip(queries, responses)])["input_ids"]
		with torch.no_grad():
			logits, _, v = self.model(input_ids)
			ref_logits, _, _ = self.ref_model(input_ids.cpu()) # TODO(dahoas): Need to make decision about what to do with ref model: keep on cpu?
			ref_logits = ref_logits.to(self.device)
		logprobs = logprobs_from_logits(logits[:,:-1,:], input_ids[:,1:])
		ref_logprobs = logprobs_from_logits(ref_logits[:,:-1,:], input_ids[:,1:])

		start = len(queries[0])-1  # TODO(dahoas): Do different queries have different lengths
		end = len(queries[0]) + len(responses[0]) - 1
		all_values = v[:, start-1:end-1]
		all_logprobs = logprobs[:, start:end]
		all_ref_logprobs = all_ref_logprobs[:, start:end]
		return all_logprobs, all_ref_logprobs, all_values

	def compute_rewards(self, scores, logprobs, ref_logprobs):
		"""Compute per token rewards from scores and KL-penalty."""
		kl = logprob - ref_logprob
		non_score_reward = -self.kl_ctl.value * kl
		reward = non_score_reward.clone()
		reward[:, -1] += score
		return rewards, non_score_rewards

	def logprobs_from_logits(self, logits, labels):
		"""
		See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
		"""
		logp = F.log_softmax(logits, dim=2)
		logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
		return logpy

	def ppo_loss(self, query_tensors, response_tensors, rewards):
		logprobs, ref_logprobs, values = self.batched_forward_pass(queries, responses)
		rewards, non_score_reward = self.compute_rewards(scores, logprobs, ref_logprobs)

		logits, _, vpred = self.model(model_input)
		logprob = logprobs_from_logits(logits[:,:-1,:], model_input[:, 1:])
		# Only the generation part of the values/logprobs is needed
		logprob, vpred = logprob[:, -gen_len:], vpred[:,-gen_len-1:-1]

		# PPO loss computation
		vpredclipped = clip_by_value(vpred,
									 values - self.ppo_params["cliprange_value"],
									 values + self.ppo_params["cliprange_value"])

		vf_losses1 = (vpred - returns)**2
		vf_losses2 = (vpredclipped - returns)**2
		vf_loss = .5 * torch.mean(torch.max(vf_losses1, vf_losses2))
		vf_clipfrac =  torch.mean(torch.gt(vf_losses2, vf_losses1).double())

		ratio = torch.exp(logprob - old_logprobs)

		pg_losses = -advantages * ratio
		pg_losses2 = -advantages * torch.clamp(ratio,
											   1.0 - self.ppo_params['cliprange'],
											   1.0 + self.ppo_params['cliprange'])

		pg_loss = torch.mean(torch.max(pg_losses, pg_losses2))
		pg_clipfrac = torch.mean(torch.gt(pg_losses2, pg_losses).double())

		loss = pg_loss + self.ppo_params['vf_coef'] * vf_loss
		return loss

	def compute_loss(self, model, inputs):
		batch_input_tokens = inputs.get('input_ids')
		print(batch_input_tokens)
		response_tensors = model.generate(batch_input_tokens, **self.gen_kwargs)  # I can batch generate
		print(response_tensors)
		responses = [self.tokenizer.decode(r.squeeze()) for r in response_tensors]
		texts = [q + r for q,r in zip(input_queries, responses)]
		pipe_outputs = sentiment_pipe(texts, **self.sent_kwargs)  # TODO(dahoas): set up sentiment pipe
		rewards = torch.tensor([output[1]["score"] for output in pipe_outputs]).to(self.device)  # TODO(dahoas): Correct device?

		loss = self.ppo_loss(query_tensors, response_tensors, rewards)
		return loss