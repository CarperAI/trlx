import torch
from transformers import pipeline as tfpipeline
from framework.data.accelerate_base_datatypes import PromptBatch
from framework.data.ppo_types import PPORLElement

from framework.orchestrator import Orchestrator, register_orchestrator
from framework.pipeline.ppo_pipeline import PPOPipeline
from framework.pipeline.sentiment import SentimentPipeline
from framework.model import BaseRLModel
from framework.utils import chunk, flatten, sentiment_score

from tqdm import tqdm

from framework.utils import Clock

from framework.utils.modeling import logprobs_from_logits
import wandb

@register_orchestrator
class PPOOrchestrator(Orchestrator):
	def __init__(self, pipeline : PPOPipeline, rl_model : BaseRLModel, chunk_size : int = 512):
		self.pipeline = pipeline
		self.rl_model = rl_model
		self.chunk_size = chunk_size

		self.pipeline_loader = self.pipeline.create_loader(self.chunk_size, shuffle = True, num_workers = 2)
		self.pipeline_loader = self.rl_model.accelerator.prepare(self.pipeline_loader)
		self.pipeline_iterator = iter(self.pipeline_loader)

		self.ref_model = self.rl_model.get_arch(self.rl_model.config)

		self.rl_model.orch = self


	def score(self, texts):
		"""
		Batched scoring function taking text and generating scalar
		"""
		pass


	def make_experience(self, num_rollouts : int = 1024, iter_count : int = 0):

		ppo_rl_elements = []
		stats = {}
		clock = Clock()
		for i in tqdm(range(num_rollouts // self.chunk_size)):
			# Get next batch in prompt dataset and refresh if exhausted
			try :
				batch : PromptBatch = next(self.pipeline_iterator)
			except StopIteration:
				self.pipeline_iterator = iter(self.pipeline_loader)
				batch = next(self.pipeline_iterator)

			query_tensors, response_tensors, response_text = self.rl_model.act(batch)
			texts = [q + r for q,r in zip(batch.text, response_text)]
			scores = self.score(texts)

			# Precompute logprobs, values
			all_tokens = torch.cat((query_tensors, response_tensors), dim=1)
			assert all_tokens.size()[1] == query_tensors.size()[1] + response_tensors.size()[1]
			with torch.no_grad():
				logits, _, v = self.rl_model.model(all_tokens)
				ref_logits, _, _ = self.ref_model(all_tokens.cpu()) # TODO(dahoas): Need to make decision about what to do with ref model: keep on cpu?
				ref_logits = ref_logits.to(self.rl_model.accelerator.device)
			logprobs = logprobs_from_logits(logits[:,:-1,:], all_tokens[:,1:])
			ref_logprobs = logprobs_from_logits(ref_logits[:,:-1,:], all_tokens[:,1:])
			start = query_tensors.size()[1]-1
			end = query_tensors.size()[1] + response_tensors.size()[1] - 1
			all_values = v[:, start-1 : end-1]
			all_logprobs = logprobs[:, start : end]
			all_ref_logprobs = ref_logprobs[:, start : end]

			# Compute rewards
			kls = all_logprobs - all_ref_logprobs
			non_score_rewards = -self.rl_model.config.method.init_kl_coef * kls
			all_rewards = non_score_rewards.clone()
			all_rewards[:, -1] += scores.to(self.rl_model.accelerator.device)

			query_tensors = query_tensors.cpu()
			response_tensors = response_tensors.cpu()
			all_logprobs = all_logprobs.cpu()
			all_values = all_values.cpu()
			all_rewards = all_rewards.cpu()

			exp_time = clock.tick()

			# Evaluate model on first chunk
			if i == 0:
				mean_score = torch.mean(scores).item()
				rows = list(zip(texts, scores.tolist()))
				stats = {"exp_time": exp_time, "mean_score": mean_score, 'responses': wandb.Table(columns=['response', 'score'], rows=rows[:16])}
				self.rl_model.accelerator.log(stats, step=iter_count)


			new_ppo_rl_elements = [PPORLElement(
										query_tensor=query_tensors[i, :],
										response_tensor=response_tensors[i, :],
										logprobs=all_logprobs[i, :],
										values=all_values[i, :],
										rewards=all_rewards[i, :],
									) for i in range(query_tensors.size()[0])]
			ppo_rl_elements += new_ppo_rl_elements

		# Push text and sentiment (i.e. reward) to models rollout storage
		self.rl_model.push_to_store(ppo_rl_elements)
		print(len(self.rl_model.store))
