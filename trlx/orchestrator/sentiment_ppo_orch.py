import torch
from transformers import pipeline as tfpipeline
from trlx.data.accelerate_base_datatypes import PromptBatch
from trlx.data.ppo_types import PPORLElement

from trlx.orchestrator import Orchestrator, register_orchestrator
from trlx.orchestrator.ppo_orchestrator import PPOOrchestrator
from trlx.model import BaseRLModel
from trlx.utils import chunk, flatten, sentiment_score
from trlx.pipeline.ppo_pipeline import PPOPipeline

from tqdm import tqdm

from trlx.utils.modeling import logprobs_from_logits
from transformers import pipeline as sentiment_pipeline


@register_orchestrator
class PPOSentimentOrchestrator(PPOOrchestrator):
	def __init__(self, pipeline : PPOPipeline, rl_model : BaseRLModel, chunk_size = 512):
		super().__init__(pipeline, rl_model, chunk_size)
		self.sentiment_pipe = sentiment_pipeline("sentiment-analysis","lvwerra/distilbert-imdb", device=-1)

	def score(self, texts):
		"""
		Batched scoring function taking text and generating scalar
		"""
		sent_kwargs = {
				"return_all_scores": True,
				"function_to_apply": None,
				"batch_size": self.chunk_size,
			}
		pipe_outputs = self.sentiment_pipe(texts, **sent_kwargs)
		scores = torch.tensor([output[1]["score"] for output in pipe_outputs])
		return scores