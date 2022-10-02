import torch
from transformers import pipeline as tfpipeline

from framework.orchestrator import Orchestrator, register_orchestrator
from framework.pipeline.sentiment import SentimentPipeline
from framework.model import BaseRLModel
from framework.utils import chunk, flatten, sentiment_score

@register_orchestrator
class OfflineSentimentOrchestrator(Orchestrator):
    def __init__(self, model: BaseRLModel, train_pipeline, eval_pipeline, reward_fn):
        self.model = model

        rewards = torch.as_tensor(reward_fn(train_pipeline.texts)).float()
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-30)
        self.model.train_store.push((train_pipeline.texts, rewards))

        self.model.eval_pipeline = eval_pipeline
        self.model.reward_fn = reward_fn

    def score(samples):
        return self.model.reward_fn(samples)
