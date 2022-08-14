import torch
from transformers import pipeline

from framework.orchestrator import Orchestrator, register_orchestrator
from framework.pipeline.sentiment import SentimentPipeline
from framework.model import BaseRLModel
from framework.utils import chunk, flatten

@register_orchestrator
class SentimentOrchestrator(Orchestrator):
    def __init__(self, pipeline : SentimentPipeline, rl_model : BaseRLModel):
        self.pipeline = pipeline
        self.rl_model = rl_model

        pipe_device = 0 if torch.cuda.is_available() else -1
        self.sentiment_pipe= pipeline('sentiment-analysis', 'lvwerra/distilbert-imdb', device=pipe_device)

    def make_experience(self, chunk_size = 2048):
        text = self.pipeline.text
        sentiments = [self.sentiment_pipe(batch) for batch in chunk(text, chunk_size)]
        sentiments = flatten(sentiments)

        self.rl_model.push_to_store((text, sentiments))