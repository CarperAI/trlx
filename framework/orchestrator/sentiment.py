import torch
from transformers import pipeline as tfpipeline

from framework.orchestrator import Orchestrator, register_orchestrator
from framework.pipeline.sentiment import SentimentPipeline
from framework.model import BaseRLModel
from framework.utils import chunk, flatten, sentiment_score

from tqdm import tqdm

@register_orchestrator
class OfflineSentimentOrchestrator(Orchestrator):
    def __init__(self, pipeline : SentimentPipeline, rl_model : BaseRLModel):
        self.pipeline = pipeline
        self.rl_model = rl_model

        pipe_device = 0 if torch.cuda.is_available() else -1
        self.sentiment_pipe = tfpipeline('sentiment-analysis', 'lvwerra/distilbert-imdb', device=pipe_device)

    def make_experience(self, chunk_size = 512):
        text = self.pipeline.text

        # Run all text in pipeline through sentiment analysis model to get sentiment scores for entire dataset
        sentiments = [self.sentiment_pipe(batch, truncation = True, max_length = 512) for batch in tqdm(chunk(text, chunk_size))]
        sentiments = flatten(sentiments)
        sentiments = sentiment_score(sentiments)

        # Push text and sentiment (i.e. reward) to models rollout storage
        self.rl_model.push_to_store((text, sentiments))