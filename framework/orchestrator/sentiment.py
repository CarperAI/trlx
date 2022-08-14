import torch
from transformers import pipeline as PIPE

from framework.orchestrator import Orchestrator, register_orchestrator
from framework.pipeline.sentiment import SentimentPipeline
from framework.model import BaseRLModel
from framework.utils import chunk, flatten

from tqdm import tqdm

@register_orchestrator
class SentimentOrchestrator(Orchestrator):
    def __init__(self, pipeline : SentimentPipeline, rl_model : BaseRLModel):
        self.pipeline = pipeline
        self.rl_model = rl_model

        pipe_device = 0 if torch.cuda.is_available() else -1
        self.sentiment_pipe = PIPE('sentiment-analysis', 'lvwerra/distilbert-imdb', device=pipe_device)

    def make_experience(self, chunk_size = 32):
        text = self.pipeline.text[:32*4]
        sentiments = [self.sentiment_pipe(batch, truncation = True, max_length = 512) for batch in tqdm(chunk(text, chunk_size))]
        sentiments = flatten(sentiments)
        sentiments = torch.tensor([-s['score'] if s['label'] == "NEGATIVE" else s['score'] for s in sentiments])

        self.rl_model.push_to_store((text, sentiments))