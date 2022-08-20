from transformers import pipeline as tfpipeline
import torch

from framework.model import BaseRLModel
from framework.utils import chunk, flatten, sentiment_score

def sentiment_eval(model : BaseRLModel) -> float:
    """
    Evaluate model by having it generate text and guaging the sentiment score

    :param model: Model with a sampling method
    :type model: BaseRLModel

    :returns: Average sentiment score (< 0 => negative > 0 => positive)
    :rtype: float
    """

    # Different chunk sizes for LM sampling, and the model predicting sentiment
    total_samples = 32
    sample_chunk_size = 8
    sentiment_chunk_size = min(512, total_samples)

    with torch.no_grad():
        # Sample from the AR model
        samples = [model.sample(length = 32, n_samples = sample_chunk_size) for _ in range(total_samples//sample_chunk_size)]
        samples = flatten(samples) # flatten and chunk again for the sentiment analysis model
        samples_chunks = chunk(samples, sentiment_chunk_size) 

        # compute sentiments for every sample
        pipe_device = 0 if torch.cuda.is_available() else -1
        pipe = tfpipeline('sentiment-analysis', 'lvwerra/distilbert-imdb', device=pipe_device)
        sentiments = [pipe(chunk) for chunk in samples_chunks]
        sentiments = flatten(sentiments)
        sentiments = sentiment_score(sentiments)
        
        # return average sentiment across all samples
        return sentiments.mean().item()
        





