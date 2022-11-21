# Generates positive movie reviews by tuning a pretrained on IMDB model
# with a sentiment reward function

from datasets import load_dataset
from transformers import pipeline
from trlx.data.configs import TRLConfig

import trlx

if __name__ == "__main__":
    config = TRLConfig.load_yaml("configs/ppo_config.yml")

    sentiment_fn = pipeline("sentiment-analysis", "lvwerra/distilbert-imdb", device=-1)

    def reward_fn(samples):
        outputs = sentiment_fn(samples, return_all_scores=True, batch_size=config.method.chunk_size)
        sentiments = [output[1]["score"] for output in outputs]
        return sentiments

    # Take few words off of movies reviews as prompts
    imdb = load_dataset("imdb", split="train+test")
    prompts = [" ".join(review.split()[:4]) for review in imdb["text"]]

    model = trlx.train(
        reward_fn=reward_fn,
        config=config,
        prompts=prompts,
        eval_prompts=["I don't know much about Hungarian underground"] * 64,
    )
