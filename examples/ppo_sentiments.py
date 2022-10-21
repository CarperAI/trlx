# Generates positive movie reviews by tuning a pretrained on IMDB model
# with a sentiment reward function

from datasets import load_dataset
from transformers import pipeline

import trlx

if __name__ == "__main__":
    sentiment_fn = pipeline("sentiment-analysis", "lvwerra/distilbert-imdb")

    def reward_fn(samples):
        outputs = sentiment_fn(samples, return_all_scores=True)
        sentiments = [output[1]["score"] for output in outputs]
        return sentiments

    # Take few words off of movies reviews as prompts
    imdb = load_dataset("imdb", split="train+test")
    prompts = [" ".join(review.split()[:4]) for review in imdb["text"]]

    model = trlx.train(
        "lvwerra/gpt2-imdb",
        reward_fn=reward_fn,
        prompts=prompts,
        eval_prompts=["I don't know much about Hungarian underground"] * 64,
    )
