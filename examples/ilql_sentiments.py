# Generates positive movie reviews by learning from sentiment-labeled IMDB dataset

from datasets import load_dataset
from transformers import pipeline

import os
import trlx

if __name__ == "__main__":
    sentiment_fn = pipeline(
        "sentiment-analysis",
        "lvwerra/distilbert-imdb",
        device=int(os.environ.get("LOCAL_RANK", 0)),
    )

    def metric_fn(samples):
        outputs = sentiment_fn(samples, return_all_scores=True, batch_size=256)
        sentiments = [output[1]["score"] for output in outputs]
        return {"sentiments": sentiments}

    imdb = load_dataset("imdb", split="train+test")

    trlx.train(
        dataset=(imdb["text"], imdb["label"]),
        eval_prompts=["I don't know much about Hungarian underground"] * 64,
        metric_fn=metric_fn,
    )
