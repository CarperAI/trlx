# Generates positive movie reviews by learning from sentiment-labeled IMDB dataset
import sys
from accelerate.utils import imports


def always_false():
    return False


imports.is_megatron_lm_available = always_false


sys.path.append("/fsx/home-uwu/gpt-neox/")

import megatron
from datasets import load_dataset
from transformers import pipeline

import trlx
from typing import List, Dict


def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]


def main():
    sentiment_fn = pipeline(
        "sentiment-analysis",
        "lvwerra/distilbert-imdb",
        top_k=2,
        truncation=True,
        batch_size=256,
        device=-1,
    )

    def metric_fn(samples: List[str]) -> Dict[str, List[float]]:
        sentiments = list(map(get_positive_score, sentiment_fn(samples)))
        return {"sentiments": sentiments}

    imdb = load_dataset("imdb", split="train+test")

    trlx.train(
        "gpt2",
        dataset=(imdb["text"], imdb["label"]),
        eval_prompts=["I don't know much about Hungarian underground"] * 64,
        metric_fn=metric_fn,
    )


if __name__ == "__main__":
    main()
