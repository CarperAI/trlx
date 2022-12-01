from datasets import load_dataset
from transformers import pipeline

import trlx
import yaml
from typing import List, Dict
import os
from trlx.data.configs import TRLConfig


def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]


import os.path

default_config = yaml.safe_load(
    open(os.path.dirname(__file__) + "/../configs/ilql_config.yml")
)


def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)

    sentiment_fn = pipeline(
        "sentiment-analysis",
        "lvwerra/distilbert-imdb",
        top_k=2,
        truncation=True,
        batch_size=256,
        device=0 if int(os.environ.get("LOCAL_RANK", 0)) == 0 else -1,
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
        config=config,
    )


if __name__ == "__main__":
    main()
