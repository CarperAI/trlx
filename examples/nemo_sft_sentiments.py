import os
import pathlib
from typing import Dict, List

import yaml
from datasets import load_dataset
from transformers import pipeline
from trlx.trainer.nemo_sft_trainer import NeMoSFTTrainer

import trlx
from trlx.data.configs import TRLConfig

config_path = pathlib.Path(__file__).parent.joinpath("../configs/nemo_sft_config.yml")
with config_path.open() as f:
    default_config = yaml.safe_load(f)


def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]


def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)

    imdb = load_dataset("imdb", split="train+test")
    # Finetune on only positive reviews
    imdb = imdb.filter(lambda sample: sample["label"] == 1)

    sentiment_fn = pipeline(
        "sentiment-analysis",
        "lvwerra/distilbert-imdb",
        top_k=2,
        truncation=True,
        batch_size=256,
        device=-1,
    )

    def metric_fn(samples: List[str], **kwargs) -> Dict[str, List[float]]:
        sentiments = list(map(get_positive_score, sentiment_fn(samples)))
        return {"sentiments": sentiments}

    trlx.train(
        samples=imdb["text"],
        eval_prompts=["I don't know much about Hungarian underground"] * 64,
        metric_fn=metric_fn,
        config=config,
    )


if __name__ == "__main__":
    main()
