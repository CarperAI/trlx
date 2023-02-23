from typing import Dict, List

from datasets import load_dataset
from transformers import pipeline

import trlx
from trlx.data.default_configs import default_ilql_config


def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]


default_config = default_ilql_config()


def main(hparams={}):
    config = default_config.evolve(
        train=dict(
            seq_length=1024,
            batch_size=512,
            total_steps=200,
            trainer="NeMoILQLTrainer",
            trainer_kwargs=dict(
                pretrained_model="/mnt/nvme/home/uwu/nemo-megatron-gpt-20B/",
                megatron_cfg="megatron_20b.yaml",
            ),
        )
    )
    config = config.evolve(**hparams)

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

    imdb = load_dataset("imdb", split="train+test")

    trlx.train(
        dataset=(imdb["text"], imdb["label"]),
        eval_prompts=["I don't know much about Hungarian underground"] * 128,
        metric_fn=metric_fn,
        config=config,
    )


if __name__ == "__main__":
    main()
