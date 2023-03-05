from typing import Dict, List

from datasets import load_dataset
from transformers import pipeline

import trlx
from trlx.data.default_configs import default_sft_config


def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]


default_config = default_sft_config()


def main(hparams={}):
    # Merge sweep config with default config if given

    config = default_config.evolve(
        train=dict(
            trainer="NeMoSFTTrainer",
            trainer_kwargs=dict(
                pretrained_model=None,
                megatron_cfg="megatron_20b.yaml",
            ),
        )
    )
    config = config.evolve(**hparams)

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
