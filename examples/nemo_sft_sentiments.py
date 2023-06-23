import os
from typing import Dict, List

from datasets import load_dataset
from transformers import pipeline

import trlx
from trlx.data.default_configs import (
    TRLConfig,
    default_nemo_1_3b_config,
    default_nemo_20b_config,
    default_sft_config,
)


def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]


def main(hparams={}):
    # Merge sweep config with default config if given

    default_config = TRLConfig.update(default_sft_config(), hparams)

    cfg_name = os.environ.get("NEMO_CONFIG", "1.3B")
    if cfg_name == "1.3B":
        nemo_config = default_nemo_1_3b_config()
    elif cfg_name == "20B":
        nemo_config = default_nemo_20b_config()
    else:
        raise ValueError(f"Unknown NEMO_CONFIG: {cfg_name}")

    nemo_config.exp_manager.create_wandb_logger = True
    nemo_config.exp_manager.wandb_logger_kwargs.name = f"nemo-sft-sentiments-{cfg_name}"

    config = default_config.evolve(
        train=dict(
            trainer="NeMoSFTTrainer",
            trainer_kwargs=dict(
                pretrained_model=f"/mnt/hdd/nemo-megatron-gpt-{cfg_name}/",
                megatron_cfg=nemo_config,
            ),
        ),
        model=dict(num_layers_unfrozen=-1),
        tags=["nemo", "sft", "sentiments", cfg_name],
    )

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
