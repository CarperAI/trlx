from pathlib import Path
from typing import Dict, List

from datasets import load_dataset
from omegaconf import OmegaConf
from transformers import pipeline

import trlx
from trlx.data.default_configs import default_ilql_config


def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]


default_config = default_ilql_config()


def main(hparams={}):
    # load 20b config relative to this file
    nemo_config = OmegaConf.load(Path(__file__).parent.parent / "configs" / "nemo_configs" / "megatron_20b.yaml")
    model_config = OmegaConf.load("/mnt/hdd/nemo_gpt40B/nemo_gpt40B_tp8_pp2/model_config.yaml")
    model_config.sequence_parallel = True
    model_config.optim.name = "distributed_fused_adam"
    model_config.encoder_seq_length = 1024
    model_config.data.seq_length = 1024
    model_config.global_batch_size = 128
    nemo_config.model = model_config

    config = default_config.evolve(
        train=dict(
            seq_length=1024,
            batch_size=128,
            total_steps=200,
            trainer="NeMoILQLTrainer",
            trainer_kwargs=dict(
                pretrained_model="/mnt/hdd/nemo_gpt40B/nemo_gpt40B_tp8_pp2/",
                megatron_cfg=nemo_config,
            ),
        ),
        method=dict(
            gen_kwargs=dict(
                beta=2.0,
                temperature=0.9,
            )
        ),
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
        samples=imdb["text"],
        rewards=imdb["label"],
        eval_prompts=["I don't know much about Hungarian underground"] * 128,
        metric_fn=metric_fn,
        config=config,
    )


if __name__ == "__main__":
    main()
