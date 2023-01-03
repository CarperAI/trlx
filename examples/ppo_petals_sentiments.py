# Generates positive movie reviews by tuning a pretrained model on IMDB dataset
# with a sentiment reward function

import os
from typing import List

import torch
import yaml
from datasets import load_dataset
from petals.client import DistributedBloomForCausalLM
from transformers import pipeline

import trlx
from trlx.data.configs import TRLConfig


def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]


default_config = yaml.safe_load(open("configs/ppo_petals_config.yml"))


def model_provider(
    config: str, pre_seq_len: int = 16, tuning_mode: str = "shallow_ptune"
):
    model = DistributedBloomForCausalLM.from_pretrained(
        config,
        pre_seq_len=pre_seq_len,
        tuning_mode=tuning_mode,
    )
    return model


def ref_model_provider(config: str):
    return DistributedBloomForCausalLM.from_pretrained(config)


def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)

    if torch.cuda.is_available():
        device = int(os.environ.get("LOCAL_RANK", 0))
    else:
        device = -1

    sentiment_fn = pipeline(
        "sentiment-analysis",
        "lvwerra/distilbert-imdb",
        top_k=2,
        truncation=True,
        batch_size=256,
        device=device,
    )

    def reward_fn(samples: List[str]) -> List[float]:
        sentiments = list(map(get_positive_score, sentiment_fn(samples)))
        return sentiments

    # Take few words off of movies reviews as prompts
    imdb = load_dataset("imdb", split="train+test")
    prompts = [" ".join(review.split()[:4]) for review in imdb["text"]]

    return trlx.train(
        reward_fn=reward_fn,
        prompts=prompts,
        eval_prompts=["I don't know much about Hungarian underground"] * 64,
        config=config,
        base_model_provider=model_provider,
        base_model_transformer_args=["input_ids"],
        ref_model_provider=ref_model_provider,
    )


if __name__ == "__main__":
    main()
