# Generates positive movie reviews by tuning a pretrained model on IMDB dataset
# with a sentiment reward function
import json
import os
import sys
from typing import List

import torch
from datasets import load_dataset
from transformers import pipeline

import trlx
from trlx.data.default_configs import TRLConfig, default_ppo_config


def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]


def get_negative_score(scores):
    return dict(map(lambda x: tuple(x.values()), scores))["NEGATIVE"]


def main(hparams={}):
    # Merge sweep config with default config if given
    config = TRLConfig.update(default_ppo_config().to_dict(), hparams)

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

    def dense_reward_fn(samples: List[str], prompts: List[str], outputs: List[str], tokenizer, **kwargs) -> List[float]:
        # Reward positively for initially negative then positive review
        # Reward functions should never receive padded text except for a single EOS at the end
        # Reward function should return token rewards for just the response
        first_halves = [".".join(sample.split(".")[: len(sample.split(".")) // 2]) for sample in samples]
        negative_first_halves = list(map(get_negative_score, sentiment_fn(first_halves)))
        second_halves = [".".join(sample.split(".")[len(sample.split(".")) // 2 :]) for sample in samples]
        positive_second_halves = list(map(get_positive_score, sentiment_fn(second_halves)))
        text_scores = [[f, s] for f, s in zip(negative_first_halves, positive_second_halves)]
        tok_scores = []
        for sample, prompt, response, text_score in zip(samples, prompts, outputs, text_scores):
            toks = tokenizer(response).input_ids
            tok_score = [0] * len(toks)
            tok_score[len(tok_score) // 2] = text_score[0]
            tok_score[-1] = text_score[1]
            tok_scores.append(tok_score)
        return tok_scores

    # Take few words off of movies reviews as prompts
    imdb = load_dataset("imdb", split="train+test")
    prompts = [" ".join(review.split()[:4]) for review in imdb["text"]]

    trlx.train(
        reward_fn=dense_reward_fn,
        prompts=prompts,
        eval_prompts=["I don't know much about Hungarian underground"] * 256,
        config=config,
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)
