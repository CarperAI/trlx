# Generates positive movie reviews by tuning a pretrained model on IMDB dataset
# with a sentiment reward function
import json
import os
import sys
from typing import List

import torch
from datasets import load_dataset
from datasets import Dataset
from transformers import pipeline
import pandas as pd

import trlx
from trlx.data.default_configs import TRLConfig, default_ppo_config


def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["pos"]


def main(hparams={}):
    # Merge sweep config with default config if given
    config = TRLConfig.update(default_ppo_config().to_dict(), hparams)
    config.model.model_path = 'nakcnx/OTG-Math-680'
    config.tokenizer.tokenizer_path = 'nakcnx/OTG-Math-680' #'nakcnx/TGPT-2-345M' #'nakcnx/TGPT-Neo-125M'
    config.chunk_size = 1

    if torch.cuda.is_available():
        device = int(os.environ.get("LOCAL_RANK", 0))
    else:
        device = -1

    sentiment_fn = pipeline(
        "sentiment-analysis",
        "poom-sci/WangchanBERTa-finetuned-sentiment",
        top_k=3,
        truncation=True,
        batch_size=16, #256
        device=device,
    )

    def reward_fn(samples: List[str], **kwargs) -> List[float]:
        sentiments = list(map(get_positive_score, sentiment_fn(samples)))
        print(f"sentiments: {sentiments}")
        return sentiments

    # Take few words off of movies reviews as prompts
    imdb = load_dataset("nakcnx/Thai-IMDB")
    df = pd.DataFrame( imdb['train'] )
    df = df.dropna()
    imdb2 = Dataset.from_pandas(df)
    prompts = [" ".join(str(review).split()[:4]) for review in imdb['train']["review_th"]]

    trlx.train(
        reward_fn=reward_fn,
        prompts=prompts,
        eval_prompts=["หนังเรื่องนี้ไม่สนุกเลย"] * 16, #64,
        config=config,
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)
