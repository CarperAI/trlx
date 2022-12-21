import sys
from typing import List

import torch
from transformers import pipeline
def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]
import os
import trlx
from trlx.data.configs import TRLConfig

if __name__ == "__main__":
    
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
    from datasets import load_dataset

    imdb = load_dataset("imdb", split="train+test")
    prompts = ["Complete this film review: " + " ".join(review.split()[:4]) for review in imdb["text"]]


    # prompts = val_openai_summ
    # print(len(prompts))
    config = TRLConfig.load_yaml("ppo_config_sent_t5.yml")
    model = trlx.train(
        config.model.model_path,
        reward_fn=reward_fn,
        prompts=prompts,
        eval_prompts=prompts[0:100],
        config=config
    )
