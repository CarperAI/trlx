# Generates positive movie reviews by tuning a pretrained model on IMDB dataset
# with a sentiment reward function
import json
import os
import sys
from typing import List

from datasets import load_dataset
from transformers import DistilBertForSequenceClassification, pipeline

import trlx
from trlx.data.default_configs import TRLConfig, default_ppo_config


def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]


def main(hparams={}):
    # Merge sweep config with default config if given
    default_config = TRLConfig.update(default_ppo_config().to_dict(), hparams)

    config = default_config.evolve(
        train=dict(
            seq_length=2048,
            batch_size=32,
            epochs=100,
            eval_interval=64,
            trainer="NeMoPPOTrainer",
            trainer_kwargs=dict(
                pretrained_model="/mnt/nvme/home/uwu/nemo-megatron-gpt-20B/",
                megatron_cfg="megatron_20b.yaml",
            ),
            checkpoint_interval=1,
            checkpoint_dir="nemo_ppo_sentiments",
            seed=1910,
            project_name="trlxnemo",
        ),
        model=dict(num_layers_unfrozen=-1),
        method=dict(num_rollouts=32, gen_kwargs=dict(temperature=0.9, max_new_tokens=256), chunk_size=32, ppo_epochs=4),
    )

    rank = int(os.environ["SLURM_PROCID"])
    local_rank = rank % 8

    reward_model = DistilBertForSequenceClassification.from_pretrained("lvwerra/distilbert-imdb")
    reward_model.to(local_rank)
    sentiment_fn = pipeline(
        "sentiment-analysis",
        model=reward_model,  # "lvwerra/distilbert-imdb",
        tokenizer="lvwerra/distilbert-imdb",
        top_k=2,
        truncation=True,
        batch_size=256,
        device=local_rank,
    )

    def reward_fn(samples: List[str], **kwargs) -> List[float]:
        reward_model.to(local_rank)
        sentiments = list(map(get_positive_score, sentiment_fn(samples)))
        reward_model.to("cpu")
        return sentiments

    # Take few words off of movies reviews as prompts
    imdb = load_dataset("imdb", split="train+test")
    prompts = [" ".join(review.split()[:4]) for review in imdb["text"]]

    trlx.train(
        reward_fn=reward_fn,
        prompts=prompts,
        eval_prompts=["I don't know much about Hungarian underground"] * 256,
        config=config,
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)
