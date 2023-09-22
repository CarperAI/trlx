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


def load_nemo_config():
    """Load nemo-megatron-1.3b model and trainer config"""
    # Import here to not require nemo as a dependency
    from omegaconf import OmegaConf

    return OmegaConf.load("nemo_llama2_7b/megatron_7b.yaml")


def main(hparams={}):
    # Merge sweep config with default config if given
    default_config = TRLConfig.update(default_ppo_config().to_dict(), hparams)
    nemo_config = load_nemo_config()
    print(nemo_config)
    cfg_name = "llama2-7b"
    config = default_config.evolve(
        train=dict(
            total_steps=1600,
            seq_length=256,
            batch_size=16,
            epochs=100,
            eval_interval=100,
            trainer="NeMoPPOTrainer",
            trainer_kwargs=dict(
                pretrained_model="nemo_llama2_7b/",
                megatron_cfg=nemo_config,
            ),
            checkpoint_interval=256,
            checkpoint_dir=f"nemo_{cfg_name}_ppo_sentiments",
            seed=2023,
            project_name="trlxnemo",
            tags=["nemo", "ppo", "sentiments", cfg_name],
        ),
        optimizer=dict(
            name="adamw",
            kwargs=dict(
                lr=1e-5,
                weight_decay=1e-06,
                eps=1.0e-8,
                betas=(0.9, 0.95),
            ),
        ),
        scheduler=dict(
            name="CosineAnnealing",
        ),
        model=dict(num_layers_unfrozen=24),
        method=dict(
            num_rollouts=128,
            init_kl_coef=0.05,
            vf_coef=1,
            scale_reward="ignored",
            gamma=1,
            lam=0.95,
            cliprange=0.2,
            cliprange_value=0.2,
            gen_kwargs=dict(temperature=1.0, max_new_tokens=64),
            chunk_size=64,
            ppo_epochs=4,
        ),
    )
    config.scheduler.kwargs = dict(warmup_steps=0, constant_steps=1e12, min_lr=1e-6)

    rank = int(os.environ["SLURM_PROCID"])
    local_rank = rank % 8

    reward_model = DistilBertForSequenceClassification.from_pretrained("lvwerra/distilbert-imdb")
    reward_model.to("cpu")
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
