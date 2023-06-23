# Generates positive movie reviews by tuning a pretrained model on IMDB dataset
# with a sentiment reward function
import json
import os
import sys
from typing import List

from datasets import load_dataset
from transformers import DistilBertForSequenceClassification, pipeline

import trlx
from trlx.data.default_configs import (
    TRLConfig,
    default_nemo_1_3b_config,
    default_nemo_2b_config,
    default_nemo_20b_config,
    default_ppo_config,
)


def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]


def main(hparams={}):
    # Merge sweep config with default config if given
    default_config = TRLConfig.update(default_ppo_config().to_dict(), hparams)

    cfg_name = os.environ.get("NEMO_CONFIG", "1.3B")
    if cfg_name == "1.3B":
        nemo_config = default_nemo_1_3b_config()
    elif cfg_name == "2B":
        nemo_config = default_nemo_2b_config()
    elif cfg_name == "20B":
        nemo_config = default_nemo_20b_config()
    else:
        raise ValueError(f"Unknown NEMO_CONFIG: {cfg_name}")

    config = default_config.evolve(
        train=dict(
            total_steps=512,
            seq_length=2048,
            batch_size=32,
            epochs=100,
            eval_interval=64,
            trainer="NeMoPPOTrainer",
            trainer_kwargs=dict(
                pretrained_model=f"/mnt/hdd/nemo-megatron-gpt-{cfg_name}/",
                megatron_cfg=nemo_config,
            ),
            checkpoint_interval=256,
            checkpoint_dir=f"nemo_{cfg_name}_ppo_sentiments",
            seed=2023,
            project_name="trlxnemo",
            tags=["nemo", "ppo", "sentiments", cfg_name],
        ),
        optimizer=dict(
            name="distributed_fused_adam",
            kwargs=dict(
                lr=6.001e-5,
                weight_decay=1e-06,
                eps=1.0e-8,
                betas=(0.9, 0.95),
            ),
        ),
        scheduler=dict(
            name="CosineAnnealing",
        ),
        model=dict(num_layers_unfrozen=2),
        method=dict(
            num_rollouts=128,
            init_kl_coef=0.05,
            scale_reward="ref",
            vf_coef=1,
            gen_kwargs=dict(temperature=1.0, max_new_tokens=40),
            chunk_size=128,
            ppo_epochs=4,
        ),
    )
    config.scheduler.kwargs = dict(warmup_steps=0, constant_steps=1e12, min_lr=6.0e-5)

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
