# Generates positive movie reviews by tuning a pretrained model on IMDB dataset
# with a sentiment reward function
import json
import os
import sys
from typing import List

import torch
from datasets import load_dataset
from transformers import DistilBertForSequenceClassification, pipeline

import trlx
from trlx.data.default_configs import (
    TRLConfig,
    default_nemo_1_3b_config,
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
    elif cfg_name == "20B":
        nemo_config = default_nemo_20b_config()
    else:
        raise ValueError(f"Unknown NEMO_CONFIG: {cfg_name}")

    config = default_config.evolve(
        train=dict(
            seq_length=2048,
            batch_size=32,
            epochs=100,
            eval_interval=64,
            trainer="NeMoPPOTrainer",
            trainer_kwargs=dict(
                pretrained_model=f"/mnt/nvme/home/uwu/nemo-megatron-gpt-{cfg_name}/",
                megatron_cfg=nemo_config,
            ),
            checkpoint_interval=64,
            checkpoint_dir=f"nemo_{cfg_name}_ppo_math",
            seed=1910,
            project_name="trlxnemo",
            tags=["nemo", "ppo", "math", cfg_name],
        ),
        model=dict(num_layers_unfrozen=-1),
        method=dict(num_rollouts=32, gen_kwargs=dict(temperature=0.9, max_new_tokens=4), chunk_size=32, ppo_epochs=4),
    )

    # sample dataset of numbers
    # use generator to ensure same prompt dataset across ranks
    gen = torch.Generator()
    gen.manual_seed(config.train.seed)

    xs = torch.randint(0, 100000, (1000,), generator=gen)
    ys = torch.randint(0, 100000, (1000,), generator=gen)

    prompts = [f"{x} + {y} = " for x, y in zip(xs, ys)]
    print(f"{prompts[:10]}...")

    def try_err_or_min(x, y, o):
        try:
            return abs((int(x) + int(y)) - int(o))
        except:
            return 100

    def reward_fn(samples: List[str], prompts: List[str], outputs: List[str]):
        pxs = [p.split()[0] for p in prompts]
        pys = [p.split()[2] for p in prompts]
        err = [try_err_or_min(x, y, o) for x, y, o in zip(pxs, pys, outputs)]
        return [1 - e / 100 for e in err]

    trlx.train(
        reward_fn=reward_fn,
        prompts=prompts[:-128],
        eval_prompts=prompts[-128:],
        config=config,
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)
