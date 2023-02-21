import os

import yaml
from datasets import load_dataset
from ppo_hh import create_reward_fn

import trlx
from trlx.data.configs import TRLConfig


def preprocess(sample):
    sample["chosen_sample"] = sample["prompt"] + sample["chosen"]
    return sample


def main(hparams={}):
    config_path = os.path.join(os.path.dirname(__file__), os.environ.get("CONFIG_PATH", "configs/sft_hh.yml"))
    default_config = yaml.safe_load(open(config_path))
    config = TRLConfig.update(default_config, hparams)

    dataset = load_dataset("Dahoas/full-hh-rlhf").map(preprocess)
    reward_fn = create_reward_fn()

    trlx.train(
        config=config,
        samples=dataset["train"]["chosen_sample"],
        eval_prompts=dataset["test"]["prompt"][:280],
        metric_fn=lambda **kwargs: {"reward": reward_fn(**kwargs)},
        stop_sequences=["Human:", "human:", "Assistant:", "assistant:"],
    )


if __name__ == "__main__":
    import json
    import sys

    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)
