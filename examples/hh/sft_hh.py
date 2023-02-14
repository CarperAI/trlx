import os
import re

import yaml
from datasets import load_dataset
from ppo_hh import create_reward_fn

import trlx
from trlx.data.configs import TRLConfig


def split_dialog(dialog):
    dialog = re.split(r"(\n\nHuman:|\n\nAssistant:)", dialog)[1:]
    return ["".join(dialog[:-1]), dialog[-1]]


def main(hparams={}):
    config_path = os.path.join(os.path.dirname(__file__), os.environ.get("CONFIG_PATH", "configs/sft_hh.yml"))
    default_config = yaml.safe_load(open(config_path))
    config = TRLConfig.update(default_config, hparams)

    dataset = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base")
    reward_fn = create_reward_fn()

    trlx.train(
        samples=dataset["train"]["chosen"],
        config=config,
        eval_prompts=[split_dialog(sample)[0] for sample in dataset["test"]["chosen"][:280]],
        metric_fn=lambda **kwargs: {"reward": reward_fn(**kwargs)},
        stop_sequences=["Human:", "human:", "Assistant:", "assistant:"],
    )


if __name__ == "__main__":
    import json
    import sys

    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)
