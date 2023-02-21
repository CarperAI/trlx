import os

import yaml
from datasets import load_dataset
from ppo_hh import create_reward_fn

import trlx
from trlx.data.configs import TRLConfig


def preprocess(sample):
    sample["prompt_output"] = [
        [sample["prompt"], sample["chosen"]],
        [sample["prompt"], sample["rejected"]],
    ]
    sample["reward"] = [1, -1]
    return sample


def main(hparams={}):
    config_path = os.path.join(os.path.dirname(__file__), os.environ.get("CONFIG_PATH", "configs/ilql_hh.yml"))
    default_config = yaml.safe_load(open(config_path))
    config = TRLConfig.update(default_config, hparams)

    dataset = load_dataset("Dahoas/full-hh-rlhf").map(preprocess)
    prompts_outputs = sum(dataset["train"]["prompt_output"], [])

    rewards = sum(dataset["train"]["reward"], [])
    eval_prompts = [prompt_output[0][0] for prompt_output in dataset["test"]["prompt_output"]][:280]
    reward_fn = create_reward_fn()

    trlx.train(
        samples=prompts_outputs,
        rewards=rewards,
        config=config,
        eval_prompts=eval_prompts,
        metric_fn=lambda **kwargs: {"reward": reward_fn(**kwargs)},
        stop_sequences=["Human:", "human:", "Assistant:", "assistant:"],
    )


if __name__ == "__main__":
    import json
    import sys

    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)
