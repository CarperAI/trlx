import pathlib

import yaml

import trlx
from examples.randomwalks import generate_random_walks
from trlx.data.configs import TRLConfig
from trlx.data.default_configs import default_ppo_config

def main(hparams={}):
    config = default_ppo_config()
    metric_fn, prompts, *_ = generate_random_walks(seed=config.train.seed)

    trlx.train(
        # An "optimality" reward function is used, with scores in [0,1]
        # depending on how close the path is to the shortest possible path.
        reward_fn=lambda samples, prompts, outputs: metric_fn(samples)["optimality"],
        # The prompts are simply the first nodes (represented as letters) to
        # start from.
        prompts=prompts,
        eval_prompts=prompts,
        metric_fn=lambda samples, prompts, outputs: metric_fn(samples),
        config=config
    )


if __name__ == "__main__":
    main()
