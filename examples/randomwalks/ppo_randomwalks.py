import pathlib

import yaml

import trlx
from examples.randomwalks import generate_random_walks
from trlx.data.configs import TRLConfig

config_path = pathlib.Path(__file__).parent.joinpath("configs/ppo_randomwalks.yml")
with config_path.open() as f:
    default_config = yaml.safe_load(f)


def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)

    metric_fn, prompts, *_ = generate_random_walks(seed=config.train.seed)

    trlx.train(
        "CarperAI/randomwalks",
        # An "optimality" reward function is used, with scores in [0,1]
        # depending on how close the path is to the shortest possible path.
        reward_fn=lambda samples, prompts, outputs: metric_fn(samples)["optimality"],
        # The prompts are simply the first nodes (represented as letters) to
        # start from.
        prompts=prompts,
        eval_prompts=prompts,
        metric_fn=lambda samples, prompts, outputs: metric_fn(samples),
        config=config,
    )


if __name__ == "__main__":
    main()
