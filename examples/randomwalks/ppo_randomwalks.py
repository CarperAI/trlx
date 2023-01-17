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
        reward_fn=lambda samples, **kwargs: metric_fn(samples)["optimality"],
        prompts=prompts,
        eval_prompts=prompts,
        metric_fn=lambda samples, **kwargs: metric_fn(samples),
        config=config,
    )


if __name__ == "__main__":
    main()
