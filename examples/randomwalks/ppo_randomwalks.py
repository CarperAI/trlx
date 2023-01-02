import os

import yaml

import trlx
from examples.randomwalks import generate_random_walks
from trlx.data.configs import TRLConfig

config_path = os.path.join(os.path.dirname(__file__), "configs/ppo_randomwalks.yml")
default_config = yaml.safe_load(open(config_path))


def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)

    metric_fn, prompts, *_ = generate_random_walks(seed=config.train.seed)

    trlx.train(
        "CarperAI/randomwalks",
        reward_fn=lambda walks: metric_fn(walks)["optimality"],
        prompts=prompts,
        eval_prompts=prompts,
        metric_fn=metric_fn,
        config=config,
    )


if __name__ == "__main__":
    main()
