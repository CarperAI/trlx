from randomwalks import generate_random_walks

import yaml
import trlx
from trlx.data.configs import TRLConfig

default_config = yaml.safe_load(open("configs/ppo_randomwalks.yml"))


def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)

    metric_fn, eval_prompts, *_ = generate_random_walks(seed=config.train.seed)

    trlx.train(
        "randomwalks/1M",
        reward_fn=lambda walks: metric_fn(walks)["rewards"],
        prompts=eval_prompts,
        eval_prompts=eval_prompts,
        metric_fn=metric_fn,
        config=config,
    )


if __name__ == "__main__":
    main()
