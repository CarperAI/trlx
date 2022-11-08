from randomwalks import generate_random_walks

import trlx
from trlx.data.configs import TRLConfig


def main():
    walks, logit_mask, metric_fn, eval_prompts = generate_random_walks(seed=1000)
    config = TRLConfig.load_yaml("configs/ppo_randomwalks.yml")

    trlx.train(
        reward_fn=lambda xs: metric_fn(xs)["rewards"],
        eval_prompts=eval_prompts * 10,
        metric_fn=metric_fn,
        config=config,
    )


if __name__ == "__main__":
    main()
