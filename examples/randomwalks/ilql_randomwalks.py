import os

import yaml
from transformers import GPT2Config

import trlx
from examples.randomwalks import generate_random_walks
from trlx.data.configs import TRLConfig

config_path = os.path.join(os.path.dirname(__file__), "configs/ilql_randomwalks.yml")
default_config = yaml.safe_load(open(config_path))


def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)

    metric_fn, eval_prompts, walks, _ = generate_random_walks(seed=config.train.seed)
    rewards = metric_fn(walks)["optimality"]
    # split each random walk into (starting state, rest of the walk)
    walks = [[walk[:1], walk[1:]] for walk in walks]

    trlx.train(
        GPT2Config(n_layer=6, n_embd=144, vocab_size=23),
        dataset=(walks, rewards),
        eval_prompts=eval_prompts,
        metric_fn=lambda samples, **kwargs: metric_fn(samples),
        config=config,
    )


if __name__ == "__main__":
    main()
