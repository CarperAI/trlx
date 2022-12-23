from examples.randomwalks import generate_random_walks

import os
import trlx
from trlx.data.configs import TRLConfig
import yaml
from transformers import GPT2Config

config_path = os.path.join(os.path.dirname(__file__), "configs/ilql_randomwalks.yml")
default_config = yaml.safe_load(open(config_path))


def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)

    metric_fn, eval_prompts, walks, _ = generate_random_walks(seed=config.train.seed)
    rewards = metric_fn(walks)["optimality"]
    # split each random walk into (starting state, rest of the walk)
    walks = [[walk[:1], walk[1:]] for w in walks]

    trlx.train(
        GPT2Config(n_layer=6, n_embd=144, vocab_size=23),
        dataset=(walks, rewards),
        eval_prompts=eval_prompts,
        metric_fn=metric_fn,
        config=config,
    )


if __name__ == "__main__":
    main()
