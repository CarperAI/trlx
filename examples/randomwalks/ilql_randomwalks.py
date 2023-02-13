import pathlib

import yaml
from transformers import GPT2Config

import trlx
from examples.randomwalks import generate_random_walks
from trlx.data.default_configs import default_ilql_config

def main():
    config = default_ilql_config()

    metric_fn, eval_prompts, walks, _ = generate_random_walks(seed=config.train.seed)
    rewards = metric_fn(walks)["optimality"]
    # split each random walk into (starting state, rest of the walk)
    walks = [[walk[:1], walk[1:]] for walk in walks]

    trlx.train(
        model_path=GPT2Config(n_layer=6, n_embd=144, vocab_size=23),
        samples=walks,
        rewards=rewards,
        eval_prompts=eval_prompts,
        metric_fn=lambda samples, **kwargs: metric_fn(samples),
        config=config,
        stop_sequences=["|"],
    )


if __name__ == "__main__":
    main()
