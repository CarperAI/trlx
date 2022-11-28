from .randomwalks import generate_random_walks

from transformers import GPT2Config
import trlx
from trlx.data.configs import TRLConfig
import yaml

default_config = yaml.safe_load(open("configs/ilql_randomwalks.yml"))


def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)

    metric_fn, eval_prompts, walks, _ = generate_random_walks(seed=config.train.seed)
    lengths = metric_fn(walks)["lengths"]

    trlx.train(
        GPT2Config(n_layer=6, n_embd=144, vocab_size=23),
        dataset=(walks, lengths),
        eval_prompts=eval_prompts,
        metric_fn=metric_fn,
        config=config,
    )


if __name__ == "__main__":
    main()
