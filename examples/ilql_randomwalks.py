from randomwalks import generate_random_walks

import trlx
from trlx.data.configs import TRLConfig
from transformers import GPT2Config

if __name__ == "__main__":
    walks, logit_mask, metric_fn, eval_prompts = generate_random_walks(seed=1000)
    rewards = metric_fn(walks)["rewards"]

    config = TRLConfig.load_yaml("configs/ilql_randomwalks.yml")

    trlx.train(
        GPT2Config(n_layer=6, n_embd=144, vocab_size=23),
        # "randomwalks/1M",
        dataset=(walks, rewards),
        eval_prompts=eval_prompts,
        metric_fn=metric_fn,
        config=config,
    )
