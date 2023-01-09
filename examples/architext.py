# Toy example of optimizing textual interior designs to output the least number of rooms
# Also see https://architext.design/
import yaml

import trlx
from trlx.data.configs import TRLConfig


def reward_fn(samples):
    "Gives a negative count of rooms for each sample"
    return [-sample.count(":") for sample in samples]


prompts = [
    "[prompt] the bedroom is adjacent to the living room [layout]",
    "[prompt] a bedroom is adjacent to the living room [layout]",
    "[prompt] the bedroom is adjacent to the kitchen [layout]",
    "[prompt] a bedroom is adjacent to the kitchen [layout]",
    "[prompt] the bedroom is adjacent to the kitchen [layout]",
    "[prompt] the kitchen is adjacent to the bathroom [layout]",
    "[prompt] a bathroom is adjacent to the living room [layout]",
    "[prompt] the bathroom is adjacent to the living room [layout]",
    "[prompt] the bedroom is not adjacent to the living room [layout]",
    "[prompt] a bedroom is not adjacent to the living room [layout]",
    "[prompt] the bedroom is not adjacent to the kitchen [layout]",
    "[prompt] a bedroom is not adjacent to the kitchen [layout]",
    "[prompt] the bedroom is not adjacent to the kitchen [layout]",
    "[prompt] the kitchen is not adjacent to the bathroom [layout]",
]

default_config = yaml.safe_load(open("configs/ppo_config.yml"))


def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)

    trlx.train(
        "architext/gptj-162M", reward_fn=reward_fn, prompts=prompts, config=config
    )


if __name__ == "__main__":
    main()
