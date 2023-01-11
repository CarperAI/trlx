# Optimize prompts by training on prompts-ratings pairings dataset
# taken from https://github.com/JD-P/simulacra-aesthetic-captions

import os
import sqlite3
from urllib.request import urlretrieve
import math

import trlx
from trlx.data.configs import TRLConfig


if __name__ == "__main__":
    def reward_fn(list_of_str):
        return [s.count('s')/math.sqrt(len(s)) for s in list_of_str]


    trlx.train(
        "EleutherAI/gpt-j-6b",
        reward_fn=reward_fn,
        config=TRLConfig.load_yaml("configs/ppo_gptj.yml"),
        #eval_prompts=["She sells"] * 4,
    )
