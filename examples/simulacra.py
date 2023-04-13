# Optimize prompts by training on prompts-ratings pairings dataset
# taken from https://github.com/JD-P/simulacra-aesthetic-captions

import os
import sqlite3
from urllib.request import urlretrieve

from accelerate import Accelerator

import trlx
from trlx.data.default_configs import default_ilql_config

url = "https://raw.githubusercontent.com/JD-P/simulacra-aesthetic-captions/main/sac_public_2022_06_29.sqlite"
dbpath = "sac_public_2022_06_29.sqlite"

if __name__ == "__main__":
    accelerator = Accelerator()
    if os.environ.get("LOCAL_RANK", "0") == "0" and not os.path.exists(dbpath):
        print(f"fetching {dbpath}")
        urlretrieve(url, dbpath)
    accelerator.wait_for_everyone()

    conn = sqlite3.connect(dbpath)
    c = conn.cursor()
    c.execute(
        "SELECT prompt, rating FROM ratings "
        "JOIN images ON images.id=ratings.iid "
        "JOIN generations ON images.gid=generations.id "
        "WHERE rating IS NOT NULL;"
    )

    prompts, ratings = tuple(map(list, zip(*c.fetchall())))
    trlx.train(
        config=default_ilql_config(),
        samples=prompts,
        rewards=ratings,
        eval_prompts=["An astronaut riding a horse"] * 64,
    )
