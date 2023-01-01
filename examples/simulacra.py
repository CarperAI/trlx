# Optimize prompts by training on prompts-ratings pairings dataset
# taken from https://github.com/JD-P/simulacra-aesthetic-captions

import os
import sqlite3
from urllib.request import urlretrieve

import trlx

url = "https://raw.githubusercontent.com/JD-P/simulacra-aesthetic-captions/main/sac_public_2022_06_29.sqlite"
dbpath = "sac_public_2022_06_29.sqlite"

if __name__ == "__main__":
    if not os.path.exists(dbpath):
        print(f"fetching {dbpath}")
        urlretrieve(url, dbpath)

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
        "gpt2",
        dataset=(prompts, ratings),
        eval_prompts=["Hatsune Miku, Red Dress"] * 64,
    )
