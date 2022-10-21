# Optimize prompts by training on prompts-ratings pairings dataset
# taken from https://github.com/JD-P/simulacra-aesthetic-captions

import sqlite3

import trlx

if __name__ == "__main__":
    conn = sqlite3.connect("data/sac_public_2022_06_29.sqlite")
    c = conn.cursor()
    c.execute(
        "SELECT prompt, rating FROM ratings "
        "JOIN images ON images.id=ratings.iid "
        "JOIN generations ON images.gid=generations.id "
        "WHERE rating IS NOT NULL;"
    )

    prompts, ratings = tuple(map(list, zip(*c.fetchall())))
    model = trlx.train(
        "gpt2",
        dataset=(prompts, ratings),
        eval_prompts=["Hatsune Miku, Red Dress"] * 64,
    )
