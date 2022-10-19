import trlx


def reward_fn(samples):
    "Gives a negative count of rooms for each sample"
    return [-sample.count(":") for sample in samples]


if __name__ == "__main__":
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

    model = trlx.train("architext/gptj-162M", reward_fn=reward_fn, prompts=prompts)
