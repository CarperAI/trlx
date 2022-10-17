import trlx

if __name__ == "__main__":
    def reward_fn(samples):
        "Gives a negative count of rooms for each sample"
        nrooms = []
        for s in samples:
            count = 0.0
            for char in s:
                if char == ':':
                    count -= 1
            nrooms.append(count)

        return nrooms

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
        "[prompt] the kitchen is not adjacent to the bathroom [layout]"
    ]

    model = trlx.train(
        model_path="architext/gptj-162M",
        reward_fn=reward_fn,
        prompts=prompts
    )

