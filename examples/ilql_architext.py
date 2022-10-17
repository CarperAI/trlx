import trlx
import csv

if __name__ == "__main__":
    def metric_fn(samples):
        "Gives a negative count of rooms for each sample"
        nrooms = []
        for s in samples:
            count = 0.0
            for char in s:
                if char == ':':
                    count -= 1
            nrooms.append(count)

        return {'nrooms': nrooms}

    samples, rewards = tuple(map(list, zip(*csv.reader(open("data/architext-samples.csv")))))
    rewards = list(map(float, rewards))
    prompts = open("data/architext-prompts.csv").read().splitlines()

    trlx.train(
        samples,
        rewards,
        model_path='architext/gptj-162M',
        eval_prompts=prompts,
        metric_fn=metric_fn,
        split_token='[layout] '
    )

