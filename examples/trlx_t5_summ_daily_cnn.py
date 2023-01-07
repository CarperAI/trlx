from typing import List

import evaluate
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

import trlx
from trlx.data.configs import TRLConfig

meteor = evaluate.load("meteor")

if __name__ == "__main__":

    def reward_fn(samples: List[str]):
        sep_token = tokenizer.sep_token
        articles = [sample.split(sep_token)[0].strip() for sample in samples]
        summs = [sample.split(sep_token)[1].strip() for sample in samples]
        labels = [prompt_label[sample] for sample in articles]
        scores = [
            meteor.compute(predictions=[summary], references=[label])
            for (summary, label) in zip(summs, labels)
        ]
        scores = [score["meteor"] for score in scores]
        return scores

    config = TRLConfig.load_yaml("configs/ppo_config_cnn_daily.yml")

    dataset = load_dataset("cnn_dailymail", "3.0.0", split="train", cache_dir="data")
    prompts = dataset["article"][0:10000]
    summaries = dataset["highlights"][0:10000]
    prompts = ["Summarize: " + prompt for prompt in prompts]
    val_dataset = load_dataset(
        "cnn_dailymail", "3.0.0", split="validation", cache_dir="data"
    )
    val_prompts = ["Summarize: " + prompt for prompt in val_dataset["article"][0:1000]]
    val_summaries = val_dataset["highlights"][0:1000]

    tokenizer = AutoTokenizer.from_pretrained(config.model.model_path)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"
    tokenizer.sep_token = "<sep>"
    prompt_label = {}
    max_length = config.train.seq_length - config.method.gen_kwargs["max_new_tokens"]

    for i in tqdm(range(len(prompts))):
        key = tokenizer.decode(
            tokenizer(prompts[i], truncation=True, max_length=max_length)["input_ids"],
            skip_special_tokens=True,
        )
        prompt_label[key.strip()] = summaries[i]

    for i in tqdm(range(len(val_prompts))):
        key = tokenizer.decode(
            tokenizer(val_prompts[i], truncation=True, max_length=max_length)[
                "input_ids"
            ],
            skip_special_tokens=True,
        )
        prompt_label[key.strip()] = val_summaries[i]

    model = trlx.train(
        config.model.model_path,
        reward_fn=reward_fn,
        prompts=prompts,
        eval_prompts=val_prompts[0:100],
        config=config,
    )
