import json

import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import Dataset


def get_dataset_from_jsonl(jsonl_file, return_summary=True):
    # if return_summary is True, return a list of posts with summary concatenated
    # if return_summary is False, return a list of posts and a list of summaries
    with open(jsonl_file, "r") as f:
        dataset = [json.loads(line) for line in f]
    post_list = []
    summary_list = []
    for d in dataset:
        if return_summary:
            post = f"SUBREDDIT: r/{d['subreddit']}\nTITLE: {d['title']}\nPOST: {d['post']}\nTL;DR: {d['summary']}"
        else:
            post = f"SUBREDDIT: r/{d['subreddit']}\nTITLE: {d['title']}\nPOST: {d['post']}\nTL;DR: "
            summary_list.append(d["summary"])
        post_list.append(post)
    if not return_summary:
        return post_list, summary_list
    return post_list


class TLDRDataset(Dataset):
    def __init__(self, train_path, tokenizer, split, max_length=550):
        self.post_list = []
        dataset = load_dataset(train_path, split=split)
        for sample in dataset:
            self.post_list.append(sample["prompt"] + sample["label"])
        if "valid" in split:
            self.post_list = self.post_list[0:2000]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_ids = []
        self.attn_masks = []

    def __len__(self):
        return len(self.post_list)

    def __getitem__(self, idx):
        txt = self.post_list[idx]
        encodings_dict = self.tokenizer(txt, truncation=True, max_length=self.max_length, padding="max_length")
        input_ids = torch.tensor(encodings_dict["input_ids"])
        attn_masks = torch.tensor(encodings_dict["attention_mask"])

        return {
            "input_ids": input_ids,
            "attention_mask": attn_masks,
            "labels": input_ids,
        }


class ComparisonDataset(Dataset):
    def __init__(self, comparison_path, tokenizer, max_length=550):
        with open(comparison_path, "r") as f:
            dataset = [json.loads(line) for line in f]

        self.tokenizer = tokenizer
        self.post_list = []
        self.summaries_0 = []
        self.summaries_1 = []
        self.labels = []
        self.max_length = max_length

        def make_text(post, summarize):
            return f"SUBREDDIT: r/{post['subreddit']}\nTITLE: {post['title']}\nPOST: {post['post']}\nTL;DR: {summarize}"

        for sample in dataset:  # chosen summary is always the first one
            self.post_list.append(sample["info"]["post"])
            # NOTE: The chosen summary is always the first one, i.e. `sample["summaries"][0]`
            if sample["choice"] == 0:
                self.summaries_0.append(make_text(sample["info"], sample["summaries"][0]["text"]))
                self.summaries_1.append(make_text(sample["info"], sample["summaries"][1]["text"]))
            else:
                self.summaries_0.append(make_text(sample["info"], sample["summaries"][1]["text"]))
                self.summaries_1.append(make_text(sample["info"], sample["summaries"][0]["text"]))
            self.labels.append(0)

    def __len__(self):
        return len(self.post_list)

    def __getitem__(self, idx):
        summ0 = self.summaries_0[idx]
        summ1 = self.summaries_1[idx]
        encodings_dict = self.tokenizer(
            [summ0, summ1],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        input_ids = torch.tensor(encodings_dict["input_ids"])
        attention_mask = torch.tensor(encodings_dict["attention_mask"])
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class AllSummDataset(Dataset):
    def __init__(self, train_path, tokenizer, split, max_length=1024):
        df = pd.read_parquet(train_path)
        if split == "valid":
            df = df.sample(n=5000)
        self.summarizes = []
        for i, row in df.iterrows():
            self.summarizes.append(f"Summarize: {row['text']}. TL;DR: {row['summary']}")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_ids = []
        self.attn_masks = []

    def __len__(self):
        return len(self.summarizes)

    def __getitem__(self, idx):
        txt = self.summarizes[idx]
        encodings_dict = self.tokenizer(txt, truncation=True, max_length=self.max_length, padding="max_length")
        input_ids = torch.tensor(encodings_dict["input_ids"])
        attn_masks = torch.tensor(encodings_dict["attention_mask"])

        return {
            "input_ids": input_ids,
            "attention_mask": attn_masks,
            "labels": input_ids,
        }
