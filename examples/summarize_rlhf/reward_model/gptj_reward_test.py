import random

import numpy as np
import torch
from datasets import load_dataset
from reward_model import GPTRewardModel
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def set_seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


def create_comparison_dataset(path="CarperAI/openai_summarize_comparisons", split="train"):
    dataset = load_dataset(path, split=split)
    if split == "test":
        dataset = dataset.select(range(5000))

    pairs = []
    for sample in tqdm(dataset):
        pair = {}
        prompt = sample["prompt"]
        chosen_summary = sample["chosen"]
        rejected_summary = sample["rejected"]
        if chosen_summary == rejected_summary:
            continue
        if len(chosen_summary.split()) < 5 or len(rejected_summary.split()) < 5:
            continue
        pair["chosen"] = prompt + "\n" + chosen_summary
        pair["rejected"] = prompt + "\n" + rejected_summary
        pairs.append(pair)
    return pairs


class PairwiseDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length):
        self.chosen_input_ids = []
        self.chosen_attn_masks = []
        self.rejected_input_ids = []
        self.rejected_attn_masks = []
        for pair in pairs:
            chosen, rejected = pair["chosen"], pair["rejected"]
            chosen_encodings_dict = tokenizer(
                "<|startoftext|>" + chosen + "<|endoftext|>",
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            rejected_encodings_dict = tokenizer(
                "<|startoftext|>" + rejected + "<|endoftext|>",
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            if not torch.all(torch.eq(chosen_encodings_dict["input_ids"], rejected_encodings_dict["input_ids"])).item():
                self.chosen_input_ids.append(chosen_encodings_dict["input_ids"])
                self.chosen_attn_masks.append(chosen_encodings_dict["attention_mask"])
                self.rejected_input_ids.append(rejected_encodings_dict["input_ids"])
                self.rejected_attn_masks.append(rejected_encodings_dict["attention_mask"])

    def __len__(self):
        return len(self.chosen_input_ids)

    def __getitem__(self, idx):
        return (
            self.chosen_input_ids[idx],
            self.chosen_attn_masks[idx],
            self.rejected_input_ids[idx],
            self.rejected_attn_masks[idx],
        )


class DataCollatorReward:
    def __call__(self, data):
        batch = {}
        batch["input_ids"] = torch.cat([f[0] for f in data] + [f[2] for f in data])
        batch["attention_mask"] = torch.cat([f[1] for f in data] + [f[3] for f in data])
        batch["labels"] = torch.tensor([0] * len(data) + [1] * len(data))
        return batch


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    tokenizer.pad_token = tokenizer.eos_token
    PAD_ID = tokenizer(tokenizer.pad_token)["input_ids"][0]

    model = GPTRewardModel("CarperAI/openai_summarize_tldr_sft")
    model.load_state_dict(torch.load("rm_checkpoint/pytorch_model.bin"))
    max_length = 550
    val_pairs = create_comparison_dataset("CarperAI/openai_summarize_comparisons", "test")
    dev_dataset = PairwiseDataset(val_pairs, tokenizer, max_length=max_length)

    from torch.utils.data import DataLoader

    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=6, collate_fn=DataCollatorReward())
    model.cuda()
    model.eval()
    model.half()
    correct = 0
    chosen_list = []
    reject_list = []
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dev_dataloader), total=len(dev_dataloader)):
            for x in batch:
                batch[x] = batch[x].cuda()
            outputs = model(**batch)
            correct += sum(outputs["chosen_end_scores"] > outputs["rejected_end_scores"])
            chosen_list.append(outputs["chosen_end_scores"].cpu())
            reject_list.append(outputs["rejected_end_scores"].cpu())
    print("Total accuracy: ", correct / len(dev_dataset))
