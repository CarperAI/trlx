import math
from typing import List

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, pipeline, set_seed
from random import randint

from trlx.data.configs import TRLConfig
from trlx.model.accelerate_ilql_model import ILQLModel
from trlx.orchestrator.offline_orchestrator import OfflineOrchestrator

set_seed(42)
with open('prompt_eng_template.txt', 'r') as f:
    base_prompt = f.read()
def construct_full_prompt(review1, review2):
    return base_prompt.format(review1, review2)

if __name__ == "__main__":
    generator = pipeline('text-generation', model='gpt2')
    
    config = TRLConfig.load_yaml("configs/ilql_config.yml")

    tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token

    def reward_fn(samples: List[str]) -> List[float]:
        if isinstance(samples[0], torch.Tensor):
            samples = tokenizer.batch_decode(samples, skip_special_tokens=True)

        def get_sentiment_lm(review1, review2):
            full_prompt = construct_full_prompt(review1, review2)
            reward = generator(full_prompt, max_new_tokens=1, pad_token_id=50256)[0]['generated_text'][-1]
            return 0 if reward == 'B' else 1

        scores = [get_sentiment_lm(review, samples[randint(0, len(samples))]) for review in samples]
        return torch.tensor(scores)

    model = ILQLModel(config=config, tokenizer=tokenizer)

    n_prompts = 128
    eval_prompts = torch.tensor([model.tokenizer.bos_token_id] * n_prompts).view(
        n_prompts, 1
    )
    train_samples = load_dataset("imdb", split="train+test")["text"]

    orch = OfflineOrchestrator(model, train_samples, eval_prompts, reward_fn)

    model.learn()
