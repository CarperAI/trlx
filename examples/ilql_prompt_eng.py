import math
from typing import List

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, pipeline, set_seed

from trlx.data.configs import TRLConfig
from trlx.model.accelerate_ilql_model import ILQLModel
from trlx.orchestrator.offline_orchestrator import OfflineOrchestrator

set_seed(42)
base_prompt = "The following is a series of movie reviews, along with a measure of their positive-ness on a scale of 0 to 1, with 1 indicating very positive sentiment:\nThis film is just plain horrible: 0\n This film is great:1\n"

if __name__ == "__main__":
    generator = pipeline('text-generation', model='gpt2')
    
    config = TRLConfig.load_yaml("configs/ilql_config.yml")

    tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token

    def reward_fn(samples: List[str]) -> List[float]:
        if isinstance(samples[0], torch.Tensor):
            samples = tokenizer.batch_decode(samples, skip_special_tokens=True)

        def get_sentiment_lm(review):
            full_prompt = base_prompt + review + ":"
            return int(generator(full_prompt, max_length=1)[0]['generated_text'][-1])

        scores = torch.tensor([get_sentiment_lm(review) for review in samples])
        return scores

    model = ILQLModel(config=config, tokenizer=tokenizer)

    n_prompts = 128
    eval_prompts = torch.tensor([model.tokenizer.bos_token_id] * n_prompts).view(
        n_prompts, 1
    )
    train_samples = load_dataset("imdb", split="train+test")["text"]

    orch = OfflineOrchestrator(model, train_samples, eval_prompts, reward_fn)

    model.learn()
