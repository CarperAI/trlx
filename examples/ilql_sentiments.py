import math
from typing import List

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, pipeline

from trlx.data.configs import TRLConfig
from trlx.model.accelerate_ilql_model import ILQLModel
from trlx.orchestrator.offline_orchestrator import OfflineOrchestrator

if __name__ == "__main__":
    config = TRLConfig.load_yaml("configs/ilql_config.yml")
    sentiment_pipe = pipeline(
        "sentiment-analysis", "lvwerra/distilbert-imdb", device=torch.device(0)
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token

    def reward_fn(samples: List[str]) -> List[float]:
        if isinstance(samples[0], torch.Tensor):
            samples = tokenizer.batch_decode(samples, skip_special_tokens=True)

        sent_kwargs = {
            "return_all_scores": True,
            "function_to_apply": None,
            "batch_size": 1024,
        }
        pipe_outputs = sentiment_pipe(samples, **sent_kwargs)
        scores = torch.tensor([output[1]["score"] for output in pipe_outputs])
        return scores

    model = ILQLModel(config=config, tokenizer=tokenizer)

    n_prompts = 128
    eval_prompts = torch.tensor([model.tokenizer.bos_token_id] * n_prompts).view(
        n_prompts, 1
    )
    train_samples = load_dataset("imdb", split="train+test")["text"]

    orch = OfflineOrchestrator(model, train_samples, eval_prompts, reward_fn)

    model.learn()
