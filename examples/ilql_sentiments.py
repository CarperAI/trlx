from trlx.orchestrator.offline_orchestrator import OfflineOrchestrator
from trlx.model.accelerate_ilql_model import ILQLModel
from trlx.data.configs import TRLConfig

from transformers import AutoTokenizer, pipeline
from datasets import load_dataset

import math
import torch
import numpy as np
from typing import List, Iterable, Callable
from tqdm import tqdm

def batch_map(fn: Callable, xs: Iterable, bsize: int, desc=None):
    out = []
    for ind in tqdm(range(math.ceil(len(xs) / bsize)), desc=desc, disable=not desc):
        batch = xs[ind*bsize:min(len(xs), (ind+1)*bsize)]
        out.extend(fn(batch))

    return out

if __name__ == '__main__':
    config = TRLConfig.load_yaml('configs/ilql_gptj.yml')
    sentiment_pipe = pipeline('sentiment-analysis', 'lvwerra/distilbert-imdb', device=torch.device(0))

    gpt_config_or_path = 'EleutherAI/gpt-j-6B'
    tokenizer = AutoTokenizer.from_pretrained(gpt_config_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    def reward_fn(samples: List[str]) -> List[float]:
        if isinstance(samples[0], torch.Tensor):
            samples = tokenizer.batch_decode(samples, skip_special_tokens=True)

        desc = 'sentiment pipeline' if len(samples) > 1024 else None
        sentiments = batch_map(lambda batch: sentiment_pipe(batch), samples, bsize=1024, desc=desc)
        return [1-s['score'] if s['label'] == 'NEGATIVE' else s['score'] for s in sentiments]

    model = ILQLModel(
        config=config,
        gpt_config_or_path=gpt_config_or_path,
        tokenizer=tokenizer
    )

    n_prompts = 128
    eval_prompts = torch.tensor([model.tokenizer.bos_token_id] * n_prompts).view(n_prompts, 1)
    train_samples = load_dataset('imdb', split='train+test')

    #TODO(dahoas)
    train_samples = train_samples.filter(lambda x: len(x["text"])<500, batched=False)['text']

    orch = OfflineOrchestrator(
        model,
        train_samples,
        eval_prompts,
        reward_fn
    )

    model.learn()
