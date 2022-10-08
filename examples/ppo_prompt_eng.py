from typing import List
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, pipeline, set_seed
from random import randint
import wandb
from trlx.data.configs import TRLConfig
from trlx.model.accelerate_ppo_model import AcceleratePPOModel
from trlx.orchestrator.ppo_orchestrator import PPOOrchestrator
from trlx.pipeline.ppo_pipeline import PPOPipeline
from trlx.utils.loading import get_model, get_orchestrator, get_pipeline

set_seed(42)
with open('./prompt_eng_template.txt', 'r') as f:
    base_prompt = f.read()
def construct_full_prompt(review1, review2):
    return base_prompt.format(review1, review2)

if __name__ == "__main__":
    generator = pipeline('text-generation', model='gpt2')
    
    cfg = TRLConfig.load_yaml("configs/ppo_config.yml")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_path)
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
    
    model: AcceleratePPOModel = get_model(cfg.model.model_type)(cfg)
    if model.accelerator.is_main_process:
        wandb.watch(model.model)

    pipeline: PPOPipeline = get_pipeline(cfg.train.pipeline)(model.tokenizer, cfg)
    orch: PPOOrchestrator = get_orchestrator(cfg.train.orchestrator)(
        model, pipeline, reward_fn=reward_fn, chunk_size=cfg.method.chunk_size
    )
    orch.make_experience(cfg.method.num_rollouts)
    model.learn()
