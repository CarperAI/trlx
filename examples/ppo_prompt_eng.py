from typing import List
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, pipeline, set_seed, GPT2Model, GPT2Config
from random import randint
import wandb
from trlx.data.configs import TRLConfig
from trlx.model.accelerate_ppo_model import AcceleratePPOModel
from trlx.orchestrator.ppo_orchestrator import PPOOrchestrator
from trlx.pipeline.ppo_pipeline import PPOPipeline
from trlx.utils.loading import get_model, get_orchestrator, get_pipeline

configuration = GPT2Config()
lang_model = GPT2Model(configuration)

with open('examples/prompt_eng_template.txt', 'r') as f:
    base_prompt = f.read()

def construct_full_prompt(review1, review2):
    return base_prompt.format(review1, review2)

if __name__ == "__main__":
    # generator = pipeline('text-generation', model='gpt2')
    
    cfg = TRLConfig.load_yaml("configs/ppo_config.yml")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token

    def reward_fn(samples: List[str]) -> List[float]:
        if isinstance(samples[0], torch.Tensor):
            samples = tokenizer.batch_decode(samples, skip_special_tokens=True)

        def get_probabilities_forward(list_of_reviews):
            batch = tokenizer(list_of_reviews, return_tensors="pt", padding=True)
            model = gpt2(**batch)
            logits = model.logits

            a_token = tokenizer("A", return_tensors="pt").input_ids[0][0]
            b_token = tokenizer("B", return_tensors="pt").input_ids[0][0]
            
            # get logits that correspond to 'A' and 'B'
            a_logits = logits[:, -1, a_token]
            b_logits = logits[:, -1, b_token]

            a_probs = torch.softmax(a_logits, -1)
            b_probs = torch.softmax(b_logits, -1)
            
            return (b_probs > a_probs).float()
        
        def expected(A, B):
            return 1 / (1 + 10 ** ((B - A) / 400))

        def elo(old, exp, score, k=32):
            return old + k * (score - exp)

        # for each review, get its elo score with respect to all other reviews
        # then, average the elo scores to get the final score
        scores = []
        for review in samples:
            review_pairs = [construct_full_prompt(review, review2) for review2 in samples if review != review2]
            probabilities = get_probabilities_forward(review_pairs)
            probabilities = probabilities.view(len(samples) - 1, len(samples) - 1)
            probabilities = probabilities.mean(dim=0)
            scores.append(probabilities.sum().item())
        
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
