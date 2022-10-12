from typing import List

import torch
from transformers import pipeline

import wandb
from trlx.data.configs import TRLConfig
from trlx.model.accelerate_ppo_model import AcceleratePPOModel
from trlx.orchestrator.ppo_orchestrator import PPOOrchestrator
from trlx.pipeline.ppo_pipeline import PPOPipeline
from trlx.utils.loading import get_model, get_orchestrator, get_pipeline

if __name__ == "__main__":
    cfg = TRLConfig.load_yaml("configs/ppo_gptj.yml")

    def reward_fn(samples: List[str]):
        scores = torch.tensor([1.0 if 'happy' in sample else 0.0 for sample in samples])
        return scores

    model: AcceleratePPOModel = get_model(cfg.model.model_type)(cfg)
    if model.accelerator.is_main_process:
        wandb.watch(model.model)

    pipeline: PPOPipeline = get_pipeline(cfg.train.pipeline)(model.tokenizer, cfg)
    orch: PPOOrchestrator = get_orchestrator(cfg.train.orchestrator)(
        model, pipeline, reward_fn=reward_fn, chunk_size=cfg.method.chunk_size
    )
    orch.make_experience(cfg.method.num_rollouts)
    model.learn()

    print("DONE!")
