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
    cfg = TRLConfig.load_yaml("configs/ppo_config.yml")

    sentiment_pipe = pipeline(
        "sentiment-analysis", "lvwerra/distilbert-imdb", device=-1
    )

    def reward_fn(samples: List[str]):
        sent_kwargs = {
            "return_all_scores": True,
            "function_to_apply": None,
            "batch_size": cfg.method.chunk_size,
        }
        pipe_outputs = sentiment_pipe(samples, **sent_kwargs)
        scores = torch.tensor([output[1]["score"] for output in pipe_outputs])
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
