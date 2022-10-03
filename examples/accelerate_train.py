from framework.model.accelerate_ppo_model import AcceleratePPOModel
from framework.orchestrator.sentiment_ppo_orch import PPOSentimentOrchestrator
from framework.pipeline.ppo_pipeline import PPOPipeline
from framework.data.configs import TRLConfig

from framework.utils.loading import get_model, get_pipeline, get_orchestrator
from framework.eval.sentiment import sentiment_eval

import wandb


if __name__ == "__main__":
    cfg = TRLConfig.load_yaml("configs/ppo_config.yml")

    
    model : AcceleratePPOModel = get_model(cfg.model.model_type)(cfg)
    wandb.watch(model.model)

    pipeline : PPOPipeline = get_pipeline(cfg.train.pipeline)(model.tokenize, cfg)
    orch : PPOSentimentOrchestrator = get_orchestrator(cfg.train.orchestrator)(pipeline, model, cfg.method.chunk_size)
    orch.make_experience(cfg.method.num_rollouts)
    model.learn()

    print("DONE!")
