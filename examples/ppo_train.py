from trlx.model.accelerate_ppo_model import AcceleratePPOModel
from trlx.orchestrator.sentiment_ppo_orch import PPOSentimentOrchestrator
from trlx.pipeline.ppo_pipeline import PPOPipeline
from trlx.data.configs import TRLConfig
from trlx.utils.loading import get_model, get_pipeline, get_orchestrator

import wandb


if __name__ == "__main__":
    cfg = TRLConfig.load_yaml("configs/ppo_config.yml")


    model : AcceleratePPOModel = get_model(cfg.model.model_type)(cfg)
    wandb.watch(model.model)

    pipeline : PPOPipeline = get_pipeline(cfg.train.pipeline)(model.tokenizer, cfg)
    orch : PPOSentimentOrchestrator = get_orchestrator(cfg.train.orchestrator)(pipeline, model, cfg.method.chunk_size)
    orch.make_experience(cfg.method.num_rollouts)
    model.learn()

    print("DONE!")
