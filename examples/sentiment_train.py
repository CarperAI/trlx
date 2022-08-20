from framework.model.sentiment import SentimentILQLModel
from framework.orchestrator.sentiment import OfflineSentimentOrchestrator
from framework.pipeline.sentiment import SentimentPipeline
from framework.configs import TRLConfig

from framework.utils.loading import get_model, get_pipeline, get_orchestrator
from framework.eval.sentiment import sentiment_eval

import wandb

def log_fn(d : dict):
    wandb.log(d)

def eval_fn(model : SentimentILQLModel):
    avg_score = sentiment_eval(model)
    wandb.log({"Average Sentiment" : avg_score})

if __name__ == "__main__":
    wandb.init(project = "trl-tests")

    cfg = TRLConfig.load_yaml("configs/sentiment_config.yml")

    model : SentimentILQLModel = get_model(cfg.model.model_type)(cfg)
    wandb.watch(model.model)

    pipeline : SentimentPipeline = get_pipeline(cfg.train.pipeline)()
    orch : OfflineSentimentOrchestrator = get_orchestrator(cfg.train.orchestrator)(pipeline, model)

    orch.make_experience()
    model.learn(log_fn = log_fn, eval_fn = eval_fn)

    model.save("./")
    print("DONE!")
