from typing import Callable

from framework.model.sentiment import SentimentILQLModel
from framework.orchestrator.sentiment import SentimentOrchestrator
from framework.pipeline.sentiment import SentimentPipeline
from framework.configs import TRLConfig

from framework.pipeline import _DATAPIPELINE
from framework.orchestrator import _ORCH
from framework.model import _MODELS

def get_model(name : str) -> Callable:
    """
    Return constructor for specified model
    """
    name = name.lower()
    if name in _MODELS:
        return _MODELS[name]
    else:
        raise Exception("Error: Trying to access a model that has not been registered")

def get_pipeline(name : str) -> Callable:
    """
    Return constructor for specified pipeline
    """
    name = name.lower()
    if name in _DATAPIPELINE:
        return _DATAPIPELINE[name]
    else:
        raise Exception("Error: Trying to access a pipeline that has not been registered")

def get_orchestrator(name : str) -> Callable:
    """
    Return constructor for specified orchestrator
    """
    name = name.lower()
    if name in _ORCH:
        return _ORCH[name]
    else:
        raise Exception("Error: Trying to access an orchestrator that has not been registered")

if __name__ == "__main__":
    cfg = TRLConfig.load_yaml("configs/sentiment_config.yml")

    model : SentimentILQLModel = get_model(cfg.model.model_arch)(cfg)
    pipeline : SentimentPipeline = get_pipeline(cfg.train.pipeline)()
    orch : SentimentOrchestrator = get_orchestrator(cfg.train.orchestrator)(pipeline, model)

    orch.make_experience()
    model.learn()
    print("DONE!")


