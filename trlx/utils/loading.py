from typing import Callable

# Register load orchestrators via module import
from trlx.orchestrator import _ORCH
from trlx.orchestrator.offline_orchestrator import OfflineOrchestrator
from trlx.orchestrator.ppo_orchestrator import PPOOrchestrator

# Register load pipelines via module import
from trlx.pipeline import _DATAPIPELINE
from trlx.pipeline.offline_pipeline import PromptPipeline

# Register load trainers via module import
from trlx.trainer import _TRAINERS
from trlx.trainer.accelerate_ilql_trainer import AccelerateILQLTrainer
from trlx.trainer.accelerate_ppo_trainer import AcceleratePPOTrainer


def get_trainer(name: str) -> Callable:
    """
    Return constructor for specified RL model trainer
    """
    name = name.lower()
    if name in _TRAINERS:
        return _TRAINERS[name]
    else:
        raise Exception(
            "Error: Trying to access a trainer that has not been registered"
        )


def get_pipeline(name: str) -> Callable:
    """
    Return constructor for specified pipeline
    """
    name = name.lower()
    if name in _DATAPIPELINE:
        return _DATAPIPELINE[name]
    else:
        raise Exception(
            "Error: Trying to access a pipeline that has not been registered"
        )


def get_orchestrator(name: str) -> Callable:
    """
    Return constructor for specified orchestrator
    """
    name = name.lower()
    if name in _ORCH:
        return _ORCH[name]
    else:
        raise Exception(
            "Error: Trying to access an orchestrator that has not been registered"
        )
