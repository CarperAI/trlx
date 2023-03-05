from typing import Callable, List

# Register load pipelines via module import
from trlx.pipeline import _DATAPIPELINE
from trlx.pipeline.offline_pipeline import PromptPipeline

# Register load trainers via module import
from trlx.trainer import _TRAINERS, register_trainer
from trlx.trainer.accelerate_ilql_trainer import AccelerateILQLTrainer
from trlx.trainer.accelerate_ppo_trainer import AcceleratePPOTrainer
from trlx.trainer.accelerate_sft_trainer import AccelerateSFTTrainer

try:
    from trlx.trainer.nemo_ilql_trainer import NeMoILQLTrainer
    from trlx.trainer.nemo_sft_trainer import NeMoSFTTrainer
except ImportError:
    # NeMo is not installed
    def _trainers_unavailble(names: List[str]):
        def log_error(*args, **kwargs):
            raise ImportError("NeMo is not installed. Please install `nemo_toolkit` to use NeMo-based trainers.")

        # Register dummy trainers
        for name in names:
            register_trainer(name)(log_error)

    _trainers_unavailble(["NeMoILQLTrainer", "NeMoSFTTrainer"])


def get_trainer(name: str) -> Callable:
    """
    Return constructor for specified RL model trainer
    """
    name = name.lower()
    if name in _TRAINERS:
        return _TRAINERS[name]
    else:
        raise Exception("Error: Trying to access a trainer that has not been registered")


def get_pipeline(name: str) -> Callable:
    """
    Return constructor for specified pipeline
    """
    name = name.lower()
    if name in _DATAPIPELINE:
        return _DATAPIPELINE[name]
    else:
        raise Exception("Error: Trying to access a pipeline that has not been registered")
