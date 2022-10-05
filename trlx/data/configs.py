import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import yaml

from trlx.data.method_configs import MethodConfig, get_method


@dataclass
class ModelConfig:
    model_path: str
    tokenizer_path: str
    model_type: str  # One of the architectures present in framework.model
    device: str = ""
    num_layers_unfrozen: int = -1

    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        return cls(**config)


@dataclass
class TrainConfig:
    n_ctx: int
    epochs: int
    total_steps: int
    batch_size: int
    grad_clip: float  # Clip grad norms to this value

    lr_ramp_steps: int
    lr_decay_steps: int
    weight_decay: float
    learning_rate_init: float
    learning_rate_target: float

    log_interval: int
    checkpoint_interval: int
    eval_interval: int

    pipeline: str  # One of the pipelines in framework.pipeline
    orchestrator: str  # One of the orchestrators

    input_size: int = 0  # max model input size
    gen_size: int = 1024  # max size of model generation

    accelerate: bool = True  # Use HF accelerate?
    accelerate_config_path: str = ""

    project_name: str = ""

    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        return cls(**config)


@dataclass
class TRLConfig:
    model: ModelConfig
    train: TrainConfig
    method: MethodConfig

    @classmethod
    def load_yaml(cls, yml_fp: str):
        with open(yml_fp, mode="r") as file:
            config = yaml.safe_load(file)
        return cls(
            ModelConfig.from_dict(config["model"]),
            TrainConfig.from_dict(config["train"]),
            get_method(config["method"]["name"]).from_dict(config["method"]),
        )

    def to_dict(self):
        data = self.model.__dict__.copy()
        data.update(self.train.__dict__)
        data.update(self.method.__dict__)
        return data
