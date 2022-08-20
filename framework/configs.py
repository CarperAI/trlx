from dataclasses import dataclass
from typing import Any, Dict
import yaml

@dataclass
class ModelConfig:
    model_path : str
    model_type : str # One of the architectures present in framework.model
    device : str
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        return cls(**config)

@dataclass
class TrainConfig:
    n_ctx : int
    epochs : int
    batch_size : int
    grad_clip : float # Clip grad norms to this value

    lr_ramp_steps: int
    lr_decay_steps: int
    weight_decay : float
    learning_rate_init: float
    learning_rate_target: float

    log_interval: int
    checkpoint_interval: int
    eval_interval : int

    pipeline : str # One of the pipelines in framework.pipeline
    orchestrator : str # One of the orchestrators

    accelerate : bool # Use HF accelerate?

    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        return cls(**config)

@dataclass
class TRLConfig:
    model : ModelConfig
    train : TrainConfig

    @classmethod
    def load_yaml(cls, yml_fp: str):
        with open(yml_fp, mode="r") as file:
            config = yaml.safe_load(file)
        return cls(
            ModelConfig.from_dict(config["model"]),
            TrainConfig.from_dict(config["train"]),
        )

    def to_dict(self):
        data = self.model.__dict__.copy()
        data.update(self.train_job.__dict__)
        return
        