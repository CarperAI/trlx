from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Set

import yaml

from trlx.data.method_configs import MethodConfig, get_method
import os


def merge(base: Dict, update: Dict, updated: Set) -> Dict:
    "Recursively updates a nested dictionary with new values"
    for k, v in base.items():
        if isinstance(v, dict):
            base[k] = merge(v, update, updated)

        for kk, vv in update.items():
            if k == kk:
                base[k] = vv
                updated.add(k)

    return base


@dataclass
class ModelConfig:
    """
    Config for a model.

    :param model_path: Path to the model (local or on huggingface hub)
    :type model_path: str

    :param tokenizer_path: Path to the tokenizer (local or on huggingface hub)
    :type tokenizer_path: str

    :param model_type: One of the registered RL models present in trlx.model
    :type model_type: str
    """

    model_path: str
    tokenizer_path: str
    model_type: str
    num_layers_unfrozen: int = -1

    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        return cls(**config)


@dataclass
class TrainConfig:
    """
    Config for train job on model.

    :param total_steps: Total number of training steps
    :type total_steps: int

    :param seq_length: Number of tokens to use as context (max length for tokenizer)
    :type seq_length: int

    :param epochs: Total number of passes through data
    :type epochs: int

    :param batch_size: Batch size for training
    :type batch_size: int

    :param lr_init: Initial learning rate value
    :type lr_init: float

    :param lr_target: Target learning rate after decay
    :type lr_target: float

    :param opt_betas: Beta parameters for Adam optimizer
    :type opt_betas: Tuple[float]

    :param opt_eps: Epsilon for optimizer
    :type opt_eps: float

    :param weight_decay: Weight decay for optimizer
    :type weight_decay: float

    :param checkpoint_interval: Save model every checkpoint_interval steps
    :type checkpoint_interval: int

    :param eval_interval: Evaluate model every eval_interval steps
    :type eval_interval: int

    :param pipeline: Pipeline to use for training. One of the registered pipelines present in trlx.pipeline
    :type pipeline: str

    :param orchestrator: Orchestrator to use for training. One of the registered orchestrators present in trlx.orchestrator
    :type orchestrator: str

    :param checkpoint_dir: Directory to save checkpoints
    :type checkpoint_dir: str

    :param project_name: Project name for wandb
    :type project_name: str

    :param entity_name: Entity name for wandb
    :type entity_name: str
    
    :param rollout_logging_dir: Directory to store generated rollouts for use in Algorithm Distillation. Only used by AcceleratePPOModel.
    :type rollout_logging_dir: Optional[str]
    """

    total_steps: int
    seq_length: int
    epochs: int
    batch_size: int

    lr_init: float
    lr_target: float
    opt_betas: Tuple[float]
    opt_eps: float
    weight_decay: float

    checkpoint_interval: int
    eval_interval: int
    

    pipeline: str  # One of the pipelines in framework.pipeline
    orchestrator: str  # One of the orchestrators

    checkpoint_dir: str = "ckpts"
    project_name: str = "trlx"
    entity_name: Optional[str] = None
    seed: int = 1000
    
    rollout_logging_dir: Optional[str] = None

    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        return cls(**config)


@dataclass
class TRLConfig:
    """
    Top level config for trlX. Loads configs and can be converted to dictionary.
    """

    model: ModelConfig
    train: TrainConfig
    method: MethodConfig

    @classmethod
    def load_yaml(cls, yml_fp: str):
        """
        Load yaml file as TRLConfig.

        :param yml_fp: Path to yaml file
        :type yml_fp: str
        """
        with open(yml_fp, mode="r") as file:
            config = yaml.safe_load(file)
        return cls(
            ModelConfig.from_dict(config["model"]),
            TrainConfig.from_dict(config["train"]),
            get_method(config["method"]["name"]).from_dict(config["method"]),
        )

    def to_dict(self):
        """
        Convert TRLConfig to dictionary.
        """
        data = {
            "model": self.model.__dict__,
            "train": self.train.__dict__,
            "method": self.method.__dict__,
        }

        return data

    @classmethod
    def from_dict(cls, config_dict: dict):
        """
        Convert dictionary to TRLConfig.
        """
        return cls(
            ModelConfig.from_dict(config_dict["model"]),
            TrainConfig.from_dict(config_dict["train"]),
            get_method(config_dict["method"]["name"]).from_dict(config_dict["method"]),
        )

    @classmethod
    def update(cls, baseconfig, config):
        updates = set()
        merged = merge(baseconfig, config, updates)

        for param in config:
            if param not in updates:
                raise ValueError(
                    f"parameter {param} is not present in the config (typo or a wrong config)"
                )

        return cls.from_dict(merged)
