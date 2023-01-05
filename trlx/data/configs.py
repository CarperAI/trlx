from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set, Tuple

import yaml

from trlx.data.method_configs import MethodConfig, get_method


def merge(base: Dict, update: Dict, updated: Set) -> Dict:
    "Recursively updates a nested dictionary with new values"
    for k, v in base.items():
        if isinstance(v, dict):
            base[k] = merge(v, update, updated)
        elif k in update:
            base[k] = update[k]
            updated.add(k)

    return base


@dataclass
class ModelConfig:
    """
    Config for a model.

    :param model_path: Path or name of the model (local or on huggingface hub)
    :type model_path: str

    :param tokenizer_path: Path or name of the tokenizer (local or on huggingface hub)
    :type tokenizer_path: str

    :param num_layers_unfrozen: Number of layers to unfreeze for fine-tuning.
        -1 means all layers are unfrozen.
    :type num_layers_unfrozen: int
    """

    model_path: str
    tokenizer_path: str
    num_layers_unfrozen: int = -1

    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        return cls(**config)


@dataclass
class OptimizerConfig:
    """
    Config for an optimizer.

    :param name: Name of the optimizer
    :type name: str

    :param kwargs: Keyword arguments for the optimizer (e.g. lr, betas, eps, weight_decay)
    :type kwargs: Dict[str, Any]
    """

    name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        return cls(**config)


@dataclass
class SchedulerConfig:
    """
    Config for a learning rate scheduler.

    :param name: Name of the scheduler
    :type name: str

    :param kwargs: Keyword arguments for the scheduler instance (e.g. warmup_steps, T_max)
    :type kwargs: Dict[str, Any]
    """

    name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)

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

    :param trackers: Tuple of trackers to use for logging. Default: ("wandb",)
    :type trackers: Tuple[str]

    :param checkpoint_interval: Save model every checkpoint_interval steps
    :type checkpoint_interval: int

    :param eval_interval: Evaluate model every eval_interval steps
    :type eval_interval: int

    :param pipeline: Pipeline to use for training. One of the registered pipelines present in trlx.pipeline
    :type pipeline: str

    :param orchestrator: Orchestrator to use for training. One of the registered orchestrators present in trlx.orchestrator
    :type orchestrator: str

    :param trainer: Trainer to use for training. One of the registered trainers present in trlx.trainer

    :param project_name: Project name for wandb
    :type project_name: str

    :param entity_name: Entity name for wandb
    :type entity_name: str

    :param checkpoint_dir: Directory to save checkpoints
    :type checkpoint_dir: str

    :param rollout_logging_dir: Directory to store generated rollouts for use in Algorithm Distillation.
                                Only used by AcceleratePPOTrainer.
    :type rollout_logging_dir: Optional[str]

    :param seed: Random seed
    :type seed: int
    """

    total_steps: int
    seq_length: int
    epochs: int
    batch_size: int

    checkpoint_interval: int
    eval_interval: int

    pipeline: str  # One of the pipelines in framework.pipeline
    orchestrator: str  # One of the orchestrators
    trainer: str  # One of the trainers

    project_name: str = "trlx"
    entity_name: Optional[str] = None

    checkpoint_dir: str = "ckpts"
    rollout_logging_dir: Optional[str] = None

    trackers: Tuple[str] = ("wandb",)
    seed: int = 1000

    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        return cls(**config)


@dataclass
class TRLConfig:
    """
    Top level config for trlX. Loads configs and can be converted to dictionary.
    """

    method: MethodConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    train: TrainConfig

    @classmethod
    def load_yaml(cls, yml_fp: str):
        """
        Load yaml file as TRLConfig.

        :param yml_fp: Path to yaml file
        :type yml_fp: str
        """
        with open(yml_fp, mode="r") as file:
            config = yaml.safe_load(file)
        return cls.from_dict(config)

    def to_dict(self):
        """
        Convert TRLConfig to dictionary.
        """
        data = {
            "method": self.method.__dict__,
            "model": self.model.__dict__,
            "optimizer": self.optimizer.__dict__,
            "scheduler": self.scheduler.__dict__,
            "train": self.train.__dict__,
        }

        return data

    @classmethod
    def from_dict(cls, config: Dict):
        """
        Convert dictionary to TRLConfig.
        """
        return cls(
            method=get_method(config["method"]["name"]).from_dict(config["method"]),
            model=ModelConfig.from_dict(config["model"]),
            optimizer=OptimizerConfig.from_dict(config["optimizer"]),
            scheduler=SchedulerConfig.from_dict(config["scheduler"]),
            train=TrainConfig.from_dict(config["train"]),
        )

    @classmethod
    def update(cls, baseconfig: Dict, config: Dict):
        updates = set()
        merged = merge(baseconfig, config, updates)

        for param in config:
            if param not in updates:
                raise ValueError(
                    f"parameter {param} is not present in the config (typo or a wrong config)"
                )

        return cls.from_dict(merged)

    def __str__(self):
        """Returns a human-readable string representation of the config."""
        import json

        return json.dumps(self.to_dict(), indent=4)
