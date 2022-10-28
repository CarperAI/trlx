from dataclasses import dataclass
from typing import Any, Dict, Tuple

import yaml

from trlx.data.method_configs import MethodConfig, get_method


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

    :param lr_ramp_steps: Number of steps before learning rate reaches learning_rate_init
    :type lr_ramp_steps: int

    :param lr_decay_steps: Number of after ramp up steps before learning rate decays to learning_rate_target
    :type lr_decay_steps: int

    :param weight_decay: Weight decay for optimizer
    :type weight_decay: float

    :param learning_rate_init: Initial learning rate after ramp up
    :type learning_rate_init: float

    :param learning_rate_target: Target learning rate after decay
    :type learning_rate_target: float

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
    """

    total_steps: int
    seq_length: int
    epochs: int
    batch_size: int

    lr_ramp_steps: int
    lr_decay_steps: int
    weight_decay: float
    learning_rate_init: float
    learning_rate_target: float
    opt_betas: Tuple[float]

    checkpoint_interval: int
    eval_interval: int

    pipeline: str  # One of the pipelines in framework.pipeline
    orchestrator: str  # One of the orchestrators

    checkpoint_dir: str = "ckpts"
    project_name: str = "trlx"
    seed: int = 1000

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
        data = self.model.__dict__.copy()
        data.update(self.train.__dict__)
        data.update(self.method.__dict__)
        return data
