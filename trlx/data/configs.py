import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

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

    :param device: Device to use when doing single GPU training. Not needed in most cases.
    :type device: str
    """
    model_path : str
    tokenizer_path : str
    model_type : str # One of the architectures present in framework.model
    device : str = ''
    num_layers_unfrozen : int = -1

    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        return cls(**config)


@dataclass
class TrainConfig:
    """
    Config for train job on model.

    :param n_ctx: Number of tokens to use as context (max length for tokenizer)
    :type n_ctx: int

    :param total_steps: Total number of training steps
    :type total_steps: int

    :param batch_size: Batch size for training
    :type batch_size: int

    :param grad_clip: Clip gradients to this valus
    :type grad_clip: float

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

    :param log_interval: Log training progress every log_interval steps
    :type log_interval: int

    :param checkpoint_interval: Save model every save_interval steps
    :type checkpoint_interval: int

    :param eval_interval: Evaluate model every eval_interval steps
    :type eval_interval: int

    :param pipeline: Pipeline to use for training. One of the registered pipelines present in trlx.pipeline
    :type pipeline: str

    :param orchestrator: Orchestrator to use for training. One of the registered orchestrators present in trlx.orchestrator
    :type orchestrator: str

    :param input_size: Max model input size in tokens
    :type input_size: int

    :param output_size: Max model output/generation size in tokens
    :type output_size: int

    :param project_name: Project name for wandb
    :type project_name: str
    """
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
    opt_betas: Tuple[float]

    log_interval: int
    checkpoint_interval: int
    eval_interval: int

    pipeline: str  # One of the pipelines in framework.pipeline
    orchestrator: str  # One of the orchestrators

    input_size: int = 0  # max model input size
    gen_size: int = 1024  # max size of model generation

    accelerate: bool = True  # Use HF accelerate?
    accelerate_config_path: str = ""

    checkpoint_dir: str = "ckpts"
    project_name: str = "trlx"

    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        return cls(**config)


@dataclass
class TRLConfig:
    """
    Top level config for trlX. Loads configs and can be converted to dictionary.
    """
    model : ModelConfig
    train : TrainConfig
    method : MethodConfig

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
