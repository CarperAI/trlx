from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import yaml

from trlx.data.method_configs import MethodConfig, get_method


def merge(base: Dict, update: Dict, updated: Set) -> Dict:
    "Recursively updates a nested dictionary with new values"
    for k, v in base.items():
        if k in update and isinstance(v, dict):
            base[k] = merge(v, update[k], updated)
            updated.add(k)
        elif k in update:
            base[k] = update[k]
            updated.add(k)

    return base


def _merge_dicts(base: Dict, update: Dict) -> Dict:
    "Merge two dictionaries recursively, returning a new dictionary."

    base = deepcopy(base)

    for k, v in update.items():
        if isinstance(v, dict):
            base[k] = _merge_dicts(base.get(k, {}), v)
        else:
            base[k] = v

    return base


@dataclass
class ModelConfig:
    """
    Config for a model.

    :param model_path: Path or name of the model (local or on huggingface hub)
    :type model_path: str

    :param model_arch_type: Type of model architecture. Either "causal" or "seq2seq"
    :type model_arch_type: str

    :param num_layers_unfrozen: Number of layers to unfreeze for fine-tuning.
        -1 means all layers are unfrozen.
    :type num_layers_unfrozen: int

    :param delta_kwargs: Keyword arguments for instantiating OpenDelta models for delta-tuning.
        Follow the `OpenDelta.AutoDeltaConfig` specification, e.g. for LoRA style tuning, set
        the `delta_type` to `lora` and include the model specific hyper-parameters (e.g. `lora_r`)
            {"delta_type": "lora", "modified_modules": "all", "lora_r": 8, "lora_alpha": 16, "lora_dropout": 0.0}
        or in YAML format:
            delta_kwargs:
                delta_type: lora
                modified_modules: "all"
                lora_r: 8
                lora_alpha: 16
                lora_dropout: 0.0
        See: https://opendelta.readthedocs.io/en/latest/modules/auto_delta.html#opendelta.auto_delta.AutoDeltaConfig
    :type delta_kwargs: Optional[Dict[str, Any]]
    """

    model_path: str
    model_arch_type: str = "causal"
    num_layers_unfrozen: int = -1
    delta_kwargs: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        return cls(**config)


@dataclass
class TokenizerConfig:
    """
    Config for a model.

    :param tokenizer_path: Path or name of the tokenizer (local or on huggingface hub)
    :type tokenizer_path: str

    :param padding_side: Padding side
    :type padding_path: str

    :param truncation_side: Truncation side
    :type truncation_side: str
    """

    tokenizer_path: str
    padding_side: str = "left"
    truncation_side: str = "right"

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

    :param tracker: Tracker to use for logging. Default: "wandb"
    :type tracker: str

    :param checkpoint_interval: Save model every checkpoint_interval steps.
        Each checkpoint is stored in a sub-directory of the `TrainConfig.checkpoint_dir`
        directory in the format `checkpoint_dir/checkpoint_{step}`.
    :type checkpoint_interval: int

    :param eval_interval: Evaluate model every eval_interval steps
    :type eval_interval: int

    :param pipeline: Pipeline to use for training. One of the registered pipelines present in trlx.pipeline
    :type pipeline: str

    :param trainer: Trainer to use for training. One of the registered trainers present in trlx.trainer
    :type trainer: str

    :param trainer_kwargs: Extra keyword arguments for the trainer
    :type trainer: Dict[str, Any]

    :param project_name: Project name for wandb
    :type project_name: str

    :param entity_name: Entity name for wandb
    :type entity_name: str

    :param group_name: Group name for wandb (used for grouping runs)
    :type group_name: str

    :param checkpoint_dir: Directory to save checkpoints
    :type checkpoint_dir: str

    :param rollout_logging_dir: Directory to store generated rollouts for use in Algorithm Distillation.
                                Only used by AcceleratePPOTrainer.
    :type rollout_logging_dir: Optional[str]

    :param save_best: Save best model based on mean reward
    :type save_best: bool

    :param seed: Random seed
    :type seed: int

    :param minibatch_size: Size of model input during one forward pass. Must divide batch size
    :type minibatch_size: int
    """

    total_steps: int
    seq_length: int
    epochs: int
    batch_size: int

    checkpoint_interval: int
    eval_interval: int

    pipeline: str  # One of the pipelines in framework.pipeline
    trainer: str  # One of the trainers
    trainer_kwargs: Dict[str, Any] = field(default_factory=dict)  # Extra keyword arguments for the trainer

    project_name: str = "trlx"
    entity_name: Optional[str] = None
    group_name: Optional[str] = None

    checkpoint_dir: str = "ckpts"
    rollout_logging_dir: Optional[str] = None
    save_best: bool = True
    save_optimizer: bool = True

    tracker: Optional[str] = "wandb"
    logging_dir: Optional[str] = None
    tags: Optional[List[str]] = field(default_factory=list)

    seed: int = 1000

    minibatch_size: Optional[int] = None

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
    tokenizer: TokenizerConfig
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
            "tokenizer": self.tokenizer.__dict__,
            "train": self.train.__dict__,
        }

        return data

    def evolve(self, **kwargs) -> "TRLConfig":
        """
        Evolve TRLConfig with new parameters. Can update nested parameters.
        >>> config = trlx.data.default_configs.default_ilql_config()
        >>> config = config.evolve(method=dict(gamma=0.99, gen_kwargs=dict(max_new_tokens=100))
        >>> config.method.gamma
        0.99
        """
        return TRLConfig.from_dict(_merge_dicts(self.to_dict(), kwargs))

    @classmethod
    def from_dict(cls, config: Dict):
        """
        Convert dictionary to TRLConfig.
        """
        return cls(
            method=get_method(config["method"]["name"]).from_dict(config["method"]),
            model=ModelConfig.from_dict(config["model"]),
            tokenizer=TokenizerConfig.from_dict(config["tokenizer"]),
            optimizer=OptimizerConfig.from_dict(config["optimizer"]),
            scheduler=SchedulerConfig.from_dict(config["scheduler"]),
            train=TrainConfig.from_dict(config["train"]),
        )

    @classmethod
    def update(cls, baseconfig: Dict, config: Dict):
        update = {}
        # unflatten a string variable name into a nested dictionary
        # key1.key2.key3: value -> {key1: {key2: {key3: value}}}
        for name, value in config.items():
            if isinstance(value, dict):
                update[name] = value
            else:
                *layers, var = name.split(".")
                if layers:
                    d = update.setdefault(layers[0], {})
                    for layer in layers[1:]:
                        d = d.setdefault(layer, {})
                    d[var] = value

        if not isinstance(baseconfig, Dict):
            baseconfig = baseconfig.to_dict()

        updates = set()
        merged = merge(baseconfig, update, updates)

        for param in update:
            if param not in updates:
                raise ValueError(f"parameter {param} is not present in the config (typo or a wrong config)")

        return cls.from_dict(merged)

    def __str__(self):
        """Returns a human-readable string representation of the config."""
        import json

        return json.dumps(self.to_dict(), indent=4)
