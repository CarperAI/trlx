import os
import sys
from abc import abstractmethod
from typing import Any, Callable, Dict, Iterable

import torch

from trlx.data import RLElement
from trlx.data.configs import TRLConfig
from trlx.pipeline import BaseRolloutStore

# specifies a dictionary of architectures
_TRAINERS: Dict[str, Any] = {}  # registry


def register_trainer(name):
    """Decorator used to register a trainer
    Args:
        name: Name of the trainer type to register
    """

    def register_class(cls, name):
        _TRAINERS[name] = cls
        setattr(sys.modules[__name__], name, cls)
        return cls

    if isinstance(name, str):
        name = name.lower()
        return lambda c: register_class(c, name)

    cls = name
    name = cls.__name__
    register_class(cls, name.lower())

    return cls


@register_trainer
class BaseRLTrainer:
    def __init__(self, config: TRLConfig, train_mode=False):
        self.store: BaseRolloutStore = None
        self.config = config
        self.train_mode = train_mode

    def push_to_store(self, data):
        self.store.push(data)

    def add_eval_pipeline(self, eval_pipeline):
        """Adds pipeline from with validation prompts"""
        self.eval_pipeline = eval_pipeline

    @abstractmethod
    def act(self, data: RLElement) -> RLElement:
        """
        Given RLElement with state, produce an action and add it to the RLElement.
        Orchestrator should call this, get reward and push subsequent RLElement to RolloutStore
        """
        pass

    @abstractmethod
    def sample(
        self, prompts: Iterable[str], length: int, n_samples: int
    ) -> Iterable[str]:
        """
        Sample from the language. Takes prompts and maximum length to generate.

        :param prompts: List of prompts to tokenize and use as context

        :param length: How many new tokens to genrate for each prompt
        :type length: int

        :param n_samples: Default behavior is to take number of prompts as this
        """
        pass

    @abstractmethod
    def learn(
        self,
        log_fn: Callable = None,
        save_fn: Callable = None,
        eval_fn: Callable = None,
    ):
        """
        Use experiences in RolloutStore to learn

        :param log_fn: Optional function that is called when logging and passed a dict of logging relevant values
        :type log_fn: Callable[Dict[str, any]]

        :param save_fn: Optional function to call after saving. Is passed the components.
        :type save_fn: Callable[Dict[str, any]]

        :param eval_fn: Optional function to call during evaluation. Eval doesn't do anything without this.
        :type eval_fn: Callable[BaseRLTrainer]
        """
        pass

    @abstractmethod
    def save(self, directory=None):
        """Creates a checkpoint of training states"""
        pass

    @abstractmethod
    def load(self, directory=None):
        """Loads a checkpoint created from `save`"""
        pass

    def intervals(self, steps: int) -> Dict[str, bool]:
        """
        Using config and current step number, returns a dict of whether certain things should be done
        """

        return {
            "do_log": (steps + 1) % self.config.train.log_interval == 0,
            "do_eval": (steps + 1) % self.config.train.eval_interval == 0,
            "do_save": (steps + 1) % self.config.train.checkpoint_interval == 0,
        }
