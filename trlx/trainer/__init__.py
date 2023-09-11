import sys
from abc import abstractmethod
from typing import Any, Callable, Dict, Iterable, Optional

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
    def __init__(
        self,
        config: TRLConfig,
        reward_fn=None,
        metric_fn=None,
        stop_sequences=None,
        train_mode=False,
    ):
        self.store: BaseRolloutStore = None
        self.config = config
        self.reward_fn = reward_fn
        self.metric_fn = metric_fn
        self.train_mode = train_mode
        self.stop_sequences = stop_sequences

    def push_to_store(self, data):
        """
        Append new data to the rollout store
        """
        self.store.push(data)

    @abstractmethod
    def learn(self):
        """
        Use data in the the rollout store to update the model
        """
        pass

