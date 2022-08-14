from abc import abstractmethod
from typing import Dict
import sys

from framework.data import RLElement
from framework.pipeline import BaseRolloutStore
from framework.configs import TRLConfig

# specifies a dictionary of architectures
_MODELS: Dict[str, any] = {}  # registry


def register_model(name):
    """Decorator used register a CARP architecture
    Args:
        name: Name of the architecture
    """

    def register_class(cls, name):
        _MODELS[name] = cls
        setattr(sys.modules[__name__], name, cls)
        return cls

    if isinstance(name, str):
        name = name.lower()
        return lambda c: register_class(c, name)

    cls = name
    name = cls.__name__
    register_class(cls, name.lower())

    return cls

@register_model
class BaseRLModel:
    def __init__(self, config : TRLConfig, train_mode = False):
        self.store : BaseRolloutStore = None
        self.config = config
        self.train_mode = train_mode

    def push_to_store(self, data):
        self.store.push(data)
    
    @abstractmethod
    def act(data : RLElement) -> RLElement:
        """
        Given RLElement with state, produce an action and add it to the RLElement.
        Orchestrator should call this, get reward and push subsequent RLElement to RolloutStore
        """
        pass
    
    @abstractmethod
    def learn():
        """
        Use experiences in RolloutStore to learn
        """
        pass

    @abstractmethod
    def save():
        """
        Save checkpoint. Whether or not optimizer/scheduler is saved depends on train_mode
        """
        pass

    @abstractmethod
    def load():
        """
        Load checkpoint. Whether or not optimizer/scheduler is loaded depends on train_mode
        """
