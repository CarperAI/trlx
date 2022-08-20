from abc import abstractmethod
from typing import Dict
import sys

from framework.pipeline import BasePipeline
from framework.model import *

# specifies a dictionary of architectures
_ORCH: Dict[str, any] = {}  # registry

def register_orchestrator(name):
    """Decorator used register a CARP architecture
    Args:
        name: Name of the architecture
    """

    def register_class(cls, name):
        _ORCH[name] = cls
        setattr(sys.modules[__name__], name, cls)
        return cls

    if isinstance(name, str):
        name = name.lower()
        return lambda c: register_class(c, name)

    cls = name
    name = cls.__name__
    register_class(cls, name.lower())

    return cls

@register_orchestrator
class Orchestrator:
    def __init__(self, pipeline : BasePipeline, rl_model : BaseRLModel):
        self.pipeline = pipeline
        self.rl_model = rl_model

    @abstractmethod
    def make_experience(self):
        """
        Draw from pipeline, get action, generate reward
        Push to models RolloutStorage
        """
        pass
