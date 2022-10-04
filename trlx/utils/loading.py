from typing import Callable

from trlx.pipeline import _DATAPIPELINE
from trlx.orchestrator import _ORCH
from trlx.model import _MODELS

def get_model(name : str) -> Callable:
    """
    Return constructor for specified model
    """
    name = name.lower()
    if name in _MODELS:
        return _MODELS[name]
    else:
        raise Exception("Error: Trying to access a model that has not been registered")

def get_pipeline(name : str) -> Callable:
    """
    Return constructor for specified pipeline
    """
    name = name.lower()
    if name in _DATAPIPELINE:
        return _DATAPIPELINE[name]
    else:
        raise Exception("Error: Trying to access a pipeline that has not been registered")

def get_orchestrator(name : str) -> Callable:
    """
    Return constructor for specified orchestrator
    """
    name = name.lower()
    if name in _ORCH:
        return _ORCH[name]
    else:
        raise Exception("Error: Trying to access an orchestrator that has not been registered")