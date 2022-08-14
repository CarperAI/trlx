from functools import reduce

from typing import Iterable, List, Any, Callable

def flatten(L : Iterable[Iterable[Any]]) -> Iterable[Any]:
    """
    Flatten a list of lists into a single list (i.e. [[1, 2], [3, 4]] -> [1,2,3,4])
    """
    return list(reduce(lambda acc, x: acc + x, L, []))

def chunk(L : Iterable[Any], chunk_size : int) -> List[Iterable[Any]]:
    """
    Chunk iterable into list of iterables of given chunk size
    """
    return [L[i:i+chunk_size] for i in range(0, len(L), chunk_size)]

# For loading things

from framework.pipeline import _DATAPIPELINE
from framework.orchestrator import _ORCH
from framework.model import _MODELS

def get_model(name : str) -> Callable:
    """
    Return constructor for specified model
    """
    if name in _MODELS:
        return _MODELS[name]
    else:
        raise Exception("Error: Trying to access a model that has not been registered")

def get_pipeline(name : str) -> Callable:
    """
    Return constructor for specified pipeline
    """
    if name in _DATAPIPELINE:
        return _DATAPIPELINE[name]
    else:
        raise Exception("Error: Trying to access a pipeline that has not been registered")

def get_orchestrator(name : str) -> Callable:
    """
    Return constructor for specified orchestrator
    """
    if name in _ORCH:
        return _ORCH[name]
    else:
        raise Exception("Error: Trying to access an orchestrator that has not been registered")