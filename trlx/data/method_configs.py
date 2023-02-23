from typing import Any, Dict

from attrs import asdict, define
from cattrs import register_structure_hook, register_unstructure_hook, unstructure

# specifies a dictionary of method configs
_METHODS: Dict[str, Any] = {}  # registry


def register_method(cls):
    """Decorator used register a method config
    Args:
        name: Name of the method
    """
    _METHODS[cls.__name__.lower()] = cls
    return cls


@define
@register_method
class MethodConfig:
    """
    Config for a certain RL method.

    :param name: Name of the method
    :type name: str
    """

    name: str

    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        return cls(**config)


def get_method(name: str) -> MethodConfig:
    """
    Return constructor for specified method config
    """
    name = name.lower()
    if name in _METHODS:
        return _METHODS[name]
    else:
        raise Exception("Error: Trying to access a method that has not been registered")


register_structure_hook(MethodConfig, lambda obj, _: get_method(obj["name"])(**obj))
register_unstructure_hook(MethodConfig, lambda obj: {**asdict(obj), "name": obj.__class__.__name__})
