import yaml
from ray import tune


# specifies a dictionary of functions to sweep over
_SWEEPS: Dict[str, any] = {}  # registry

def register_sweep(name):
    """Decorator used register a sweep function.
    Args:
        name: Name of the sweep
    """

    def register_class(cls, name):
        _SWEEPS[name] = cls
        setattr(sys.modules[__name__], name, cls)
        return cls

    if isinstance(name, str):
        name = name.lower()
        return lambda c: register_class(c, name)

    cls = name
    name = cls.__name__
    register_class(cls, name.lower())

    return cls
