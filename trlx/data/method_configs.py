import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

# specifies a dictionary of method configs
_METHODS: Dict[str, any] = {}  # registry


def register_method(name):
    """Decorator used register a method config
    Args:
        name: Name of the method
    """

    def register_class(cls, name):
        _METHODS[name] = cls
        setattr(sys.modules[__name__], name, cls)
        return cls

    if isinstance(name, str):
        name = name.lower()
        return lambda c: register_class(c, name)

    cls = name
    name = cls.__name__
    register_class(cls, name.lower())

    return cls


def get_method(name: str) -> Callable:
    """
    Return constructor for specified method config
    """
    name = name.lower()
    if name in _METHODS:
        return _METHODS[name]
    else:
        raise Exception("Error: Trying to access a method that has not been registered")


@dataclass
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


@dataclass
@register_method
class PPOConfig(MethodConfig):
    """
    Config for PPO method

    :param ppo_epochs: Number of updates per batch
    :type ppo_epochs: int

    :param num_rollouts: Number  of experiences to observe before learning
    :type num_rollouts: int

    :param init_kl_coef: Initial value for KL coefficient
    :type init_kl_coef: float

    :param target: Target value for KL coefficient
    :type target: float

    :param horizon: Number of steps for KL coefficient to reach target
    :type horizon: int

    :param gamma: Discount factor
    :type gamma: float

    :param lam: GAE lambda
    :type lam: float

    :param cliprange: Clipping range for PPO policy loss (1 - cliprange, 1 + cliprange)
    :type cliprange: float

    :param cliprange_value: Clipping range for predicted values (observed values - cliprange_value, observed values + cliprange_value)
    :type cliprange_value: float

    :param vf_coef: Value loss scale w.r.t policy loss
    :type vf_coef: float

    :param gen_kwargs: Additioanl kwargs for the generation
    :type gen_kwargs: Dict[str, Any]
    """
    ppo_epochs: int
    num_rollouts: int
    chunk_size: int
    init_kl_coef: float
    target: float
    horizon: int
    gamma: float
    lam: float
    cliprange: float
    cliprange_value: float
    vf_coef: float
    gen_kwargs: dict


@dataclass
@register_method
class ILQLConfig(MethodConfig):
    """
    Config for ILQL method

    :param tau: Control tradeoff in value loss between punishing value network for underestimating the target Q (i.e. Q value corresponding to the action taken) (high tau) and overestimating the target Q (low tau)
    :type tau: float

    :param gamma: Discount factor for future rewards
    :type gamma: float

    :param cql_scale: Weight for CQL loss term
    :type cql_scale: float

    :param awac_scale: Weight for AWAC loss term
    :type awac_scale: float

    :param steps_for_target_q_sync: Number of steps to wait before syncing target Q network with Q network
    :type steps_for_target_q_sync: int

    :param two_qs: Use minimum of two Q-value estimates
    :type two_qs: bool
    """
    tau: float
    gamma: float
    cql_scale: float
    awac_scale: float
    alpha: float
    steps_for_target_q_sync: int
    betas: List[float]
    two_qs: bool
