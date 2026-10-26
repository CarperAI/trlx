"""CPU checks for NaN-safe padding in PPO log-probability ratios."""

import pytest
import torch

from trlx.models.modeling_ppo import PPOConfig


def _ppo_config():
    config = PPOConfig.__new__(PPOConfig)
    config.cliprange = 0.2
    config.cliprange_value = 0.2
    config.vf_coef = 0.1
    return config


@pytest.mark.parametrize("padding_case", ["negative_infinity", "nan"])
def test_nonfinite_padding_does_not_poison_ppo_loss(padding_case):
    old = torch.tensor([[-1.0, -1.0, -1.0]])
    current = old.clone()
    if padding_case == "negative_infinity":
        old[0, 0] = float("-inf")
        current[0, 0] = float("-inf")
    else:
        current[0, 0] = float("nan")
    current.requires_grad_()

    mask = torch.tensor([[0.0, 1.0, 1.0]])
    values = torch.zeros_like(old)
    advantages = torch.ones_like(old)
    loss, _ = _ppo_config().loss(current, values, old, values, advantages, values, mask)
    loss.backward()

    torch.testing.assert_close(loss, torch.tensor(-1.0))
    torch.testing.assert_close(current.grad, torch.tensor([[0.0, -0.5, -0.5]]))
    assert torch.isfinite(current.grad).all()


def test_finite_policy_ratio_unchanged():
    old = torch.tensor([[-1.0, -1.0]])
    current = old.clone().requires_grad_(True)
    mask = torch.ones_like(old)
    values = torch.zeros_like(old)
    loss, _ = _ppo_config().loss(
        current, values, old, values, torch.ones_like(old), values, mask
    )
    loss.backward()
    torch.testing.assert_close(loss, torch.tensor(-1.0))
    torch.testing.assert_close(current.grad, torch.tensor([[-0.5, -0.5]]))
