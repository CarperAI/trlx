import torch
import torch.nn.functional as F


def whiten(values, shift_mean=True):
    """Whiten values."""
    mean, var = torch.mean(values), torch.var(values)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def clip_by_value(x, tensor_min, tensor_max):
    """
    Tensor extenstion to torch.clamp
    https://github.com/pytorch/pytorch/issues/2793#issuecomment-428784713
    """
    clipped = torch.max(torch.min(x, tensor_max), tensor_min)
    return clipped


def logprobs_from_logits(logits, labels):
    """
    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
    """
    logp = F.log_softmax(logits, dim=2)
    logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logpy
