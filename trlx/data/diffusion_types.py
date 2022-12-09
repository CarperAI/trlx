from dataclasses import dataclass

from torchtyping import TensorType

@dataclas
class DiffPPORLElement:
    """
    RL Element for Diffusion PPO

    :param prompt: Text prompt for image generation
    :param attention_mask: Attention mask for the text prompt
    :param img: latent for images in sampling steps
    :param log_probs: log probabilities of the actions taken (the noise predicted)
    """

    prompt : TensorType["num_tokens"]
    attention_mask : TensorType["num_tokens"]
    img : TensorType["sample_steps", 4, "H/8", "W/8"]
    logprobs : TensorType["sample_steps"]
    values : TensorType["sample_steps"]
    rewards : TensorType["sample_steps"]

@dataclass
class DiffPPORLBatch:
    """
    Batched RL Element for Diffusion PPO

    :param prompt: Text prompt for image generation
    :param attention_mask: Attention mask for the text prompt
    :param img: latent for images in sampling steps
    :param log_probs: log probabilities of the actions taken (the noise predicted)
    """

    prompt : TensorType["batch_size", "num_tokens"]
    attention_mask : TensorType["batch_size", "num_tokens"]
    img : TensorType["batch_size", "sample_steps", 4, "H/f", "W/f"]
    logprobs : TensorType["batch_size", "sample_steps"]
    values : TensorType["batch_size", "sample_steps"]
    rewards : TensorType["batch_size", "sample_steps"]

    
