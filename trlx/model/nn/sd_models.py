from dataclasses import dataclass
from typing import Union, Tuple, Optional, Iterable
from torchtyping import TensorType

import torch
from torch import nn
from transformers.modeling_outputs import ModelOutput
import diffusers
from diffusers import UNet2DConditionModel
from diffusers.models.unet_2d_condition import UNet2DConditionOutput

from trlx.data.method_configs import MethodConfig, register_method
from trlx.model.nn.ppo_models import PPOConfig

@dataclass 
@register_method
class DiffPPOConfig(PPOConfig):
    """
    Diffusion PPO config
    """
    img_size: Tuple[int, int]
    num_channels: int

@dataclass
class UNet2DConditionOutputHidden(UNet2DConditionOutput):
    hidden: torch.Tensor = None

@dataclass
class UNetActorCriticOutput(ModelOutput):
    """
    Output of UNetActorCritic
    """
    means: TensorType["batch", "num_channels", "height", "width"]
    values: TensorType["batch"]

class UNet2DConditionModelHidden(UNet2DConditionModel):
    """
    LDM Unet 2d conditional that also outputs middle block hidden state. Refer to diffusers documentation for more information.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        return_hidden_states: bool = False
    ) -> Union[UNet2DConditionOutput, Tuple]:
        """
        Refer to diffusers documentation for more information
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            #logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        if self.config.num_class_embeds is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")
            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        sample = self.mid_block(sample, emb, encoder_hidden_states=encoder_hidden_states)
        hidden = sample

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )
        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample, hidden)

        return UNet2DConditionOutputHidden(sample=sample, hidden = hidden)

def convert_to_hidden_model(model : UNet2DConditionModel) -> UNet2DConditionModelHidden:
    """
    Converts a UNet2DConditionModel to a UNet2DConditionModelHidden.
    """
    model.__class__ = UNet2DConditionModelHidden
    return model

class UNetActorCritic(nn.Module):
    def __init__(self, vae, model):
        super().__init__()

        self.model = convert_to_hidden_model(model)

        # Figure out hidden size to make value head
        with torch.no_grad() and torch.autocast('cuda')

        # Infer size of middle block hidden state rather than hardcoding it
        #size = cfg.sample_size * vae_scale_factor   
        #inp = torch.zeros(1, cfg.in_channels, size, size)
        #enc_hidden = torch.zeros(1, 1, 512)
        #with torch.no_grad() and torch.autocast('cuda'):
        #    out, hidden = self.model(inp, 0, enc_hidden)
        #    print(hidden.shape)

        #self.value_head()

    def forward(
        self,
        sample: TensorType["batch", "channel", "height", "width"],
        timestep: Union[TensorType["batch"], float, int],
        encoder_hidden_state: TensorType["batch", "seq_len", "d_model"]
    ) -> UNetActorCriticOutput:
        """
        :param sample: Input noise
        :param timestep: Current timestep
        """

        pass

