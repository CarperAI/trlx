import gc
import inspect
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import deepspeed
import numpy as np
import torch
import transformers
from torchtyping import TensorType
from transformers.modeling_outputs import ModelOutput
from transformers.models.bloom import modeling_bloom
from transformers.models.opt import modeling_opt

from trlx.data.method_configs import MethodConfig, register_method
from trlx.models.modeling_base import PreTrainedModelWrapper
from trlx.utils.modeling import (
    flatten_dict,
    get_tensor_stats,
    hf_get_decoder,
    hf_get_decoder_blocks,
    hf_get_decoder_final_norm,
    hf_get_hidden_size,
    hf_get_lm_head,
    hf_get_num_hidden_layers,
    make_head,
    whiten,
)

# KL Controllers


class AdaptiveKLController:
    """Adaptive KL Controller as described in Ziegler et al. "Fine-Tuning Language Models from Human Preferences"
    Reference: Section 2.2 https://arxiv.org/pdf/1909.08593.pdf#page=2
    Source: https://github.com/openai/lm-human-preferences/blob/master/lm_human_preferences/train_policy.py
    """

    def __init__(self, init_kl_coef: float, target: float, horizon: int):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current: float, n_steps: int):
        """Returns adaptively updated KL coefficient, βₜ₊₁.
        Arguments:
            current: The current KL value between the newest policy and the initial policy.
        """
        proportional_error = np.clip(current / self.target - 1, -0.2, 0.2)  # ϵₜ
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult  # βₜ₊₁


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current: float, n_steps: int):
        """Returns updated KL coefficient, βₜ₊₁.
        Arguments:
            current: The current KL value between the newest policy and the initial policy.
        """
        pass


# PPO Configs


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

    :param cliprange_value: Clipping range for predicted values
                            (observed values - cliprange_value, observed values + cliprange_value)
    :type cliprange_value: float

    :param vf_coef: Value loss scale w.r.t policy loss
    :type vf_coef: float

    :param gen_kwargs: Additional kwargs for the generation
    :type gen_kwargs: Dict[str, Any]

    :param gen_experience_kwargs: if this is not None, then the experience is generated using this
    :type gen_experience_kwargs: Dict[str, Any]
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
    scale_reward: Optional[str]
    ref_mean: Optional[float]
    ref_std: Optional[float]
    cliprange_reward: float
    gen_kwargs: dict
    gen_experience_kwargs: Optional[dict] = None
    num_value_layers_unfrozen: int = 0

    def get_advantages_and_returns(
        self,
        values: TensorType["batch_size", "response_size"],
        rewards: TensorType["batch_size", "response_size"],
        response_length: int,
        use_whitening: Optional[bool] = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Args:
            values: Tensor of shape (batch_size, response_size)
            rewards: Tensor of shape (batch_size, response_size)
            response_length: Length of the response sequence
            use_whitening: Whether to use whitening (ie. normalize advantages) or not
        """
        lastgaelam = 0
        advantages_reversed = []

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        if use_whitening:
            advantages = whiten(advantages)
        return advantages.detach(), returns

    def loss(
        self,
        logprobs: TensorType["batch_size", "response_size"],
        values: TensorType["batch_size", "response_size"],
        old_logprobs: TensorType["batch_size", "response_size"],
        old_values: TensorType["batch_size", "response_size"],
        advantages: TensorType["batch_size", "response_size"],
        returns: TensorType["batch_size", "response_size"],
        mask: TensorType["batch_size", "response_size"],
    ):
        """PPO objective function.
        References:
        - https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html
        """
        values_clipped = torch.clamp(
            values,
            old_values - self.cliprange_value,
            old_values + self.cliprange_value,
        )
        n = mask.sum()

        vf_loss1 = (values - returns) ** 2
        vf_loss2 = (values_clipped - returns) ** 2
        vf_loss = 0.5 * torch.sum(torch.max(vf_loss1, vf_loss2) * mask) / n

        vf_clipfrac = torch.sum((vf_loss2 > vf_loss1).float() * mask) / n

        log_ratio = (logprobs - old_logprobs) * mask
        ratio = torch.exp(log_ratio)
        # Unbiased KL-div estimates (`k3`). Ref: http://joschu.net/blog/kl-approx.html
        with torch.no_grad():
            approx_kl = torch.mean((ratio - 1) - log_ratio)

        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(
            ratio,
            1.0 - self.cliprange,
            1.0 + self.cliprange,
        )
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / n
        pg_clipfrac = torch.sum((pg_loss2 > pg_loss1).float() * mask) / n

        loss = pg_loss + self.vf_coef * vf_loss

        stats = dict(
            losses=dict(
                total_loss=loss.item(),
                policy_loss=pg_loss.item(),
                value_loss=vf_loss.item(),
            ),
            values=dict(
                get_tensor_stats(values, mask, n),
                values_error=torch.sum(((values - returns) * mask) ** 2) / n,
                values_mape_error=torch.sum((abs(values - returns) * mask) / abs(returns * mask + 1e-2)) / n,
                clipfrac=vf_clipfrac,
            ),
            old_values=get_tensor_stats(old_values, mask, n),
            returns=get_tensor_stats(returns, mask, n),
            policy=dict(approx_kl=approx_kl.item(), clipfrac=pg_clipfrac.item()),
            ratio=(ratio * mask).sum() / n,
            padding_percentage=1 - n / mask.numel(),
        )

        return loss, flatten_dict(stats)


# CausalLM architectures


@dataclass
class CausalLMOutputWithValue(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    value: Optional[torch.FloatTensor] = None


def make_value_branch(base_model, num_value_layers_unfrozen):
    value_head = make_head(hf_get_hidden_size(base_model.config), 1)
    if num_value_layers_unfrozen == 0:
        return value_head
    config = base_model.config
    branch_class = hf_get_branch_class(config)
    value_branch = branch_class(base_model, num_layers_unfrozen=num_value_layers_unfrozen, frozen=False)
    value_branch.lm_head = value_head
    return value_branch


class AutoModelForCausalLMWithValueHead(PreTrainedModelWrapper):
    """An `AutoModel` class wrapper for `transformers` causal models that have a
    language modeling head and a value head
    """

    _auto_model_parent_class = transformers.AutoModelForCausalLM
    _supported_modules = ["v_head"]
    _supported_args = ["peft_config", "num_value_layers_unfrozen"]

    def __init__(
        self,
        base_model: transformers.PreTrainedModel,
        peft_config=None,
        num_value_layers_unfrozen=0,
    ):
        super().__init__(base_model, peft_config=peft_config)
        self.num_value_layers_unfrozen = num_value_layers_unfrozen
        self.v_head = make_value_branch(base_model, num_value_layers_unfrozen)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        position_ids: Optional[List[torch.FloatTensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ignore_peft_adapter: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithValue]:
        forward_kwargs = self.get_compatible_forward_kwargs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        forward_kwargs["output_hidden_states"] = True
        forward_kwargs["return_dict"] = True

        if self.peft_type == "PREFIX_TUNING":
            # In this case peft redefines past_key_values, remove it to avoid an exception.
            forward_kwargs.pop("past_key_values", None)

        if self.peft_type and ignore_peft_adapter:
            if "LORA" in self.peft_type:
                # For LORA, temporarily disable the adapter
                lora_model = self.base_model.base_model
                lora_model.disable_adapter_layers()
                outputs = self.base_model(**forward_kwargs)
                lora_model.enable_adapter_layers()
            else:
                # For prompt or prefix adapters, just use the base model of PeftModel
                outputs = self.base_model.base_model(**forward_kwargs)
        else:
            outputs = self.base_model(**forward_kwargs)

        # TODO: Apply PEFT to value branch
        if self.num_value_layers_unfrozen > 0:
            output_shape = outputs.hidden_states[-1].size()
            forward_kwargs.pop("input_ids", None)
            forward_kwargs.pop("inputs_embeds", None)
            forward_kwargs["return_dict"] = False
            value = self.v_head(
                outputs.hidden_states[-(self.num_value_layers_unfrozen + 1)],
                output_shape=output_shape,
                **forward_kwargs,
            )[0].squeeze(-1)
        else:
            value = self.v_head(outputs.hidden_states[-(self.num_value_layers_unfrozen + 1)]).squeeze(-1)

        if not return_dict:
            outputs = (outputs.logits,) + outputs[1:] + (value,)
            return outputs

        return CausalLMOutputWithValue(**outputs, value=value)

    def generate(self, *args, **kwargs) -> Union[ModelOutput, torch.LongTensor]:
        return self.base_model.generate(*args, **kwargs)

    def state_dict(self, *args, heads_only=False, **kwargs):
        """
        Returns the state dictionary of the model. We add the state dictionary of the value head
        to the state dictionary of the wrapped model by prepending the key with `v_head.`.
        """
        state_dict = self.v_head.state_dict(*args, **dict(prefix="v_head.", **kwargs))
        if not heads_only:
            state_dict = {**state_dict, **self.base_model.state_dict(*args, **dict(prefix="base_model.", **kwargs))}

        return {
            **self.base_model.state_dict(*args, **dict(prefix="base_model.", **kwargs)),
            **self.v_head.state_dict(*args, **dict(prefix="v_head.", **kwargs)),
        }

        return state_dict

    def post_init(self, state_dict):
        """
        Adds the state dictionary of the value head to the state dictionary of the wrapped model
        by prepending the key with `v_head.`. This function removes the `v_head.` prefix from the
        keys of the value head state dictionary.
        """
        super().post_init()

        trlx_checkpoint = any(k.startswith("base_model.") or k.startswith("v_head.") for k in state_dict)
        self.load_state_dict(state_dict, strict=trlx_checkpoint)

        del state_dict
        gc.collect()  # noqa: E702


class AutoModelForCausalLMWithHydraValueHead(AutoModelForCausalLMWithValueHead):
    _supported_modules = ["v_head", "frozen_head"]
    _supported_args = ["num_layers_unfrozen", "peft_config", "num_value_layers_unfrozen"]

    def __init__(
        self,
        base_model: transformers.PreTrainedModel,
        *,
        num_layers_unfrozen: int = -1,
        peft_config=None,
        num_value_layers_unfrozen: int = 0,
    ):
        super().__init__(base_model, peft_config=peft_config, num_value_layers_unfrozen=num_value_layers_unfrozen)
        self.num_layers_unfrozen = num_layers_unfrozen

        if self.num_layers_unfrozen > 0 and not self.peft_type:
            config = self.base_model.config
            branch_class = hf_get_branch_class(config)
            self.frozen_head = branch_class(
                self.base_model,
                num_layers_unfrozen=self.num_layers_unfrozen,
            ).eval()
        else:
            self.frozen_head = None

    def forward_hydra(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        position_ids: Optional[List[torch.FloatTensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[torch.FloatTensor, CausalLMOutputWithValue]:
        forward_kwargs = self.get_compatible_forward_kwargs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return_dict = forward_kwargs.get("return_dict", True)
        forward_kwargs["return_dict"] = True
        forward_kwargs["output_hidden_states"] = True

        if self.peft_type:
            hydra_outputs = self.forward(**forward_kwargs, ignore_peft_adapter=True)
        else:
            outputs = self.forward(**forward_kwargs)
            # Select the hidden state before the first branching layer
            input_hidden_state = outputs.hidden_states[-(self.num_layers_unfrozen + 1)]

            output_shape = outputs.hidden_states[-1].size()
            forward_kwargs.pop("input_ids", None)  # Ignore `input_ids` for branch head
            forward_kwargs.pop("inputs_embeds", None)  # Ignore `inputs_embeds` for branch head
            hydra_outputs = self.frozen_head(input_hidden_state, output_shape, **forward_kwargs)

        if not return_dict:
            return hydra_outputs.logits
        return hydra_outputs

    def state_dict(self, *args, heads_only=False, **kwargs):
        """
        Returns the state dictionary of the model. We add the state dictionary of the value head
        to the state dictionary of the wrapped model by prepending the key with `v_head.`.
        """
        state_dict = self.v_head.state_dict(*args, **dict(prefix="v_head.", **kwargs))
        if not heads_only:
            state_dict = {
                **state_dict,
                **self.base_model.state_dict(*args, **dict(prefix="" if self.peft_type else "base_model.", **kwargs)),
            }

            if self.frozen_head:
                state_dict = {
                    **state_dict,
                    **self.frozen_head.state_dict(*args, **dict(prefix="frozen_head.", **kwargs)),
                }

        return state_dict

    def post_init(self, state_dict):
        """
        Load `state_dict` into the model. If peft was used to train the model,
        only the value head would be present in the loaded `state_dict`, so the
        loading has to be not strict. Also `frozen_head` will be recreated and
        loaded from the checkpoint, to comply with deepspeed checkpoint loading.
        """
        strict = not self.peft_type and any(k.startswith("base_model.") or k.startswith("v_head.") for k in state_dict)

        if not self.peft_type and self.frozen_head is None:
            for k in state_dict:
                match = re.search(r"^frozen_head\..+\.(\d+)\.", k)
                if match:
                    self.num_layers_unfrozen = max(self.num_layers_unfrozen, int(match.group(1)) + 1)

            config = self.base_model.config
            branch_class = hf_get_branch_class(config)
            self.frozen_head = branch_class(
                self.base_model,
                num_layers_unfrozen=self.num_layers_unfrozen,
            ).eval()

        self.load_state_dict(state_dict, strict=strict)
        del state_dict
        gc.collect()  # noqa: E702


class ModelBranch(transformers.PreTrainedModel):
    """Implements the upper trunk of the pretrained reference model used
    when computing the PPO KL-divergence penalty.
    """

    def __init__(
        self,
        base_model: transformers.PreTrainedModel,
        *,
        num_layers_unfrozen: int,
        frozen=True,
    ):
        """
        Args:
            base_model (transformers.PreTrainedModel): The pretrained model to extract upper trunk from
            num_layers_unfrozen (int): The number of trainable layers
        """
        super().__init__(base_model.config)

        # The branch is defined by the last `num_layers_unfrozen` layers of the pretrained model

        decoder_blocks = hf_get_decoder_blocks(base_model)[-num_layers_unfrozen:]
        final_norm = hf_get_decoder_final_norm(base_model)
        lm_head = hf_get_lm_head(base_model)

        with deepspeed.zero.GatheredParameters(
            list(decoder_blocks.parameters()) + list(final_norm.parameters()) + list(lm_head.parameters()),
            modifier_rank=None,
        ):
            self.decoder_blocks = deepcopy(decoder_blocks)
            self.final_norm = deepcopy(final_norm)
            self.lm_head = deepcopy(lm_head)

        self.hidden_size = hf_get_hidden_size(self.config)
        self.model_parallel = False
        self.device_map = None
        self.last_device = None
        self.gradient_checkpointing = False

        # Freeze the entire branch
        if frozen:
            for parameter in self.parameters():
                parameter.requires_grad_(False)


class GPTModelBranch(ModelBranch):
    def forward(  # noqa: max-complexity
        self,
        hidden_states: torch.Tensor,  # Takes as input hidden_states instead of input_ids
        output_shape: torch.Tensor,  # output_size given by main trunk
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = False,
    ) -> Union[Tuple, CausalLMOutputWithValue]:
        """Reference:
        https://github.com/huggingface/transformers/blob/2411f0e465e761790879e605a4256f3d4afb7f82/src/transformers/models/gpt2/modeling_gpt2.py#L743  # noqa: E501
        """
        batch_size, seq_length = hidden_states.shape[:2]

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        device = hidden_states.device

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.decoder_blocks))
        else:
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            position_ids = torch.arange(past_length, seq_length + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length)

        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        if self.config.add_cross_attention and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        head_mask = self.get_head_mask(head_mask, hf_get_num_hidden_layers(self.config))

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.decoder_blocks, past_key_values)):
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            kwargs = dict(
                layer_past=layer_past,
                attention_mask=attention_mask,
                position_ids=position_ids,
                head_mask=head_mask[i],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            # Assumes we are never training the branch
            block_params = inspect.getfullargspec(block.forward).args
            if "encoder_hidden_states" not in block_params:
                kwargs.pop("encoder_hidden_states")
                kwargs.pop("encoder_attention_mask")
            # Remove position_ids for GPT2Block
            if "position_ids" not in block_params:
                kwargs.pop("position_ids")

            outputs = block(hidden_states, **kwargs)

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_norm(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            outputs = (lm_logits,) + (None,) + (None,)
            return outputs

        return CausalLMOutputWithValue(
            logits=lm_logits,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class OPTModelBranch(ModelBranch):
    def forward(  # noqa: max-complexity
        self,
        hidden_states: torch.Tensor,
        output_shape: torch.Tensor,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = False,
    ) -> Union[Tuple, CausalLMOutputWithValue]:
        """Reference:
        https://github.com/huggingface/transformers/blob/bdb84e2bada3658f99c6a81c963ec562f8485151/src/transformers/models/opt/modeling_opt.py#L840  # noqa: E501
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)

        input_shape = hidden_states.size()[:-1]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            # `modeling_opt._make_causal_mask` @ transformers==4.27.1 doesn't have the `device` argument
            if "device" in inspect.getfullargspec(modeling_opt._make_causal_mask).args:
                kwargs = dict(device=hidden_states.device)
            else:
                kwargs = {}

            combined_attention_mask = modeling_opt._make_causal_mask(
                input_shape,
                hidden_states.dtype,
                past_key_values_length=past_key_values_length,
                **kwargs,
            ).to(hidden_states.device)

        if attention_mask is not None:
            expanded_attn_mask = modeling_opt._expand_mask(
                attention_mask, hidden_states.dtype, tgt_len=input_shape[-1]
            ).to(hidden_states.device)
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )
        attention_mask = combined_attention_mask

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for attn_mask, mask_name in zip([head_mask], ["head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.decoder_blocks)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.decoder_blocks)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.decoder_blocks):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                past_key_value=past_key_value,
                attention_mask=attention_mask,
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if self.final_norm is not None:
            hidden_states = self.final_norm(hidden_states)

        # TODO: Add output projection support
        # https://github.com/huggingface/transformers/blob/699e90437f984d69ad3c9b891dd2e9d0fc2cffe4/src/transformers/models/opt/modeling_opt.py#L499  # noqa: E501
        # if self.project_out is not None:
        #     hidden_states = self.project_out(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        lm_logits = self.lm_head(hidden_states).contiguous()

        if not return_dict:
            return tuple(
                v
                for v in [
                    lm_logits,
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                ]
                if v is not None
            )

        return CausalLMOutputWithValue(
            logits=lm_logits,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class BloomModelBranch(ModelBranch):
    def forward(  # noqa: max-complexity
        self,
        hidden_states: torch.Tensor,  # Takes as input hidden_states instead of input_ids
        output_shape: torch.Tensor,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = False,
    ) -> Union[Tuple, CausalLMOutputWithValue]:
        """Reference:
        https://github.com/huggingface/transformers/blob/2411f0e465e761790879e605a4256f3d4afb7f82/src/transformers/models/bloom/modeling_bloom.py#L623  # noqa: E501
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, seq_length = hidden_states.shape[:2]

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.decoder_blocks))

        head_mask = self.get_head_mask(head_mask, hf_get_num_hidden_layers(self.config))

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values[0] is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length_with_past), device=hidden_states.device)
        else:
            attention_mask = attention_mask.to(hidden_states.device)

        alibi = modeling_bloom.build_alibi_tensor(attention_mask, self.config.n_head, dtype=hidden_states.dtype)

        combined_attention_mask = None
        device = attention_mask.device
        input_shape = (batch_size, seq_length)
        _, src_length = input_shape

        if src_length > 1:
            combined_attention_mask = modeling_bloom._make_causal_mask(
                input_shape,
                device=device,
                past_key_values_length=past_key_values_length,
            )

        expanded_attn_mask = modeling_bloom._expand_mask(attention_mask, tgt_length=src_length)
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask | combined_attention_mask
        )
        causal_mask = combined_attention_mask

        for i, (block, layer_past) in enumerate(zip(self.decoder_blocks, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=causal_mask,
                head_mask=head_mask[i],
                use_cache=use_cache,
                output_attentions=output_attentions,
                alibi=alibi,
            )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        hidden_states = self.final_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            return tuple(
                v
                for v in [
                    lm_logits,
                    hidden_states,
                    presents,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )

        return CausalLMOutputWithValue(
            logits=lm_logits,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class LlamaModelBranch(ModelBranch):
    def _make_causal_mask(self, input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
        """
        Make causal mask used for bi-directional self-attention.
        """
        bsz, tgt_len = input_ids_shape
        mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min))
        mask_cond = torch.arange(mask.size(-1))
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(dtype)

        if past_key_values_length > 0:
            mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

    def _expand_mask(self, mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
        """
        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
        """
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

        inverted_mask = 1.0 - expanded_mask

        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, hidden_states, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = self._make_causal_mask(
                input_shape, hidden_states.dtype, past_key_values_length=past_key_values_length
            ).to(hidden_states.device)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = self._expand_mask(attention_mask, hidden_states.dtype, tgt_len=input_shape[-1]).to(
                hidden_states.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )
        return combined_attention_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_shape: torch.Tensor,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = False,
    ) -> Union[Tuple, CausalLMOutputWithValue]:
        """Reference:
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L491
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size, seq_length = hidden_states.shape[:2]
        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = hidden_states.device if hidden_states is not None else encoder_hidden_states.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), hidden_states, past_key_values_length
        )

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.decoder_blocks):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.final_norm(hidden_states)
        hidden_states = hidden_states.view(output_shape)
        lm_logits = self.lm_head(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            outputs = (lm_logits,) + (None,) + (None,)
            return outputs

        return CausalLMOutputWithValue(
            logits=lm_logits,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class GPTBigCodeModelBranch(ModelBranch):
    def __init__(
        self,
        base_model: transformers.PreTrainedModel,
        *,
        num_layers_unfrozen: int,
    ):
        """
        Args:
            base_model (transformers.PreTrainedModel): The pretrained model to extract upper trunk from
            num_layers_unfrozen (int): The number of trainable layers
        """
        super().__init__(base_model, num_layers_unfrozen=num_layers_unfrozen)
        self.config = base_model.transformer.config
        self.bias = base_model.transformer.bias
        self.multi_query = base_model.transformer.multi_query
        self.get_head_mask = base_model.transformer.get_head_mask

    def forward(  # noqa: C901
        self,
        hidden_states: torch.Tensor,  # Takes as input hidden_states instead of input_ids
        output_shape: torch.Tensor,  # output_size given by main trunk
        past_key_values: Optional[List[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithValue]:
        """Reference:
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_bigcode/modeling_gpt_bigcode.py#L539
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, seq_length = hidden_states.shape[:2]

        if batch_size <= 0:
            raise ValueError("batch_size has to be defined and > 0")

        device = hidden_states.device

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.decoder_blocks))
        else:
            past_length = past_key_values[0].size(-2)

        # Self-attention mask.
        query_length = seq_length
        key_length = past_length + query_length
        self_attention_mask = self.bias[None, key_length - query_length : key_length, :key_length].to(device)

        if attention_mask is not None:
            self_attention_mask = self_attention_mask * attention_mask.view(batch_size, 1, -1).to(
                dtype=torch.bool, device=self_attention_mask.device
            )

        # MQA models: (batch_size, query_length, n_heads, key_length)
        # MHA models: (batch_size, n_heads, query_length, key_length)
        attention_mask = self_attention_mask.unsqueeze(2 if self.multi_query else 1)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None and encoder_attention_mask is not None:
            if encoder_attention_mask.dim() == 2:
                encoder_attention_mask.unsqueeze(1)
            assert encoder_attention_mask.dim() == 3
            encoder_attention_mask = encoder_attention_mask.bool().unsqueeze(2 if self.multi_query else 1)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        presents = [] if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.decoder_blocks, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states = outputs[0]
            if use_cache:
                presents.append(outputs[1])

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

        hidden_states = self.final_norm(hidden_states)

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            return tuple(
                v
                for v in [
                    lm_logits,
                    hidden_states,
                    presents,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )

        return CausalLMOutputWithValue(
            logits=lm_logits,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


# Seq2Seq architectures


@dataclass
class Seq2SeqLMOutputWithValue(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    value: Optional[torch.FloatTensor] = None


class AutoModelForSeq2SeqLMWithValueHead(PreTrainedModelWrapper):
    """An `AutoModel` class wrapper for `transformers` sequence-to-sequence
    models that have a language modeling head and a value head
    """

    _auto_model_parent_class = transformers.AutoModelForSeq2SeqLM
    _supported_modules = ["v_head"]
    _supported_args = ["peft_config", "num_value_layers_unfrozen"]

    def __init__(
        self,
        base_model: transformers.PreTrainedModel,
        peft_config=None,
        num_value_layers_unfrozen=0,
    ):
        super().__init__(base_model, peft_config=peft_config)
        # TODO: Support Seq2Seq value branching
        if num_value_layers_unfrozen > 0:
            raise NotImplementedError("Value branches unsupported for Seq2Seq architecture")
        self.v_head = make_head(hf_get_hidden_size(self.base_model.config), 1)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = True,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = None,
        ignore_peft_adapter: Optional[bool] = None,
    ) -> Seq2SeqLMOutputWithValue:
        forward_kwargs = self.get_compatible_forward_kwargs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        forward_kwargs["output_hidden_states"] = True
        forward_kwargs["return_dict"] = True

        if self.peft_type == "PREFIX_TUNING":
            # In this case peft redefines past_key_values, remove it to avoid an exception.
            forward_kwargs.pop("past_key_values", None)

        if self.peft_type and ignore_peft_adapter:
            if "LORA" in self.peft_type:
                # For LORA, temporarily disable the adapter
                lora_model = self.base_model.base_model
                lora_model.disable_adapter_layers()
                outputs = self.base_model(**forward_kwargs)
                lora_model.enable_adapter_layers()
            else:
                # For prompt or prefix adapters, just use the base model of PeftModel
                outputs = self.base_model.base_model(**forward_kwargs)
        else:
            outputs = self.base_model(**forward_kwargs)

        last_hidden_state = outputs.decoder_hidden_states[-1]
        value = self.v_head(last_hidden_state).squeeze(-1)

        return Seq2SeqLMOutputWithValue(**outputs, value=value)

    def generate(self, *args, **kwargs) -> Union[ModelOutput, torch.LongTensor]:
        return self.base_model.generate(*args, **kwargs)

    def state_dict(self, *args, heads_only=False, **kwargs):
        """
        Returns the state dictionary of the model. We add the state dictionary of the value head
        to the state dictionary of the wrapped model by prepending the key with `v_head.`.
        """
        state_dict = self.v_head.state_dict(*args, **dict(prefix="v_head.", **kwargs))
        if not heads_only:
            state_dict = {**state_dict, **self.base_model.state_dict(*args, **dict(prefix="base_model.", **kwargs))}

        return state_dict

    def post_init(self, state_dict):
        """
        Adds the state dictionary of the value head to the state dictionary of the wrapped model
        by prepending the key with `v_head.`. This function removes the `v_head.` prefix from the
        keys of the value head state dictionary.
        """
        super().post_init()

        trlx_checkpoint = any(k.startswith("base_model.") or k.startswith("v_head.") for k in state_dict)
        self.load_state_dict(state_dict, strict=trlx_checkpoint)

        del state_dict
        gc.collect()  # noqa: E702


class AutoModelForSeq2SeqLMWithHydraValueHead(AutoModelForSeq2SeqLMWithValueHead):
    _supported_modules = ["v_head", "frozen_head"]
    _supported_args = ["num_layers_unfrozen", "peft_config", "num_value_layers_unfrozen"]

    def __init__(
        self,
        base_model: transformers.PreTrainedModel,
        *,
        num_layers_unfrozen: int = -1,
        peft_config=None,
        num_value_layers_unfrozen: int = 0,
    ):
        super().__init__(base_model, peft_config=peft_config, num_value_layers_unfrozen=num_value_layers_unfrozen)
        self.num_layers_unfrozen = num_layers_unfrozen

        if self.num_layers_unfrozen > 0 and not self.peft_type:
            branch_class = T5Branch  # TODO: Add support for other model branches
            self.frozen_head = branch_class(
                self.base_model,
                num_layers_unfrozen=self.num_layers_unfrozen,
            ).eval()
        else:
            self.frozen_head = None

    def forward_hydra(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Seq2SeqLMOutputWithValue:
        forward_kwargs = self.get_compatible_forward_kwargs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return_dict = forward_kwargs.get("return_dict", True)
        forward_kwargs["output_hidden_states"] = True
        forward_kwargs["return_dict"] = True

        if self.peft_type:
            hydra_outputs = self.forward(**forward_kwargs, ignore_peft_adapter=True)
        else:
            outputs = self.forward(**forward_kwargs)
            # Select the hidden state before the first branching layer
            input_hidden_state = outputs.decoder_hidden_states[-(self.num_layers_unfrozen + 1)]
            hydra_outputs = self.frozen_head(
                hidden_states=input_hidden_state,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=outputs.encoder_last_hidden_state,
                encoder_attention_mask=attention_mask,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=return_dict,
            )

        if not return_dict:
            return hydra_outputs.logits
        return hydra_outputs

    def state_dict(self, *args, heads_only=False, **kwargs):
        """
        Returns the state dictionary of the model. We add the state dictionary of the value head
        to the state dictionary of the wrapped model by prepending the key with `v_head.`.
        """
        state_dict = self.v_head.state_dict(*args, **dict(prefix="v_head.", **kwargs))
        if not heads_only:
            state_dict = {
                **state_dict,
                **self.base_model.state_dict(*args, **dict(prefix="" if self.peft_type else "base_model.", **kwargs)),
            }

            if self.frozen_head:
                state_dict = {
                    **state_dict,
                    **self.frozen_head.state_dict(*args, **dict(prefix="frozen_head.", **kwargs)),
                }

        return state_dict

    def post_init(self, state_dict):
        """
        Load `state_dict` into the model. If peft was used to train the model,
        only the value head would be present in the loaded `state_dict`, so the
        loading has to be not strict. Also `frozen_head` will be recreated and
        loaded from the checkpoint, to comply with deepspeed checkpoint loading.
        """
        strict = not self.peft_type and any(k.startswith("base_model.") or k.startswith("v_head.") for k in state_dict)

        if not self.peft_type and self.frozen_head is None:
            for k in state_dict:
                match = re.search(r"^frozen_head\.decoder_blocks\.(\d+)", k)
                if match:
                    self.num_layers_unfrozen = max(self.num_layers_unfrozen, int(match.group(1)) + 1)

            branch_class = T5Branch  # TODO: Add support for other model branches
            self.frozen_head = branch_class(
                self.base_model,
                num_layers_unfrozen=self.num_layers_unfrozen,
            ).eval()

        self.load_state_dict(state_dict, strict=strict)
        del state_dict
        gc.collect()  # noqa: E702


class T5Branch(ModelBranch):
    """Decoder only T5 branch"""

    def __init__(
        self,
        base_model: transformers.PreTrainedModel,
        *,
        num_layers_unfrozen: int,
    ):
        super().__init__(base_model, num_layers_unfrozen=num_layers_unfrozen)
        self.dropout = hf_get_decoder(base_model).dropout
        self.is_decoder = True

    def forward(  # noqa: max-complexity
        self,
        hidden_states: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutputWithValue]:
        """Reference:
        https://github.com/huggingface/transformers/blob/bc21aaca789f1a366c05e8b5e111632944886393/src/transformers/models/t5/modeling_t5.py#L899  # noqa: E501
        """
        batch_size, seq_length = hidden_states.shape[:2]
        input_shape = (batch_size, seq_length)

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_length, device=hidden_states.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=hidden_states.device, dtype=torch.long
            )

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=hidden_states.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        position_bias = None
        encoder_decoder_position_bias = None

        for _, layer_module in enumerate(self.decoder_blocks):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)

        hidden_states = self.final_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        sequence_output = hidden_states

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586  # noqa: E501
            sequence_output = sequence_output * (self.config.d_model**-0.5)

        lm_logits = self.lm_head(sequence_output)

        if not return_dict:
            return (lm_logits,)

        return Seq2SeqLMOutputWithValue(
            logits=lm_logits,
            decoder_hidden_states=all_hidden_states,
            decoder_attentions=all_attentions,
        )


# Branch class utils


def hf_get_branch_class(
    config: transformers.PretrainedConfig,
) -> "ModelBranch":
    """Returns the model branch class for the given config."""
    gpt_branch_supported_archs = [
        "GPTJForCausalLM",
        "GPT2LMHeadModel",
        "GPTNeoForCausalLM",
        "GPTNeoXForCausalLM",
    ]
    opt_branch_supported_archs = ["OPTForCausalLM"]
    bloom_branch_supported_archs = ["BloomModel", "BloomForCausalLM"]
    llama_branch_supported_archs = ["LlamaModel", "LlamaForCausalLM"]
    bigcode_branch_supported_archs = ["GPTBigCodeModel", "GPTBigCodeForCausalLM"]
    arch = config.architectures[0]
    if arch in gpt_branch_supported_archs:
        return GPTModelBranch
    elif arch in opt_branch_supported_archs:
        return OPTModelBranch
    elif arch in bloom_branch_supported_archs:
        return BloomModelBranch
    elif arch in llama_branch_supported_archs:
        return LlamaModelBranch
    elif arch in bigcode_branch_supported_archs:
        return GPTBigCodeModelBranch
    else:
        all_supported_archs = sum(
            [
                gpt_branch_supported_archs,
                opt_branch_supported_archs,
                bloom_branch_supported_archs,
                llama_branch_supported_archs,
                bigcode_branch_supported_archs,
            ],
            [],
        )
        raise ValueError(
            f"Unsupported architecture: `{arch}`. The following architectures are "
            f"available for model branching:\n{all_supported_archs}"
        )
