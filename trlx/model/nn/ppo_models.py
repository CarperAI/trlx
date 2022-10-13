from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Identity
from transformers import (AutoConfig, AutoModelForCausalLM, GPT2LMHeadModel,
                          GPT2Model, GPT2PreTrainedModel, GPT2Tokenizer,
                          GPTJModel, PretrainedConfig, PreTrainedModel,
                          top_k_top_p_filtering)
from transformers.modeling_outputs import ModelOutput
from copy import deepcopy
import inspect


# Cell
@dataclass
class CausalLMOutputWithCrossAttentions(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    value: Optional[torch.FloatTensor] = None


# Cell


def make_head(n_embd: int, out: int):
    return nn.Sequential(
        nn.Linear(n_embd, n_embd * 2), nn.ReLU(), nn.Linear(n_embd * 2, out)
    )


# Cell


class GPTHeadWithValueModel(nn.Module):
    """
    The GPTHeadWithValueModel class implements a GPT-type language model with a secondary, scalar head.
    """
    def __init__(self, config: Union[PretrainedConfig, str]):
        super().__init__()
        if isinstance(config, PretrainedConfig):
            self.gpt = AutoModelForCausalLM.from_config(config)
        else:
            self.gpt = AutoModelForCausalLM.from_pretrained(config)

        if hasattr(self.gpt.config, "hidden_size"):
            self.n_embd = self.gpt.config.hidden_size
        else:
            self.n_embd = self.gpt.config.n_embd

        self.v_head = make_head(self.n_embd, 1)

    def generate(self, input_ids, **x):
        return self.gpt.generate(input_ids, **x)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mc_token_ids=None,
        lm_labels=None,
        mc_labels=None,
        return_dict=False,
        output_attentions=False,
        output_hidden_states=False,
    ):
        loss = None
        transformer_outputs = self.gpt.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = transformer_outputs[0]
        lm_logits = self.gpt.lm_head(hidden_states)
        value = self.v_head(hidden_states).squeeze(-1)

        if not return_dict:
            outputs = (lm_logits,) + transformer_outputs[1:] + (value,)
            return outputs

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
            value=value,
        )


# Cell

class ModelBranch(PreTrainedModel):
    def __init__(self, config, transformer_blocks, ln_f, lm_head):
        super().__init__(config)

        # Defined by the main trunk
        self.n_embd = config.n_embd

        self.h = deepcopy(nn.ModuleList(transformer_blocks))
        self.ln_f = deepcopy(ln_f)
        self.lm_head = deepcopy(lm_head)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False
        
        # Turning off grad saves memory
        for block in self.h:
            for parameter in block.parameters():
                parameter.requires_grad = False
        for parameter in lm_head.parameters():
            parameter.requires_grad = False

    def forward(
        self,
        hidden_states: torch.Tensor,  # Takes as input hidden_states instead of input_ids
        output_shape: torch.Tensor,  # output_size given by main trunk
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = False,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:

        batch_size = hidden_states.size()[0]

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        device = hidden_states.device

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # Assumes we are never training the branch
            block_params = inspect.getfullargspec(block.forward).args
            if "encoder_hidden_states" in block_params:
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
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # last_hidden_state = hidden_states
        # past_key_values = presents
        # hidden_states = all_hidden_states
        # attentions = all_self_attentions
        # cross_attentions = all_cross_attentions

        ### START OF CAUSAL HEAD ###
        #hidden_states = hidden_states.to(torch.float32) Present for gptj

        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)
        
        lm_logits = self.lm_head(hidden_states)
        
        if not return_dict:
            outputs = (lm_logits,) + (None,) + (None,)
            return outputs

        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=lm_logits,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
            value=None,
        )



class GPTHydraHeadWithValueModel(nn.Module):
    """The GPTHeadWithValueModel class implements a GPT-type language model with a secondary, scalar head."""

    def __init__(self, config: Union[PretrainedConfig, str], num_layers_unfrozen: int = -1):
        super().__init__()
        if isinstance(config, PretrainedConfig):
            self.gpt = AutoModelForCausalLM.from_config(config)
        else:
            self.gpt = AutoModelForCausalLM.from_pretrained(config)

        if hasattr(self.gpt.config, "hidden_size"):
            self.n_embd = self.gpt.config.hidden_size
            self.gpt.config.n_embd = self.n_embd
        else:
            self.n_embd = self.gpt.config.n_embd

        self.v_head = make_head(self.n_embd, 1)

        self.num_layers_unfrozen = num_layers_unfrozen
        if num_layers_unfrozen > 0:
            transformer_blocks = list(self.gpt.transformer.h)[-num_layers_unfrozen:]
            # Retrive hf_config to init
            hf_config = AutoConfig.from_pretrained(config)
            hf_config.n_embd = self.n_embd
            self.frozen_head = ModelBranch(hf_config, transformer_blocks, self.gpt.transformer.ln_f, self.gpt.lm_head)

    def generate(self, input_ids, **x):
        return self.gpt.generate(input_ids, **x)

    def forward_hydra(self, input_ids, **x):
        if x.get('return_dict') is not None:
            return_dict = x['return_dict']
        else:
            return_dict = True
        x['return_dict'] = True
        x['output_hidden_states'] = True
        output = self.forward(input_ids, **x)
        all_hidden_states = output.hidden_states
        # Get output of last frozen hidden layer
        # Select hidden state before first layer of branch. 
        input_hidden_state = all_hidden_states[-(self.num_layers_unfrozen+1)]
        # Get size of last hidden state
        output_shape = all_hidden_states[-1].size()
        outputs = self.frozen_head(input_hidden_state, output_shape, **x)
        if not return_dict:
            return outputs.logits
        return outputs

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mc_token_ids=None,
        lm_labels=None,
        mc_labels=None,
        return_dict=False,
        output_attentions=False,
        output_hidden_states=False,
    ):
        loss = None
        transformer_outputs = self.gpt.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
        )
        hidden_states = transformer_outputs[0]
        lm_logits = self.gpt.lm_head(hidden_states)
        value = self.v_head(hidden_states).squeeze(-1)

        if not return_dict:
            outputs = (lm_logits,) + transformer_outputs[1:] + (value,)
            return outputs

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=None,
            value=value,
        )