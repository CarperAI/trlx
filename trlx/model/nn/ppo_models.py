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
    """The GPTHeadWithValueModel class implements a GPT-type language model with a secondary, scalar head."""

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
