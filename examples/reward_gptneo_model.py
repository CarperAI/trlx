import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers.utils import ModelOutput
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions, CausalLMOutputWithPast
from transformers.models.gptj.modeling_gptj import GPTJPreTrainedModel, GPTJModel, GPTJForCausalLM

class RewardOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    r0: Optional[torch.FloatTensor] = None
    r1: Optional[torch.FloatTensor] = None

class GPTJRewardModel(GPTJPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"h\.\d+\.attn\.bias"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPTJModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.reward_head = nn.Linear(config.n_embd, 1, bias=False)
        
        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.transformer.h))
        self.transformer.parallelize(self.device_map)
        #self.lm_head = self.lm_head.to(self.transformer.first_device)
        self.reward_head = self.reward_head.to(self.transformer.first_device)
        self.model_parallel = True

    def deparallelize(self):
        self.transformer.deparallelize()
        self.transformer = self.transformer.to("cpu")
        #self.lm_head = self.lm_head.to("cpu")
        self.reward_head = self.reward_head.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True 
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.reward_head.weight.device)

        if output_hidden_states:
            hidden_states_0 = hidden_states[:input_ids.shape[0] // 2,] # first half of batch is the first part of summaries
            hidden_states_1 = hidden_states[input_ids.shape[0] // 2:,] # second half of batch is the second part of summaries
            logits_0 = self.reward_head(hidden_states_0)
            logits_1 = self.reward_head(hidden_states_1)
            batch_size = input_ids.shape[0] // 2
            if self.config.pad_token_id is None:
                sequence_lengths_0 = -1
                sequence_lengths_1 = -1
            else:
                sequence_lengths_0 = torch.ne(input_ids[:input_ids.shape[0] // 2], self.config.pad_token_id).sum(-1) - 1
                sequence_lengths_1 = torch.ne(input_ids[input_ids.shape[0] // 2:], self.config.pad_token_id).sum(-1) - 1
            r0 = logits_0[torch.arange(batch_size, device=logits_0.device), sequence_lengths_0]
            r1 = logits_1[torch.arange(batch_size, device=logits_1.device), sequence_lengths_1]
            r0 = r0.squeeze(-1)
            r1 = r1.squeeze(-1)
            loss = torch.mean(-torch.log(torch.sigmoid(r0 - r1))) # loss is negative log likelihood of r0 > r1 (r0 always > r1 following train dataset class)

        return CausalLMOutputWithPast(
            loss=loss,
            logits= torch.stack([r0, r1], dim=1),
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PretrainedModel.beam_search`] or
        [`~PretrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )

    def get_pretrained_model(self, pretrained_model):
        pretrained_dict = pretrained_model.state_dict()
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)