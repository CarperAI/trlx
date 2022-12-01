# Extensible version of the GPT model
from typing import Optional, Tuple, Union
from trlx.data.ilql_types import ILQLBatch
from trlx.utils import to_device, tree_map

from nemo.collections.nlp.modules.common.megatron.module import MegatronModule
from nemo.collections.nlp.modules.common.megatron.utils import (
    get_ltor_masks_and_position_ids,
)
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import (
    MegatronGPTModel,
    post_language_model_processing,
)


class LMHeads(MegatronModule):
    def __init__(self, language_model, other_heads):
        super().__init__()
        # must be this attribute name
        self.language_model = language_model
        self.other_heads = other_heads

    # The tensor from the previous pipeline rank arrives via this method
    def set_input_tensor(self, input_tensor):
        return self.language_model.set_input_tensor(input_tensor)

    def forward(self, *args, get_key_value=False, **kwargs):
        lm_output = self.language_model(*args, get_key_value=get_key_value, **kwargs)
        loss, logits = post_language_model_processing(
            lm_output,
            labels=kwargs.get("labels", None),
            logit_weights=self.language_model.word_embeddings_weight(),
            get_key_value=kwargs.get("get_key_value", False),
            parallel_output=self.language_model.parallel_output,
            forward_model_parallel_output=forward_method_parallel_output,
            fp16_lm_cross_entropy=self.language_model.fp16_lm_cross_entropy,
            return_logits=kwargs.get("encoder_input", None) is not None,
            sequence_parallel=self.language_model.sequence_parallel,
            gradient_accumulation_fusion=self.language_model.gradient_accumulation_fusion,
        )

        if get_key_value:
            logits, logits_presents = logits
            lm_output, lm_output_presents = lm_output

        heads_output = self.other_heads(lm_output, *args, **kwargs)
        return loss, (logits, heads_output)


class GPT(MegatronGPTModel):
    def __init__(self, ilql_config, **kwargs):
        super().__init__(**kwargs)
        self.ilql_heads = ilql_config.heads

    @classmethod
    def list_available_models(cls) -> Optional[Dict[str, str]]:
        return None

    def model_provider_func(self, pre_process: bool, post_process: bool):
        """Model construction for for Apex Pipeline Parallelism.
        every rank will construct the model but inside the model,
        only the relevant layers for that rank will be constructed.
        On the first rank, pre_process will be True
        On the last rank, post_process will be True
        """
        # This disables post-processing the lm output to the vocab
        gpt = super().model_provider_func(pre_process, post_process=False)
        # This enables the final layernorm in the GPT model if there is one
        gpt.language_model.post_process = post_process
        # If running on the last pipeline stage, add the ILQL heads
        if post_process:
            return LMHeads(gpt, self.ilql_heads)
        return gpt

    def get_forward_output_and_loss_func(self, validation_step=False):
        def fwd_output_and_loss_func(
            batch: ILQLBatch, model, checkpoint_activations_all_layers=None
        ):
            batch = to_device(batch, "cuda", nonblocking=True)

            inputs = batch.input_ids[:, :-1]
            labels = batch.input_ids[:, 1:]

            position_ids, loss_mask, attention_mask = get_ltor_masks_and_position_ids(
                data=inputs,
                eod_token=self.language_model.tokenizer.eos_id,
                reset_position_ids=False,
                reset_attention_mask=False,
                eod_mask_loss=False,
            )

            loss_tensor = model(
                input_ids=inputs,
                position_ids=position_ids,
                attention_mask=attention_mask,
                labels=labels,
                checkpoint_activations_all_layers=checkpoint_activations_all_layers,
            )

            def loss_func(loss_tensor):
                print("loss_tensor", loss_tensor)
                loss_for_mb = self.loss_func(loss_mask, loss_tensor)
                if validation_step and not self.cfg.data.get(
                    "validation_drop_last", True
                ):
                    num_valid_samples_in_mb = int(
                        loss_mask.sum() / loss_mask.numel() * loss_mask.shape[0]
                    )
                    loss_sum_for_mb = num_valid_samples_in_mb * loss_for_mb
                    loss_sum_and_mb_size_all_gpu = torch.cat(
                        [
                            loss_sum_for_mb.clone().detach().view(1),
                            torch.tensor([num_valid_samples_in_mb])
                            .cuda()
                            .clone()
                            .detach(),
                        ]
                    )
                    # Could potentially reduce num_valid_samples_in_microbatch and use that to aggregate instead of len(self._validation_ds)
                    torch.distributed.all_reduce(
                        loss_sum_and_mb_size_all_gpu,
                        group=parallel_state.get_data_parallel_group(),
                    )
                    return loss_for_mb, {
                        "loss_sum_and_mb_size": loss_sum_and_mb_size_all_gpu
                    }
                else:
                    reduced_loss = average_losses_across_data_parallel_group(
                        [loss_for_mb]
                    )
                    return loss_for_mb, {"avg": reduced_loss}

            return output_tensor, loss_func

        return fwd_output_and_loss_func

    def get_forward_output_only_func(
        self, set_inference_key_value_memory=False, inference_max_sequence_len=None
    ):
        def fwd_output_only_func(
            batch: ILQLBatch,
            model,
        ):
            batch = to_device(batch, "cuda", nonblocking=True)

            inputs = batch.input_ids[:, :-1]
            labels = batch.input_ids[:, 1:]

            position_ids, loss_mask, attention_mask = get_ltor_masks_and_position_ids(
                data=inputs,
                eod_token=self.language_model.tokenizer.eos_id,
                reset_position_ids=False,
                reset_attention_mask=False,
                eod_mask_loss=False,
            )

            output_tensor = model(
                input_ids=inputs,
                position_ids=position_ids,
                attention_mask=attention_mask,
                labels=labels,
                checkpoint_activations_all_layers=checkpoint_activations_all_layers,
                set_inference_key_value_memory=set_inference_key_value_memory,
                inference_max_sequence_len=None,
            )

            def id_func(output_tensor):
                return output_tensor, {"logits": output_tensor}

            return output_tensor, id_func

        return fwd_output_only_func
