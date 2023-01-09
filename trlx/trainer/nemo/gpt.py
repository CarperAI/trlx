# Extensible version of the GPT model
from typing import Mapping, Optional, Tuple, Union

import torch
from apex.transformer import parallel_state
from apex.transformer.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
)
from einops import rearrange
from nemo.collections.nlp.models.language_modeling.megatron.gpt_model import (
    post_language_model_processing,
)
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import (
    MegatronGPTModel,
)
from nemo.collections.nlp.modules.common.megatron.module import MegatronModule
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    get_all_params_for_weight_decay_optimization,
    get_ltor_masks_and_position_ids,
    get_params_for_weight_decay_optimization,
)
from torch.nn.utils.rnn import pad_sequence

from trlx.data.ilql_types import ILQLBatch, flatten_dataclass, unflatten_dataclass
from trlx.utils import to_device, tree_map


class LogGPT(MegatronModule):
    def __init__(self, language_model):
        super().__init__()
        self.language_model = language_model
        self.pre_process = language_model.pre_process
        self.post_process = language_model.post_process

        if hasattr(language_model, "word_embeddings"):
            self.word_embeddings = language_model.word_embeddings

    def set_input_tensor(self, input_tensor):
        self.language_model.set_input_tensor(input_tensor)

    def word_embeddings_weight(self):
        return self.language_model.word_embeddings_weight()

    def forward(self, *args, **kwargs):
        print(f"{kwargs['input_ids'].shape=}")
        lm_output = self.language_model(*args, **kwargs)
        print(f"{lm_output.shape=}")
        return lm_output


class LMHeads(MegatronModule):
    def __init__(self, language_model, other_heads):
        super().__init__()
        # must be this attribute name
        self.pre_process = language_model.pre_process
        self.post_process = language_model.post_process
        self.language_model = language_model

        self.other_heads = other_heads

        if hasattr(language_model, "word_embeddings"):
            self.word_embeddings = language_model.word_embeddings

    # The tensor from the previous pipeline rank arrives via this method
    def set_input_tensor(self, input_tensor):
        return self.language_model.set_input_tensor(input_tensor)

    def word_embeddings_weight(self):
        return self.language_model.word_embeddings_weight()

    def forward(
        self,
        *args,
        get_key_value=False,
        forward_method_parallel_output=None,
        heads_kwargs={},
        **kwargs,
    ):
        # print("LMHeads forward")
        lm_output = self.language_model(*args, get_key_value=get_key_value, **kwargs)
        logits = post_language_model_processing(
            lm_output,
            labels=None,
            logit_weights=self.language_model.word_embeddings_weight(),
            get_key_value=get_key_value,
            parallel_output=False,  # self.language_model.parallel_output,
            forward_method_parallel_output=forward_method_parallel_output,
            fp16_lm_cross_entropy=self.language_model.fp16_lm_cross_entropy,
            return_logits=True,
            sequence_parallel=self.language_model.sequence_parallel,
            gradient_accumulation_fusion=self.language_model.gradient_accumulation_fusion,
        )

        if get_key_value:
            logits, logits_presents = logits
            lm_output, lm_output_presents = lm_output

        heads_output = self.other_heads(
            rearrange(lm_output, "T N C -> N T C"), **heads_kwargs
        )
        return logits, heads_output


class ILQLGPT(MegatronGPTModel):
    def __init__(self, ilql_config, **kwargs):
        self.ilql_config = ilql_config
        super().__init__(**kwargs)
        if len(list(self.parameters())) == 0:
            raise ValueError("No parameters in model")
        params = list(self.parameters())

    @classmethod
    def list_available_models(cls) -> Optional[Mapping[str, str]]:
        return None

    def process_global_batch(self, batch: ILQLBatch, global_batch_size=None):
        return batch

    def build_train_valid_test_datasets(self):
        pass
        #    self._train_ds =

    def model_provider_func(self, pre_process: bool, post_process: bool):
        """Model construction for Apex Pipeline Parallelism.
        every rank will construct the model but inside the model,
        only the relevant layers for that rank should be constructed.
        On the first rank, pre_process will be True
        On the last rank, post_process will be True
        """
        print("model_provider_func", pre_process, post_process)
        # This disables post-processing the lm output to the vocab
        gpt = super().model_provider_func(pre_process, post_process=False)
        # This enables the final layernorm in the GPT model if there is one
        gpt.language_model.post_process = post_process
        # If running on the last pipeline stage, add the ILQL heads
        if post_process:
            return LMHeads(
                gpt,
                self.ilql_config.heads(self.cfg.hidden_size, self.padded_vocab_size),
            )
        else:
            return gpt
            old_fwd = gpt.forward

            def log_forward(*args, **kwargs):
                input_shape = None
                if kwargs["input_ids"] is not None:
                    input_shape = kwargs["input_ids"].shape
                elif gpt.language_model.encoder.input_tensor is not None:
                    input_shape = gpt.language_model.encoder.input_tensor.shape
                lm_output = old_fwd(*args, **kwargs)
                return lm_output

            gpt.forward = log_forward
            return gpt

    def get_forward_output_and_loss_func(self, validation_step=False):
        def fwd_output_and_loss_func(
            batch: ILQLBatch, model, checkpoint_activations_all_layers=None
        ):
            # On first and last pipeline stages, the input data is passed in
            if batch is not None:
                batch = unflatten_dataclass(ILQLBatch)(batch)
                batch = to_device(batch, torch.cuda.current_device(), non_blocking=True)

                inputs = batch.input_ids
                labels = batch.input_ids
                if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                    stage = "first"
                elif parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    stage = "last"
                else:
                    stage = "middle"

                pad_by = self.cfg.encoder_seq_length - inputs.shape[1]
                inputs = torch.nn.functional.pad(
                    inputs, (0, pad_by), value=self.tokenizer.eos_id
                )
                labels = torch.nn.functional.pad(
                    labels, (0, pad_by), value=self.tokenizer.eos_id
                )

                (
                    attention_mask,
                    loss_mask,
                    position_ids,
                ) = get_ltor_masks_and_position_ids(
                    data=inputs,
                    eod_token=self.tokenizer.eos_id,
                    reset_position_ids=False,
                    reset_attention_mask=False,
                    eod_mask_loss=False,
                )

                mp_rank = parallel_state.get_tensor_model_parallel_rank()
                mp_size = parallel_state.get_tensor_model_parallel_world_size()

                model_output = model(
                    input_ids=inputs,
                    position_ids=position_ids.long(),
                    attention_mask=attention_mask,
                    labels=labels,
                    checkpoint_activations_all_layers=checkpoint_activations_all_layers,
                )
            else:
                # In-between stages are given data via the pipeline engine
                # Still need to specify thes arguments to avoid errors
                model_output = model(
                    input_ids=None, position_ids=None, attention_mask=None
                )

            def loss_func(model_output):

                # TODO: implement this in a sequence parallel way
                model_output = tree_map(
                    lambda t: rearrange(t, "N T C -> T N C"), model_output
                )
                model_output = tree_map(
                    lambda t: gather_from_sequence_parallel_region(
                        t, to_model_parallel=False
                    ),
                    model_output,
                )
                model_output = tree_map(
                    lambda t: rearrange(t, "T N C -> N T C"), model_output
                )
                logits, (qs, target_qs, vs) = model_output
                qs = tree_map(
                    lambda t: t.gather(
                        dim=1,
                        index=batch.actions_ixs.unsqueeze(-1).repeat(1, 1, t.shape[-1]),
                    ).contiguous(),
                    qs,
                )
                target_qs = tree_map(
                    lambda t: t.gather(
                        dim=1,
                        index=batch.actions_ixs.unsqueeze(-1).repeat(1, 1, t.shape[-1]),
                    ).contiguous(),
                    target_qs,
                )
                vs = tree_map(
                    lambda t: t.gather(
                        dim=1,
                        index=batch.states_ixs.unsqueeze(-1).repeat(1, 1, t.shape[-1]),
                    ).contiguous(),
                    vs,
                )

                model_output = (logits, (qs, target_qs, vs))

                if mp_rank == 0:
                    loss_for_mb, stats = self.ilql_config.loss(model_output, batch)
                else:
                    loss_for_mb, stats = self.ilql_config.loss(model_output, batch)
                    loss_for_mb *= 0
                    stats = {}

                stats = tree_map(lambda v: v.detach().item(), stats)

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
                        "loss_sum_and_mb_size": loss_sum_and_mb_size_all_gpu,
                        **stats,
                    }
                else:
                    reduced_loss = average_losses_across_data_parallel_group(
                        [loss_for_mb]
                    )
                    return loss_for_mb, {"avg": reduced_loss, **stats}

            def contiguous_loss(x):
                loss, stats = loss_func(x)
                return tree_map(lambda t: t.contiguous(), loss), stats

            return tree_map(lambda t: t.contiguous(), model_output), contiguous_loss

        return fwd_output_and_loss_func

    def get_forward_output_only_func(
        self,
        set_inference_key_value_memory=False,
        inference_max_sequence_len=None,
        checkpoint_activations_all_layers=None,
    ):
        def fwd_output_only_func(
            batch: ILQLBatch,
            model,
        ):
            batch = unflatten_dataclass(ILQLBatch)(batch)
            batch = to_device(batch, torch.cuda.current_device(), non_blocking=True)

            inputs = batch.input_ids[:, :-1].long()
            labels = batch.input_ids[:, 1:].long()

            attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
                data=inputs,
                eod_token=self.tokenizer.eos_id,
                reset_position_ids=False,
                reset_attention_mask=False,
                eod_mask_loss=False,
            )

            output_tensor = model(
                input_ids=inputs,
                position_ids=position_ids.long(),
                attention_mask=attention_mask,
                labels=labels,
                set_inference_key_value_memory=set_inference_key_value_memory,
                inference_max_sequence_len=None,
                heads_kwargs=dict(
                    states_ixs=batch.states_ixs, actions_ixs=batch.actions_ixs
                ),
            )

            def id_func(output_tensor):
                return output_tensor, {"logits": output_tensor}

            return output_tensor, id_func

        return fwd_output_only_func
