# Extensible version of the GPT model
import os
import sys
from functools import reduce
from math import floor
from typing import List, Mapping, Optional, Union

import torch
import torch.distributed
import torch.fx
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F
from apex.transformer import parallel_state, tensor_parallel
from apex.transformer.pipeline_parallel.utils import get_num_microbatches
from apex.transformer.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
)
from einops import rearrange
from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_samplers import (
    MegatronPretrainingBatchSampler,
    MegatronPretrainingRandomBatchSampler,
)
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
from nemo.collections.nlp.modules.common.text_generation_utils import (
    get_default_length_params,
    get_default_sampling_params,
)
from nemo.collections.nlp.modules.common.transformer.text_generation import (
    LengthParam,
    OutputType,
    SamplingParam,
)

# import trlx.trainer.nemo.generate_ilql as generate_ilql
from trlx.data.ilql_types import ILQLBatch, flatten_dataclass, unflatten_dataclass
from trlx.trainer.nn.ilql_models import ILQLConfig
from trlx.trainer.nn.ppo_models import PPOConfig
from trlx.utils import set_seed, to_device, tree_map


class ParallelLinear(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        init_method=nn.init.xavier_normal_,
        use_cpu_initialization=False,
        bias=True,
        sequence_parallel=False,
        gradient_accumulation_fusion=False,
        gather_output=True,
        input_is_parallel=False,
    ):
        """Linear layer with optional bias and activation fused into the linear layer."""
        super().__init__()

        no_async_tensor_model_parallel_allreduce = (
            parallel_state.get_tensor_model_parallel_world_size() == 1
            or sequence_parallel
        )

        if in_size < out_size:
            self.layer = tensor_parallel.ColumnParallelLinear(
                in_size,
                out_size,
                gather_output=gather_output,
                init_method=init_method,
                skip_bias_add=False,
                use_cpu_initialization=use_cpu_initialization,
                bias=bias,
                sequence_parallel_enabled=sequence_parallel,
                no_async_tensor_model_parallel_allreduce=no_async_tensor_model_parallel_allreduce,
                gradient_accumulation_fusion=gradient_accumulation_fusion,
            )
        else:
            self.layer = tensor_parallel.RowParallelLinear(
                in_size,
                out_size,
                input_is_parallel=input_is_parallel,
                init_method=init_method,
                skip_bias_add=False,
                use_cpu_initialization=use_cpu_initialization,
                bias=bias,
                sequence_parallel_enabled=sequence_parallel,
                gradient_accumulation_fusion=gradient_accumulation_fusion,
            )

    def forward(self, x):
        output, bias = self.layer(x)
        if bias is not None:
            return output + bias
        return output


def make_parallel_head(n_embd: int, out: int, sequence_parallel=False) -> nn.Sequential:
    """Returns a generic sequential model parallel MLP head."""
    parallel_intermediate = out < (n_embd * 2)
    return nn.Sequential(
        ParallelLinear(
            n_embd,
            n_embd * 2,
            sequence_parallel=sequence_parallel,
            gather_output=not parallel_intermediate,
        ),
        nn.ReLU(),
        ParallelLinear(
            n_embd * 2,
            out,
            sequence_parallel=sequence_parallel,
            input_is_parallel=parallel_intermediate,
        ),
    )


class ParallelILQLHeads(nn.Module):
    def __init__(
        self,
        config: ILQLConfig,
        hidden_size: int,
        vocab_size: int,
        sequence_parallel=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.v_head = make_parallel_head(
            hidden_size, 1, sequence_parallel=sequence_parallel
        )
        self.config = config

        n_qs = 2 if self.config.two_qs else 1

        self.q_heads = nn.ModuleList(
            make_parallel_head(self.hidden_size, self.vocab_size) for _ in range(n_qs)
        )

        self.target_q_heads = nn.ModuleList(
            make_parallel_head(self.hidden_size, self.vocab_size) for _ in range(n_qs)
        )

        # Copy q_heads params into target_q_heads params
        self._sync_target_q_heads(1.0)

    def forward(self, hidden_states):
        qs = tuple(q_head(hidden_states) for q_head in self.q_heads)
        with torch.no_grad():
            target_qs = tuple(q_head(hidden_states) for q_head in self.target_q_heads)
        vs = self.v_head(hidden_states)

        qs, target_qs, vs = tree_map(
            lambda t: rearrange(t, "T N ... -> N T ..."), (qs, target_qs, vs)
        )

        return qs, target_qs, vs

    def _sync_target_q_heads(self, alpha: float):
        for target_q_head, q_head in zip(self.target_q_heads, self.q_heads):
            for target_param, copy_param in zip(
                target_q_head.parameters(), q_head.parameters()
            ):
                target_param.data.copy_(
                    (alpha * copy_param.data) + (1.0 - alpha) * target_param.data
                )

    def sync_target_q_heads(self):
        self._sync_target_q_heads(self.config.alpha)


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

        heads_output = self.other_heads(lm_output)
        return logits, heads_output


class HydraWithValueHeads(MegatronModule):
    def __init__(self, language_models):
        super().__init__()
        self.language_models = language_models

    def set_input_tensor(self, input_tensor):
        for model in self.language_models:
            model.set_input_tensor(input_tensor)

    def word_embeddings_weight(self):
        return self.language_models[0].word_embeddings_weight()

    def forward(
        self,
        *args,
        get_key_value=False,
        forward_method_parallel_output=None,
        heads_kwargs={},
        **kwargs,
    ):
        # print("LMHeads forward")
        lm_outputs = []
        logits = []
        for model in self.language_models:
            lm_output = model(*args, get_key_value=get_key_value, **kwargs)
            model_logits = post_language_model_processing(
                lm_output,
                labels=None,
                logit_weights=model.word_embeddings_weight(),
                get_key_value=get_key_value,
                parallel_output=False,  # self.language_model.parallel_output,
                forward_method_parallel_output=forward_method_parallel_output,
                fp16_lm_cross_entropy=model.fp16_lm_cross_entropy,
                return_logits=True,
                sequence_parallel=model.sequence_parallel,
                gradient_accumulation_fusion=model.gradient_accumulation_fusion,
            )
            lm_outputs.append(lm_output)
            logits.append(model_logits)

        return lm_outputs, logits


class ILQLGPT(MegatronGPTModel):
    ilql_config: ILQLConfig

    def __init__(self, ilql_config, **kwargs):
        self.ilql_config = ilql_config
        super().__init__(**kwargs)
        if len(list(self.parameters())) == 0:
            raise ValueError("No parameters in model")

    @classmethod
    def list_available_models(cls) -> Optional[Mapping[str, str]]:
        return None

    def process_global_batch(self, batch: ILQLBatch, global_batch_size=None):
        return batch

    def build_train_valid_test_datasets(self):
        pass
        #    self._train_ds =

    def build_data_loader(self, dataset, collate_fn):
        dp_rank = parallel_state.get_data_parallel_rank()
        dp_size = parallel_state.get_data_parallel_world_size()
        print(
            f"Building data loader for {dataset=} {len(dataset)=} {dp_rank=} {dp_size=}",
            file=sys.stderr,
        )
        batch_sampler = MegatronPretrainingRandomBatchSampler(
            total_samples=len(dataset),
            consumed_samples=0,
            micro_batch_size=self.cfg.micro_batch_size,
            global_batch_size=self.cfg.global_batch_size,
            data_parallel_rank=dp_rank,
            data_parallel_size=dp_size,
            drop_last=True,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def set_train_dataset(self, train_dataset, collate_fn):
        self._train_dataset = train_dataset
        self._train_collate_fn = collate_fn

    def setup_training_data(self, _):
        if hasattr(self, "_train_dataset"):
            self._train_dl = self.build_data_loader(
                self._train_dataset, self._train_collate_fn
            )

    def setup(self, stage=None):
        """PTL hook that is executed after DDP spawns"""
        print(f"ILQLGPT setup {stage=}")
        super().setup(stage)
        # self._trail_dl = self.build_data_loader(self._train_dataset, self._train_collate_fn)

    def model_provider_func(self, pre_process: bool, post_process: bool):
        """
        Model construction for Apex Pipeline Parallelism.
        Each rank will construct the model but inside the model,
        only the relevant layers for that rank should be constructed.
        On the first rank, pre_process will be True
        On the last rank, post_process will be True
        """
        # This disables post-processing the lm output to the vocab
        gpt = super().model_provider_func(pre_process, post_process=False)
        # return gpt
        # This enables the final layernorm in the GPT model if there is one
        gpt.language_model.post_process = post_process
        # If running on the last pipeline stage, add the ILQL heads
        if post_process:
            parallel_ilql_heads = ParallelILQLHeads(
                self.ilql_config,
                self.cfg.hidden_size,
                self.padded_vocab_size,
                self.cfg.sequence_parallel,
            )

            return LMHeads(
                gpt,
                parallel_ilql_heads,
            )
        else:
            old_fwd = gpt.forward

            def log_forward(*args, **kwargs):
                input_shape = None
                if kwargs["input_ids"] is not None:
                    input_shape = kwargs["input_ids"].shape
                elif gpt.language_model.encoder.input_tensor is not None:
                    input_shape = gpt.language_model.encoder.input_tensor.shape
                # print(f"{parallelz_state.get_tensor_model_parallel_rank()} {input_shape=}")
                lm_output = old_fwd(*args, **kwargs)
                # print(
                #     f"{parallel_state.get_tensor_model_parallel_rank()} {input_shape=} {lm_output.shape=}"
                # )
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
                if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                    stage = "first"
                elif parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    stage = "last"
                    if not validation_step:
                        if (
                            self.global_step % self.ilql_config.steps_for_target_q_sync
                            == 0
                        ):
                            model.module.other_heads.sync_target_q_heads()
                else:
                    stage = "middle"

                pad_by = self.cfg.encoder_seq_length - inputs.shape[1]
                inputs = torch.nn.functional.pad(
                    inputs, (0, pad_by), value=self.tokenizer.eos_id
                )

                # print(f"{pad_by=} {self.cfg.encoder_seq_length=} {inputs.shape=}")

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
                )
            else:
                # In-between stages are given data via the pipeline engine
                # Still need to specify thes arguments to avoid errors
                model_output = model(
                    input_ids=None, position_ids=None, attention_mask=None
                )

            def gather_ntc(t: torch.Tensor):
                """Gather sequence parallel tensor [batch, seq, hidden]"""
                t = rearrange(t, "N T ... -> T N ...")
                t = gather_from_sequence_parallel_region(t, to_model_parallel=False)
                t = rearrange(t, "T N ... -> N T ...")
                return t

            def loss_func(model_output):

                # # TODO: implement this in a sequence parallel way
                if self.cfg.sequence_parallel:
                    model_output = tree_map(gather_ntc, model_output)

                logits, (qs, target_qs, vs) = model_output

                # tree_map(lambda t: print(f"{mp_rank=} {t.shape=}"), qs)
                # tree_map(lambda t: print(f"{mp_rank=} {t.shape=}"), target_qs)
                # print(f"{mp_rank=} {batch.actions_ixs.shape=}")
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

                # We compute the loss on the last rank because NeMo takes the loss to
                # log from the last rank
                if mp_rank == 0:
                    loss_for_mb, stats = self.ilql_config.loss(model_output, batch)
                else:
                    loss_for_mb, stats = self.ilql_config.loss(model_output, batch)
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

            return model_output, loss_func

        return fwd_output_and_loss_func

    def get_forward_output_only_func(
        self,
        set_inference_key_value_memory=False,
        inference_max_sequence_len=None,
        checkpoint_activations_all_layers=None,
    ):
        print(f"get_forward_output_only_func")

        def fwd_output_only_func(
            batch: torch.Tensor,
            model,
        ):
            if batch is not None:
                print(f"{batch=}")
                batch = to_device(batch, torch.cuda.current_device(), non_blocking=True)

                extra_arg = {}

                if len(batch) == 3:
                    tokens, attention_mask, position_ids = batch
                    attention_mask = attention_mask[0:1]
                else:
                    (
                        tokens,
                        attention_mask,
                        position_ids,
                        set_inference_key_value_memory,
                        inference_max_sequence_len,
                    ) = batch

                    extra_arg[
                        "set_inference_key_value_memory"
                    ] = set_inference_key_value_memory[0].item()
                    extra_arg[
                        "inference_max_sequence_len"
                    ] = self.cfg.encoder_seq_length

                    # extra_arg['inference_max_sequence_len'] = inference_max_sequence_len[0].item()

                attention_mask = attention_mask[0:1]

                # pad_by = floor(tokens.shape[1] / mp_size) * mp_size - tokens.shape[1]
                # pad_by = self.cfg.encoder_seq_length - tokens.shape[1]
                # # )
                # tokens = torch.nn.functional.pad(
                #     tokens, (0, pad_by), value=self.tokenizer.eos_id
                # )

                # position_ids = torch.arange(
                #     0, tokens.shape[1], device=tokens.device
                # ).long()[None, :]

                model_output = model(
                    input_ids=tokens,
                    position_ids=position_ids.long(),
                    attention_mask=attention_mask,
                    **extra_arg,
                )
            else:
                model_output = model(
                    input_ids=None, position_ids=None, attention_mask=None
                )

            def ilql_postprocess(model_output):
                logits, (_, target_qs, vs) = model_output
                last_qs = [q[:, -1] for q in target_qs]
                target_q = reduce(torch.minimum, last_qs)
                advantage = target_q - vs[:, -1]
                pi_beta = F.log_softmax(logits[:, -1], -1)
                beta = self.ilql_config.gen_kwargs.get("beta", 1.0)

                print(f"{pi_beta.shape=}, {advantage.shape=}, {logits.shape=} {beta=}")

                pi_adjusted = pi_beta + advantage * beta
                logits[:, -1] = pi_adjusted

                # target_q = reduce(torch.minimum, target_qs)
                # advantage = target_q - vs
                # pi_beta = F.log_softmax(logits, -1)
                # beta = self.ilql_config.gen_kwargs.get("beta", 1.0)

                # print(f"{pi_beta.shape=}, {advantage.shape=}, {logits.shape=} {beta=}")

                # logits = pi_beta + advantage * beta

                return logits, {"logits": logits}

            return model_output, ilql_postprocess

        return fwd_output_only_func

    # def generate(
    #     self,
    #     inputs: Union[List[str], torch.Tensor, List[dict]],
    #     length_params: LengthParam = None,
    #     sampling_params: SamplingParam = None,
    # ) -> OutputType:

    #     # check whether the DDP is initialized
    #     if parallel_state.is_unitialized():

    #         def dummy():
    #             return

    #         if self.trainer.strategy.launcher is not None:
    #             self.trainer.strategy.launcher.launch(dummy, trainer=self.trainer)
    #         self.trainer.strategy.setup_environment()

    #     # set the default sampling params if it is None.
    #     # default do greedy sampling
    #     if sampling_params is None:
    #         sampling_params = get_default_sampling_params()

    #     # set the default length params if it is None.
    #     # default do greedy sampling
    #     if length_params is None:
    #         length_params = get_default_length_params()

    #     return generate_ilql.generate(
    #         self.cuda(),
    #         inputs=inputs,
    #         task_ids=None,
    #         tokens_to_generate=length_params["max_length"],
    #     )


class HydraWithValueHeadGPT(MegatronGPTModel):
    ppo_config: PPOConfig

    def __init__(self, ppo_config, **kwargs):
        self.ppo_config = ppo_config
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
        # Currently the Hydra head can only be as large as one pipeline parallelism stage
        def freeze_gpt(gpt):
            for p in gpt.parameters():
                p.requires_grad = False

        if post_process:
            gpts = [
                super().model_provider_func(pre_process, post_process) for i in range(2)
            ]

            for gpt in gpts[1:]:
                freeze_gpt(gpt)

            return HydraWithValueHeads(gpts)
        else:
            gpt = super().model_provider_func(pre_process, post_process)
            freeze_gpt(gpt)
            return gpt
