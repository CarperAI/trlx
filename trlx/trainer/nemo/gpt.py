# Extensible version of the GPT model
import os
import sys
from collections import OrderedDict
from copy import deepcopy
from functools import partial, reduce
from math import sqrt
from pathlib import Path
from typing import List, Mapping, Optional, Union

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
from apex.transformer import parallel_state, tensor_parallel
from apex.transformer.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
)
from einops import rearrange
from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_samplers import (
    MegatronPretrainingRandomBatchSampler,
)
from nemo.collections.nlp.models.language_modeling.megatron.gpt_model import (
    post_language_model_processing,
)
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import (
    MegatronGPTModel,
)
from nemo.collections.nlp.modules.common.megatron.module import (
    Float16Module,
    MegatronModule,
)
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    get_ltor_masks_and_position_ids,
)
from nemo.collections.nlp.modules.common.transformer.text_generation import (
    LengthParam,
    OutputType,
    SamplingParam,
)
from nemo.collections.nlp.parts.utils_funcs import get_last_rank

# import trlx.trainer.nemo.generate_ilql as generate_ilql
from trlx.data.ilql_types import ILQLBatch, flatten_dataclass, unflatten_dataclass
from trlx.trainer.nn.ilql_models import ILQLConfig, batched_index_select
from trlx.trainer.nn.ppo_models import PPOConfig
from trlx.utils import to_device, tree_map


class ParallelLinear(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        init_method=partial(nn.init.kaiming_uniform_, a=sqrt(5), nonlinearity="relu"),
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

        self.target_q_heads = nn.ModuleList(deepcopy(q_head) for q_head in self.q_heads)
        self.target_q_heads.requires_grad_(False)

    def forward(self, hidden_states):
        qs = tuple(q_head(hidden_states) for q_head in self.q_heads)
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

    def load_state_dict(self, state_dict, strict=True):
        """Load GPTModel state dict."""

        def trim_key(key, prefix):
            assert key.startswith(prefix)
            return key[len(prefix) :]

        lm_state_dict = {
            trim_key(k, "model.language_model."): v for k, v in state_dict.items()
        }

        encoder_state_dict = {
            trim_key(k, "encoder."): v
            for k, v in lm_state_dict.items()
            if k.startswith("encoder.")
        }
        lm_state_dict = {**lm_state_dict, "encoder": encoder_state_dict}
        self.language_model.language_model.load_state_dict(lm_state_dict, strict=True)

    def forward(
        self,
        *args,
        get_key_value=False,
        forward_method_parallel_output=None,
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


def unwrap_float16_module(module):
    if isinstance(module, Float16Module):
        return module.module
    return module


class ILQLGPT(MegatronGPTModel):
    ilql_config: ILQLConfig

    def __init__(self, ilql_config, metric_fn=None, **kwargs):
        self.ilql_config = ilql_config
        self.metric_fn = metric_fn
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

    def build_data_loader(self, dataset, collate_fn, consumed_samples=0):
        dp_rank = parallel_state.get_data_parallel_rank()
        dp_size = parallel_state.get_data_parallel_world_size()
        print(
            f"Building data loader for {dataset=} {len(dataset)=} {dp_rank=} {dp_size=}",
            file=sys.stderr,
        )
        batch_sampler = MegatronPretrainingRandomBatchSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
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

    def set_valid_dataset(self, valid_dataset, collate_fn):
        self._valid_dataset = valid_dataset
        self._valid_collate_fn = collate_fn

    # Called by superclass to build data loaders
    def setup_training_data(self, _):
        if hasattr(self, "_train_dataset"):
            self._train_dl = self.build_data_loader(
                self._train_dataset, self._train_collate_fn
            )

    def setup_validation_data(self, _):
        if hasattr(self, "_valid_dataset"):
            self._validation_dl = self.build_data_loader(
                self._valid_dataset, self._valid_collate_fn
            )

    def load_from_pretrained(self, checkpoint_dir):
        mp_rank = parallel_state.get_tensor_model_parallel_rank()
        rank_subfolder = f"mp_rank_{mp_rank:02d}"
        rank_params = Path(checkpoint_dir) / rank_subfolder / "model_weights.ckpt"
        print(f"Loading from {rank_params}")
        state_dict = torch.load(rank_params)

        unwrap_float16_module(self.model).load_state_dict(state_dict, strict=False)

    def model_provider_func(self, pre_process: bool, post_process: bool):
        """
        Model construction for Apex Pipeline Parallelism.
        Each rank will construct the model but inside the model,
        only the relevant layers for that rank should be constructed.
        On the first rank, pre_process will be True
        On the last rank, post_process will be True
        """
        gpt = super().model_provider_func(pre_process, post_process=post_process)
        # This disables post-processing the lm output to the vocab
        gpt.post_process = False
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
            return gpt

    def training_step(self, batch: ILQLBatch, batch_idx: int):
        mp_rank = parallel_state.get_tensor_model_parallel_rank()
        dp_rank = parallel_state.get_data_parallel_rank()

        """
            Our dataloaders produce a micro-batch and then we fetch
            a number of microbatches depending on the global batch size and model parallel size
            from the dataloader to produce a list of microbatches.
            Batch should be a list of microbatches and those microbatches should on CPU.
            Microbatches are then moved to GPU during the pipeline.
            The list of microbatches is then piped through the pipeline using Apex fwd/bwd functions.
        """

        # we zero grads here because we also call backward in the apex fwd/bwd functions
        self._optimizer.zero_grad()

        if parallel_state.is_pipeline_first_stage(
            ignore_virtual=True
        ) or parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            # we prepare the micro batches for the apex fwd/bwd function
            batch_for_pipeline = self.process_global_batch(batch)
        else:
            # The intermediate pipeline stages do not need any inputs from data loader
            # GPT3 uses decoder with AttnMask:causal, thus doesn't need attention_mask
            batch_for_pipeline = None

        tensor_shape = [
            self.cfg.encoder_seq_length,
            self.cfg.micro_batch_size,
            self.cfg.hidden_size,
        ]

        # handle asynchronous grad reduction
        if self.with_distributed_adam:
            if self.megatron_amp_o2:
                # copy grads to main grad
                custom_sync_context_handler = lambda: self._optimizer.no_sync(
                    greedy_grad_copy=True
                )
            else:
                # keep grad tensors around
                custom_sync_context_handler = lambda: self._optimizer.no_sync(
                    greedy_grad_copy=False
                )
        else:
            if self.megatron_amp_o2 and not self.cfg.get("sequence_parallel", False):
                custom_sync_context_handler = self._optimizer.no_sync
            else:
                # TODO: enable async grad all reduce for O1/autocast mixed precision training
                custom_sync_context_handler = None

        # run forward and backwards passes for an entire global batch
        # we do this inside training_step to support pipeline parallelism
        fwd_bwd_function = self._get_fwd_bwd_function()

        losses_reduced_per_micro_batch = fwd_bwd_function(
            forward_step_func=self.get_forward_output_and_loss_func(),
            batch=batch_for_pipeline,
            model=self.model,
            forward_only=False,
            tensor_shape=tensor_shape,
            dtype=self.autocast_dtype,
            grad_scaler=self.trainer.precision_plugin.scaler
            if self.cfg.precision == 16
            else None,
            custom_sync_context_handler=custom_sync_context_handler,
            sequence_parallel_enabled=self.cfg.get("sequence_parallel", False),
            sync_batch_comm=self.cfg.get("sync_batch_comm", False),
            num_micro_batches_with_partial_activation_checkpoints=self.cfg.get(
                "num_micro_batches_with_partial_activation_checkpoints", None
            ),
        )

        # only the last stages of the pipeline return losses
        if losses_reduced_per_micro_batch:
            # average loss across micro batches
            loss_tensors_list = [
                loss_reduced["avg"] for loss_reduced in losses_reduced_per_micro_batch
            ]
            loss_tensor = torch.concat(loss_tensors_list)
            loss_mean = loss_tensor.mean()
        else:
            loss_mean = torch.tensor(0.0).cuda()

        # when using sequence parallelism, the sequence parallel layernorm grads must be all-reduced
        if self.cfg.get("tensor_model_parallel_size", 1) > 1 and self.cfg.get(
            "sequence_parallel", False
        ):
            self.allreduce_sequence_parallel_gradients()
        if self.with_distributed_adam:
            # launch grad reductions
            # Note: grads in first pipeline stage have already been
            # reduced
            if not parallel_state.is_pipeline_first_stage():
                self.reduce_overlap_gradients()
        elif self.megatron_amp_o2:
            # when using pipeline parallelism grads must be all-reduced after the pipeline (not asynchronously)
            if self.cfg.get("pipeline_model_parallel_size", 1) > 1 or self.cfg.get(
                "sequence_parallel", False
            ):
                # main grads are stored in the MainParamsOptimizer wrapper
                self._optimizer.allreduce_main_grads()
        else:
            # async grad allreduce is not currently implemented for O1/autocasting mixed precision training
            # so we all-reduce gradients after the pipeline
            self.allreduce_gradients()  # @sangkug we think this is causing memory to blow up (hurts perf)

        if self.cfg.get("pipeline_model_parallel_size", 1) > 1:
            # when using pipeline parallelism the first and last stage must keep embeddings in sync
            self.allreduce_first_last_embeddings()

        ## logging
        # we can only log on one rank if it is rank zero so we broadcast from last rank
        # we can avoid this broadcast by updating the PTL log function to accept specific ranks
        torch.distributed.broadcast(loss_mean, get_last_rank())

        if self.cfg.precision == 16:
            loss_scale = self.trainer.precision_plugin.scaler._scale
            if loss_scale is not None:
                self.log("loss_scale", loss_scale)

        self.log("reduced_train_loss", loss_mean, prog_bar=True, rank_zero_only=True)
        lr = self._optimizer.param_groups[0]["lr"]
        self.log("lr", lr, rank_zero_only=True)
        self.log(
            "global_step", self.trainer.global_step, prog_bar=True, rank_zero_only=True
        )
        # TODO: make sure compute_consumed_samples works for pipeline parallelism
        self.log(
            "consumed_samples",
            self.compute_consumed_samples(
                self.trainer.global_step - self.init_global_step
            ),
            prog_bar=True,
            rank_zero_only=True,
        )

        if (
            self.trainer.global_step % self.ilql_config.steps_for_target_q_sync == 0
            and self.trainer.global_step > 0
        ):
            if parallel_state.is_pipeline_last_stage():
                unwrap_float16_module(self.model).other_heads.sync_target_q_heads()
                if dp_rank == 0:
                    print(
                        f"sync target q {self.trainer.global_step=} {batch_idx=} {mp_rank=}"
                    )

        return loss_mean

    def sequence_parallel_(self, enabled: bool):
        self.cfg.sequence_parallel = enabled

        def toggle_sp(m):
            if hasattr(m, "sequence_parallel"):
                m.sequence_parallel = enabled
            if hasattr(m, "sequence_parallel_enabled"):
                if hasattr(m, "input_is_parallel"):
                    m.sequence_parallel_enabled = enabled and m.input_is_parallel
                else:
                    m.sequence_parallel_enabled = enabled

        self.model.apply(toggle_sp)

    def validation_step(self, batch, batch_idx):
        if self.metric_fn is None:
            raise ValueError("Must set metric_fn to use validation")

        sp_was_enabled = self.cfg.get("sequence_parallel", False)
        if sp_was_enabled:
            self.sequence_parallel_(False)

        batch = self.process_global_batch(batch)
        gen = self.model.generate(batch, dict(max_length=20, min_length=0))
        metrics = self.metric_fn(gen["sentences"])
        for k, v in metrics.items():
            mean_v = torch.as_tensor(v).mean().item()
            self.log(f"metrics/{k}", mean_v, prog_bar=True, rank_zero_only=True)

        if sp_was_enabled:
            self.sequence_parallel_(True)

    def setup_optimizer_param_groups(self):
        # To support parameters without gradients, we need to manually
        # set the optimizer param groups to exclude them
        super().setup_optimizer_param_groups()
        param_groups = self._optimizer_param_groups

        def unfrozen_params_only(params):
            return [p for p in params if p.requires_grad]

        param_groups = [
            {**pg, "params": unfrozen_params_only(pg["params"])} for pg in param_groups
        ]

        self._optimizer_param_groups = tuple(param_groups)

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
                logits, (qs, target_qs, vs) = model_output

                if self.cfg.sequence_parallel:
                    qs, target_qs, vs = tree_map(gather_ntc, (qs, target_qs, vs))

                qs = tree_map(
                    lambda t: batched_index_select(t, batch.actions_ixs, 1),
                    qs,
                )

                target_qs = tree_map(
                    lambda t: batched_index_select(t, batch.actions_ixs, 1),
                    target_qs,
                )

                vs = batched_index_select(vs, batch.states_ixs, 1)

                model_output = (logits, (qs, target_qs, vs))
                loss_for_mb, stats = self.ilql_config.loss(model_output, batch)

                for k, v in stats.items():
                    self.log(k, v, rank_zero_only=True)

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
                    # Could potentially reduce num_valid_samples_in_microbatch and use that to
                    # aggregate instead of len(self._validation_ds)
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
        def fwd_output_only_func(
            batch: torch.Tensor,
            model,
        ):
            if batch is not None:
                batch = to_device(batch, torch.cuda.current_device(), non_blocking=True)

                extra_arg = {}

                if len(batch) == 3:
                    tokens, attention_mask, position_ids = batch
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
                    ] = inference_max_sequence_len[0].item()
                    print(
                        f"{set_inference_key_value_memory=} {inference_max_sequence_len=}"
                    )
                    # extra_arg['inference_max_sequence_len'] = inference_max_sequence_len[0].item()

                attention_mask = attention_mask[0:1]

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

                pi_adjusted = pi_beta + advantage * beta
                logits[:, -1] = pi_adjusted

                return logits, {"logits": logits}

            return model_output, ilql_postprocess

        return fwd_output_only_func

    def generate(
        self,
        inputs: Union[List[str], torch.Tensor, List[dict]],
        length_params: LengthParam,
        sampling_params: SamplingParam = None,
    ) -> OutputType:
        if sampling_params is None:
            sampling_params = {
                "use_greedy": False,
                "temperature": 0.7,
                "top_k": 0,
                "top_p": 1.0,
                "repetition_penalty": 1.0,
                "add_BOS": True,
                "all_probs": False,
                "compute_logprob": False,
            }

        return super().generate(inputs, length_params, sampling_params)


class HydraWithValueHeadGPT(MegatronGPTModel):
    ppo_config: PPOConfig

    def __init__(self, ppo_config, **kwargs):
        self.ppo_config = ppo_config
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
