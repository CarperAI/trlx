# Extensible version of the GPT model
import sys
from contextlib import contextmanager
from functools import partial
from math import ceil, sqrt
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.distributed
import torch.nn as nn
import wandb
from apex.transformer import parallel_state, tensor_parallel
from apex.transformer.pipeline_parallel.utils import _reconfigure_microbatch_calculator
from apex.transformer.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
)
from einops import rearrange
from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_samplers import (
    MegatronPretrainingBatchSampler,
)
from nemo.collections.nlp.models.language_modeling.megatron.gpt_model import (
    post_language_model_processing,
)
from nemo.collections.nlp.models.language_modeling.megatron_base_model import (
    MegatronBaseModel,
)
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import (
    MegatronGPTModel,
)
from nemo.collections.nlp.modules.common.megatron.attention import ParallelAttention
from nemo.collections.nlp.modules.common.megatron.module import (
    Float16Module,
    MegatronModule,
)
from nemo.collections.nlp.modules.common.megatron.transformer import (
    ParallelTransformerLayer,
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
from nemo.utils import AppState

from trlx.data.ilql_types import unflatten_dataclass
from trlx.data.ppo_types import PPORLBatch
from trlx.models.modeling_ppo import PPOConfig
from trlx.utils import to_device, tree_map
from trlx.utils.modeling import logprobs_of_labels, whiten

# Track a per dp rank RNG to sample different rollouts
# per dp rank
_PER_DP_RANK_RNG = "per-data-parallel-rank-rng"


def patch_attention_for_llama(m):
    if isinstance(m, ParallelAttention):
        m.megatron_legacy = True


class ParallelLinear(nn.Module):
    """Linear layer parallelized over the longer dimension."""

    def __init__(
        self,
        in_size: int,
        out_size: int,
        init_method=None,
        use_cpu_initialization=False,
        bias=True,
        sequence_parallel=False,
        gradient_accumulation_fusion=False,
        gather_output=True,
        input_is_parallel=False,
        dtype=torch.bfloat16,
    ):
        super().__init__()

        if init_method is None:
            init_method = partial(nn.init.uniform_, a=-sqrt(1.0 / in_size), b=sqrt(1.0 / in_size))

        no_async_tensor_model_parallel_allreduce = (
            parallel_state.get_tensor_model_parallel_world_size() == 1 or sequence_parallel
        )

        # Fork TP rng so each TP rank has different random weights
        with tensor_parallel.random.get_cuda_rng_tracker().fork():
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
                    params_dtype=dtype,
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
                    params_dtype=dtype,
                )
            self.layer.bias.data.uniform_(-sqrt(1.0 / out_size), sqrt(1.0 / out_size))

    def forward(self, x):
        output, bias = self.layer(x)
        if bias is not None:
            return output + bias
        return output


def make_parallel_head(n_embd: int, out: int, sequence_parallel=False, dtype=torch.bfloat16) -> nn.Sequential:
    """Returns a generic sequential model parallel MLP head."""
    parallel_intermediate = out < (n_embd * 2)
    return nn.Sequential(
        ParallelLinear(
            n_embd,
            n_embd * 2,
            sequence_parallel=sequence_parallel,
            gather_output=not parallel_intermediate,
            dtype=dtype,
        ),
        nn.ReLU(),
        ParallelLinear(
            n_embd * 2,
            out,
            sequence_parallel=sequence_parallel,
            input_is_parallel=parallel_intermediate,
            dtype=dtype,
        ),
    )


class ValueHead(nn.Module):
    def __init__(self, hidden_size: int, sequence_parallel=False, dtype=torch.bfloat16):
        super().__init__()
        self.hidden_size = hidden_size
        self.v_head = make_parallel_head(hidden_size, 1, sequence_parallel=sequence_parallel, dtype=dtype)

        self.sequence_parallel = sequence_parallel

    def forward(self, x):
        vs = self.v_head(x)
        if self.sequence_parallel:
            vs = gather_from_sequence_parallel_region(vs, to_model_parallel=False)
        return rearrange(vs, "T N 1 -> N T")


class RefLMHeads(MegatronModule):
    def __init__(self, language_model, other_heads, build_reference_model=True):
        super().__init__()
        # must be this attribute name
        self.pre_process = language_model.pre_process
        self.post_process = language_model.post_process

        # nest GPTModel
        self._lm = language_model
        # MegatronGPTModel expects this attribute so we un-nest it
        self.language_model = language_model.language_model
        self.build_reference_model = build_reference_model
        if build_reference_model:
            for p in self.language_model.parameters():
                if p.requires_grad:
                    p._ref_data = p.data.clone().cpu().pin_memory()
                    p._policy_data = p.data

            self.reference_model_offloaded = True
        else:
            self.reference_model_offloaded = True

        self.other_heads = other_heads
        if hasattr(language_model, "output_layer"):
            self.output_layer = self._lm.language_model.output_layer
            self.word_embeddings = self.output_layer.weight
        else:
            if hasattr(language_model, "word_embeddings"):
                self.word_embeddings = language_model.word_embeddings
            self.output_layer = None

    # The tensor from the previous pipeline rank arrives via this method
    def set_input_tensor(self, input_tensor):
        self._lm.set_input_tensor(input_tensor)

    def word_embeddings_weight(self):
        return self._lm.word_embeddings_weight()

    def load_state_dict(self, lm_state_dict, strict=True):
        """Load GPTModel state dict."""
        self.language_model.load_state_dict(lm_state_dict, strict=strict)

        if "output_layer.weight" in lm_state_dict:
            dtype = lm_state_dict["output_layer.weight"].dtype
            device = self.language_model.output_layer.weight.device
            params = torch.nn.Parameter(
                lm_state_dict["output_layer.weight"].to(device, dtype=dtype), requires_grad=True
            )
            self.language_model.output_layer.weight = params
            print("Loaded output_layer.weight from lm_state_dict")

        if self.build_reference_model:
            for p in self.language_model.parameters():
                if p.requires_grad:
                    p._ref_data = p.data.clone().cpu().pin_memory()
                    p._policy_data = p.data

    def pretrained_state_dict(self):
        """Load GPTModel state dict."""
        return self._lm.state_dict()

    def offload_reference_model(self):
        """Move reference model to CPU."""
        if self.reference_model_offloaded or not self.build_reference_model:
            return
        for p in self.language_model.parameters():
            if p.requires_grad:
                p.data = p._policy_data.to(torch.cuda.current_device())
        self.reference_model_offloaded = True

    def offload_policy_model(self):
        """Move language model to CPU."""
        if self.reference_model_offloaded:
            for p in self.language_model.parameters():
                if p.requires_grad:
                    p._policy_data = p.data.to("cpu", non_blocking=True)
                    p.data = p._ref_data.to(torch.cuda.current_device(), non_blocking=True)
            self.reference_model_offloaded = False

    def forward(
        self,
        *args,
        get_key_value=False,
        forward_method_parallel_output=None,
        run_policy_model=True,
        run_reference_model=False,
        run_value_head=False,
        **kwargs,
    ):
        if hasattr(self._lm.language_model, "output_layer"):
            logit_weights = self._lm.language_model.output_layer.weight
        else:
            logit_weights = self._lm.word_embeddings_weight()

        if run_policy_model:
            self.offload_reference_model()

            lm_output = self._lm(*args, get_key_value=get_key_value, **kwargs)
            logits = post_language_model_processing(
                lm_output,
                labels=None,
                logit_weights=logit_weights,
                get_key_value=get_key_value,
                parallel_output=False,  # self.language_model.parallel_output,
                forward_method_parallel_output=forward_method_parallel_output,
                fp16_lm_cross_entropy=self._lm.fp16_lm_cross_entropy,
                return_logits=True,
                sequence_parallel=self._lm.sequence_parallel,
                gradient_accumulation_fusion=self._lm.gradient_accumulation_fusion,
            )

            if get_key_value:
                logits, presents = logits
                lm_output, lm_output_presents = lm_output

            if run_value_head:
                heads_output = self.other_heads(lm_output)
            else:
                heads_output = None
        else:
            logits = None
            heads_output = None

        if run_reference_model:
            self.offload_policy_model()
            ref_lm_output = self._lm(*args, get_key_value=get_key_value, **kwargs)
            ref_logits = post_language_model_processing(
                ref_lm_output,
                labels=None,
                logit_weights=logit_weights,
                get_key_value=get_key_value,
                parallel_output=False,  # self.reference_model.model.parallel_output,
                forward_method_parallel_output=forward_method_parallel_output,
                fp16_lm_cross_entropy=self._lm.fp16_lm_cross_entropy,
                return_logits=True,
                sequence_parallel=self._lm.sequence_parallel,
                gradient_accumulation_fusion=self._lm.gradient_accumulation_fusion,
            )

            if get_key_value:
                ref_logits, ref_presents = ref_logits
                ref_lm_output, ref_lm_output_presents = ref_lm_output

            return logits, heads_output, ref_logits

        return logits, heads_output, None


def unwrap_float16_module(module):
    if isinstance(module, Float16Module):
        return module.module
    return module


def reshard_for_pipeline_parallelism(num_layers, state_dict):
    """Filter out the layers that are not in the current pipeline stage
    and shift the layer ids to match the local stage layer ids."""
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    pp_size = parallel_state.get_pipeline_model_parallel_world_size()

    stage_layers = num_layers // pp_size
    pp_offset = pp_rank * stage_layers

    encoder_layers_key = "model.language_model.encoder.layers."

    def filter_in_pp_rank(key):
        if key.startswith(encoder_layers_key):
            layer_idx = int(key.split(".")[4])
            return pp_offset <= layer_idx < (pp_offset + stage_layers)
        elif key.startswith("model.language_model.encoder.final_layernorm") and not pp_rank == (pp_size - 1):
            return False
        else:
            return True

    def shift_layer_idx(key):
        """If the key is for a transformer layer, shift down the layer index to select the
        correct layer for this pipeline stage."""
        if key.startswith(encoder_layers_key):
            layer_idx = int(key.split(".")[4])
            return f"{encoder_layers_key}{str(layer_idx - pp_offset)}.{'.'.join(key.split('.')[5:])}"
        else:
            return key

    state_dict = {shift_layer_idx(k): v for k, v in state_dict.items() if filter_in_pp_rank(k)}

    return state_dict


class PPOGPT(MegatronGPTModel):
    ppo_config: PPOConfig

    def __init__(
        self,
        ppo_config,
        metric_fn=None,
        stop_sequences=(),
        num_layers_unfrozen=None,
        build_reference_model=True,
        **kwargs,
    ):
        self.ppo_config = ppo_config
        self.metric_fn = metric_fn
        self.stop_sequences = stop_sequences
        if num_layers_unfrozen == -1:
            self.num_layers_unfrozen = None
        else:
            self.num_layers_unfrozen = num_layers_unfrozen
        self.build_reference_model = build_reference_model

        super().__init__(**kwargs)
        if len(list(self.parameters())) == 0:
            raise ValueError("No parameters in model")

        self._ori_activations_checkpoint_granularity = self.cfg.get("activations_checkpoint_granularity", None)
        self._ori_activations_checkpoint_method = self.cfg.get("activations_checkpoint_method", None)
        self._ori_activations_checkpoint_num_layers = self.cfg.get("activations_checkpoint_num_layers", None)

    def maybe_initalize_per_dp_rng(self):
        tracker = tensor_parallel.random.get_cuda_rng_tracker()
        if _PER_DP_RANK_RNG in tracker.get_states():
            return
        seed = torch.cuda.initial_seed()
        if parallel_state._DATA_PARALLEL_GROUP is not None:
            dp_rank = parallel_state.get_data_parallel_rank()
        else:
            dp_rank = 0
        tracker.add(_PER_DP_RANK_RNG, seed * (1 + dp_rank))

    @classmethod
    def list_available_models(cls) -> Optional[Mapping[str, str]]:
        return None

    def build_train_valid_test_datasets(self):
        pass

    def build_data_loader(self, dataset, collate_fn, consumed_samples=0):
        dp_rank = parallel_state.get_data_parallel_rank()
        dp_size = parallel_state.get_data_parallel_world_size()
        print(
            f"Building data loader for {type(dataset)=} {len(dataset)=} {dp_rank=} {dp_size=}",
            file=sys.stderr,
        )
        batch_sampler = MegatronPretrainingBatchSampler(
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
            # For some reason this causes a crash when using >0 workers
            # with grad accumulation > 1
            num_workers=0,
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
            self._train_dl = self.build_data_loader(self._train_dataset, self._train_collate_fn)

    def setup_validation_data(self, _):
        if hasattr(self, "_valid_dataset"):
            self._validation_dl = self.build_data_loader(self._valid_dataset, self._valid_collate_fn)

    def save_pretrained(self, checkpoint_dir):
        torch.cuda.synchronize()
        torch.distributed.barrier()
        assert (
            parallel_state.get_pipeline_model_parallel_world_size() == 1
        ), "Pipeline parallelism not supported for saving"

        if parallel_state.get_data_parallel_rank() == 0:
            state_dict = unwrap_float16_module(self.model).pretrained_state_dict()
            state_dict = {f"model.{k}": v for k, v in state_dict.items()}

            mp_rank = parallel_state.get_tensor_model_parallel_rank()
            mp_world = parallel_state.get_tensor_model_parallel_world_size()

            if mp_world > 1:
                rank_params = Path(checkpoint_dir) / f"mp_rank_{mp_rank:02d}" / "model_weights.ckpt"
            else:
                rank_params = Path(checkpoint_dir) / "model_weights.ckpt"

            rank_params.parent.mkdir(parents=True, exist_ok=True)

            print(f"Saving to {rank_params}")
            torch.save(state_dict, rank_params)

    def load_from_pretrained(self, checkpoint_dir):
        mp_rank = parallel_state.get_tensor_model_parallel_rank()
        mp_world = parallel_state.get_tensor_model_parallel_world_size()

        if mp_world > 1:
            rank_subfolder = f"mp_rank_{mp_rank:02d}"
            rank_params = Path(checkpoint_dir) / rank_subfolder / "model_weights.ckpt"
        else:
            rank_params = Path(checkpoint_dir) / "model_weights.ckpt"

        print(f"Loading from {rank_params}")
        state_dict = torch.load(rank_params)

        state_dict = reshard_for_pipeline_parallelism(self.cfg.num_layers, state_dict)

        def trim_key(key, prefix):
            assert key.startswith(prefix), f"key {key} in state_dict does not start with {prefix}"
            return key[len(prefix) :]

        lm_state_dict = {trim_key(k, "model.language_model."): v for k, v in state_dict.items()}

        encoder_state_dict = {trim_key(k, "encoder."): v for k, v in lm_state_dict.items() if k.startswith("encoder.")}

        lm_state_dict = {**lm_state_dict, "encoder": encoder_state_dict}

        unwrap_float16_module(self.model).load_state_dict(lm_state_dict, strict=True)
        print(f"Loaded from pretrained {rank_params}")

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

        # Unfreeze only last N layers if specified
        if self.num_layers_unfrozen is not None:

            def freeze_layers(m):
                if isinstance(m, ParallelTransformerLayer):
                    if m.layer_number < (self.cfg.num_layers - self.num_layers_unfrozen):
                        m.eval()
                        for p in m.parameters():
                            p.requires_grad_(False)
                    else:
                        for p in m.parameters():
                            p.requires_grad_(True)

            for p in gpt.parameters():
                p.requires_grad_(False)
            gpt.language_model.apply(freeze_layers)

        if self.cfg.get("megatron_legacy", False):
            gpt.apply(patch_attention_for_llama)
        # If running on the last pipeline stage, add the PPO value head and hydra reference model
        if post_process:
            value_head = ValueHead(self.cfg.hidden_size, self.cfg.sequence_parallel)

            return RefLMHeads(gpt, value_head, build_reference_model=self.build_reference_model)
        else:
            return gpt

    def configure_optimizers(self):  # noqa: C901
        if self.with_distributed_adam:
            # Disable overlapped grad sync for embedding grad when
            # pipeline parallelism is enabled
            if parallel_state.get_pipeline_model_parallel_world_size() > 1:
                if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                    if isinstance(self.model, list):
                        module = self.model[0]  # only the first virtual rank has the embeddings
                    else:
                        module = self.model
                    if module.share_token_embeddings:
                        param = module.word_embeddings_weight()
                        param._disable_greedy_grad_copy = not self.megatron_amp_o2
                        param._disable_overlap_grad_sync = True
                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    if isinstance(self.model, list):
                        module = self.model[-1]  # only the last virtual rank has the embeddings
                    else:
                        module = self.model
                    if module.share_token_embeddings:
                        param = module.word_embeddings_weight()
                        param._disable_greedy_grad_copy = not self.megatron_amp_o2
                        param._disable_overlap_grad_sync = True

            # Disable overlapped grad sync for layer norm grads when
            # sequence parallelism is enabled
            for param in self.parameters():
                if getattr(param, "sequence_parallel_enabled", False):
                    param._disable_greedy_grad_copy = not self.megatron_amp_o2
                    param._disable_overlap_grad_sync = True

            # Initialize parameter buckets for overlapped grad and param syncs
            # Note: Params with disabled overlapping are put in the
            # last param bucket
            buckets = []
            if self.cfg.get("virtual_pipeline_model_parallel_size", None) is not None:
                # Initialize a bucket for each virtual pipeline stage
                for module in self.model:
                    if isinstance(module, Float16Module):
                        module = module.module
                    stage_bucket = []
                    for layer in module.language_model.encoder.layers:
                        stage_bucket.extend(
                            p
                            for p in layer.parameters()
                            if not getattr(p, "_disable_overlap_grad_sync", False) and p.requires_grad
                        )
                    buckets.append(stage_bucket)
            else:
                # Initialize a bucket for each Transformer layer
                modules = self.model if isinstance(self.model, list) else [self.model]
                for module in modules:
                    if isinstance(module, Float16Module):
                        module = module.module
                    for layer in module.language_model.encoder.layers:
                        buckets.append(
                            [
                                p
                                for p in layer.parameters()
                                if not getattr(p, "_disable_overlap_grad_sync", False) and p.requires_grad
                            ]
                        )

            buckets.reverse()
            used_params = set()
            for bucket in buckets:
                used_params.update(bucket)
            buckets[-1].extend(p for p in self.parameters() if p not in used_params)

            total_numel = sum(sum(p.numel() for p in bucket) for bucket in buckets)
            print(f"Total number of optimizable parameters: {total_numel:,}")
            self.distributed_adam_buckets = buckets
        return MegatronBaseModel.configure_optimizers(self)

    def _append_sequence_parallel_module_grads(self, module, grads):
        """Helper method for allreduce_sequence_parallel_gradients"""

        for param in module.parameters():
            if not param.requires_grad:
                continue
            sequence_parallel_param = getattr(param, "sequence_parallel", False)
            # grad can be None when performing PeFT training.
            if sequence_parallel_param and param.requires_grad:
                if self.megatron_amp_o2:
                    grad = param.main_grad
                else:
                    grad = param.grad
                grads.append(grad.data)

    def allreduce_sequence_parallel_gradients(self):
        """All-reduce layernorm parameters across model parallel nodes when sequence parallelism is used.
        Modified from megatron-lm:
        https://gitlab-master.nvidia.com/ADLR/megatron-lm/-/blob/3f91f09bb2ab32f9904b47f46f19d2fc3f518ed8/megatron/training.py#L425
        """

        grads = []
        if isinstance(self.model, list):
            for module in self.model:
                self._append_sequence_parallel_module_grads(module, grads)
        else:
            self._append_sequence_parallel_module_grads(self.model, grads)

        if len(grads) == 0:
            return
        coalesced = torch._utils._flatten_dense_tensors(grads)
        torch.distributed.all_reduce(coalesced, group=parallel_state.get_tensor_model_parallel_group())
        for buf, synced in zip(grads, torch._utils._unflatten_dense_tensors(coalesced, grads)):
            buf.copy_(synced)

    def setup_optimizer_param_groups(self):
        self._optimizer_param_groups = ({"params": list(self.parameters())},)

    # Adapted from NeMo
    # https://github.com/NVIDIA/NeMo/blob/r1.13.0/nemo/collections/nlp/models/language_modeling/megatron_gpt_model.py#L259
    def training_step(self, batch: PPORLBatch, batch_idx: int):  # noqa: C901
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

        if parallel_state.is_pipeline_first_stage(ignore_virtual=True) or parallel_state.is_pipeline_last_stage(
            ignore_virtual=True
        ):
            # we prepare the micro batches for the apex fwd/bwd function
            batch_for_pipeline = batch
        else:
            # The intermediate pipeline stages do not need any inputs from data loader
            # GPT3 uses decoder with AttnMask:causal, thus doesn't need attention_mask
            batch_for_pipeline = None

        # Pipeline stages will transfer this shape tensor to and from the
        # previous and next stages
        # The model must output a tensor of this shape if not the last pipeline
        # stage. The model is given input of this shape if not the first pipeline
        # stage via .set_input_tensor
        tensor_shape = [
            batch_for_pipeline[0].shape[1],  # self.cfg.encoder_seq_length,
            self.cfg.micro_batch_size,
            self.cfg.hidden_size,
        ]

        # handle asynchronous grad reduction
        custom_sync_context_handler = None
        custom_grad_sync_func = None
        custom_param_sync_func = None
        if self.with_distributed_adam:
            if self.megatron_amp_o2:
                # copy grads to main grad
                def custom_sync_context_handler():
                    return self._optimizer.no_sync(greedy_grad_copy=True)

            else:
                # keep grad tensors around
                def custom_sync_context_handler():
                    return self._optimizer.no_sync(greedy_grad_copy=False)

            custom_grad_sync_func = self.reduce_overlap_gradients
            custom_param_sync_func = self.sync_overlap_parameters
        else:
            if self.megatron_amp_o2 and not self.cfg.get("sequence_parallel", False):
                custom_sync_context_handler = self._optimizer.no_sync
            else:
                # TODO: enable async grad all reduce for O1/autocast mixed precision training
                custom_sync_context_handler = None

        # run forward and backwards passes for an entire global batch
        # we do this inside training_step to support pipeline parallelism
        # This gets the correct fwd/bwd pipeline step depending on the pipeline
        # parallelism configuration
        fwd_bwd_function = self._get_fwd_bwd_function()

        last_stage_output = fwd_bwd_function(
            forward_step_func=self.get_forward_output_and_loss_func(),
            batch=batch_for_pipeline,
            model=self.model,
            forward_only=False,
            tensor_shape=tensor_shape,
            dtype=self.autocast_dtype,
            grad_scaler=self.trainer.precision_plugin.scaler if self.cfg.precision == 16 else None,
            custom_sync_context_handler=custom_sync_context_handler,
            custom_grad_sync_func=custom_grad_sync_func,
            custom_param_sync_func=custom_param_sync_func,
            sequence_parallel_enabled=self.cfg.get("sequence_parallel", False),
            sync_batch_comm=self.cfg.get("sync_batch_comm", False),
            num_micro_batches_with_partial_activation_checkpoints=self.cfg.get(
                "num_micro_batches_with_partial_activation_checkpoints", None
            ),
        )

        # only the last stages of the pipeline return losses
        if last_stage_output:
            # average loss across micro batches
            outputs = {k: [output[k] for output in last_stage_output] for k in last_stage_output[0].keys()}
            outputs = {k: torch.concat([torch.as_tensor(vi).unsqueeze(0) for vi in v]) for k, v in outputs.items()}

            mean_outputs = {k: v.mean() for k, v in outputs.items()}
            loss_mean = mean_outputs["avg_loss"]
        else:
            mean_outputs = {}
            loss_mean = torch.tensor(0.0).cuda()

        # when using sequence parallelism, the sequence parallel layernorm grads must be all-reduced
        if self.cfg.get("tensor_model_parallel_size", 1) > 1 and self.cfg.get("sequence_parallel", False):
            self.allreduce_sequence_parallel_gradients()
        if self.with_distributed_adam:
            # launch grad reductions
            # Note: grads in first pipeline stage have already been
            # reduced
            if not parallel_state.is_pipeline_first_stage():
                self.reduce_overlap_gradients()
            self._optimizer._finish_bucket_grad_sync()
        elif self.megatron_amp_o2:
            # when using pipeline parallelism grads must be all-reduced after the pipeline (not asynchronously)
            if self.cfg.get("pipeline_model_parallel_size", 1) > 1 or self.cfg.get("sequence_parallel", False):
                # main grads are stored in the MainParamsOptimizer wrapper
                self._optimizer.allreduce_main_grads()
        else:
            # async grad allreduce is not currently implemented for O1/autocasting mixed precision training
            # so we all-reduce gradients after the pipeline
            self.allreduce_gradients()  # @sangkug we think this is causing memory to blow up (hurts perf)

        if self.cfg.get("pipeline_model_parallel_size", 1) > 1 and self.cfg.get(
            "share_embeddings_and_output_weights", True
        ):
            # when using pipeline parallelism the first and last stage must keep embeddings in sync
            self.allreduce_first_last_embeddings()

        # we can only log on one rank if it is rank zero so we broadcast from last rank
        # we can avoid this broadcast by updating the PTL log function to accept specific ranks
        torch.distributed.broadcast(loss_mean, get_last_rank())

        if torch.distributed.get_rank() == 0:
            lr = self._optimizer.param_groups[0]["lr"]
            logs = dict(mean_outputs, reduced_train_loss=loss_mean, global_step=batch_idx, lr=lr)

            if self.cfg.precision == 16:
                loss_scale = self.trainer.precision_plugin.scaler._scale
                if loss_scale is not None:
                    logs["loss_scale"] = loss_scale
            logs["trainer/global_step"] = batch_idx
            wandb.log(logs, step=batch_idx)

        return loss_mean

    def activation_checkpointing_(self, enable: bool):
        def toggle_checkpointing(module):
            if hasattr(module, "activations_checkpoint_granularity"):
                if enable:
                    module.activations_checkpoint_granularity = self._ori_activations_checkpoint_granularity
                else:
                    module.activations_checkpoint_granularity = None

            if hasattr(module, "activations_checkpoint_method"):
                if enable:
                    module.activations_checkpoint_method = self._ori_activations_checkpoint_method
                else:
                    module.activations_checkpoint_method = None

            if hasattr(module, "activations_checkpoint_num_layers"):
                if enable:
                    module.activations_checkpoint_num_layers = self._ori_activations_checkpoint_num_layers
                else:
                    module.activations_checkpoint_num_layers = None

        self.model.apply(toggle_checkpointing)

        if enable:
            self.cfg.activations_checkpoint_granularity = self._ori_activations_checkpoint_granularity
            self.cfg.activations_checkpoint_method = self._ori_activations_checkpoint_method
            self.cfg.activations_checkpoint_num_layers = self._ori_activations_checkpoint_num_layers
        else:
            self.cfg.activations_checkpoint_granularity = None
            self.cfg.activations_checkpoint_method = None
            self.cfg.activations_checkpoint_num_layers = None

    # TODO: replace this with less magical code
    def sequence_parallel_(self, enabled: bool):
        self.cfg.sequence_parallel = enabled

        def toggle_sp(m):
            if hasattr(m, "sequence_parallel"):
                m.sequence_parallel = enabled

            # for the Row/ColumnParallelLinear layers
            if hasattr(m, "sequence_parallel_enabled"):
                if hasattr(m, "input_is_parallel"):
                    m.sequence_parallel_enabled = enabled and m.input_is_parallel
                elif hasattr(m, "gather_output"):
                    m.sequence_parallel_enabled = enabled and not m.gather_output
                else:
                    m.sequence_parallel_enabled = enabled

        self.model.apply(toggle_sp)

    @contextmanager
    def inference_mode(self):
        sp_was_enabled = self.cfg.get("sequence_parallel", False)
        if sp_was_enabled:
            self.sequence_parallel_(False)

        activations_checkpointing_was_enabled = self.cfg.get("activations_checkpoint_granularity", None) is not None

        if activations_checkpointing_was_enabled:
            self.activation_checkpointing_(False)

        was_training = self.model.training
        self.model.eval()

        try:
            yield
        finally:
            if sp_was_enabled:
                self.sequence_parallel_(True)

            if activations_checkpointing_was_enabled:
                self.activation_checkpointing_(True)

            if was_training:
                self.model.train()

            _reconfigure_microbatch_calculator(
                rank=AppState().global_rank,
                rampup_batch_size=None,
                global_batch_size=self.cfg.global_batch_size,
                micro_batch_size=self.cfg.micro_batch_size,
                data_parallel_size=AppState().data_parallel_size,
            )

    def validation_step(self, batch: Tuple[List[int], List[int]], batch_idx: int):
        if self.metric_fn is None:
            raise ValueError("Must set metric_fn to use validation")

        input_ids, lengths = batch
        input_ids, lengths = torch.as_tensor(input_ids), torch.as_tensor(lengths)

        input_ids, lengths = to_device((input_ids, lengths), torch.cuda.current_device(), non_blocking=True)

        max_new_tokens = self.ppo_config.gen_kwargs.get("max_new_tokens", 64)

        gen = self.generate((input_ids, lengths), dict(max_length=max_new_tokens, min_length=0))

        metrics = self.metric_fn(samples=gen["sentences"], prompts=gen["prompts"], outputs=gen["responses"])

        metric_keys, metric_values = zip(*metrics.items())

        columns = ["prompts", "responses", *metric_keys]
        rows = list(zip(gen["prompts"], gen["responses"], *metric_values))

        metrics = {f"{k}": torch.as_tensor(v) for k, v in metrics.items()}

        return metrics, (rows, columns)

    def validation_epoch_end(self, outputs: List[Tuple[dict, Tuple[List[str], List[str]]]], batch_idx: int):
        self.free_kv_cache()

        metrics, tables = zip(*outputs)
        _, columns = tables[0]
        rows = [r for trows, _ in tables for r in trows]

        dp_world = parallel_state.get_data_parallel_world_size()
        dp_group = parallel_state.get_data_parallel_group()

        global_rows = [None for _ in range(dp_world)]
        torch.distributed.all_gather_object(global_rows, rows, group=dp_group)
        rows = [r for rlist in global_rows for r in rlist]

        table = wandb.Table(data=rows, columns=columns)

        outputs_soa = {k: torch.cat([d[k] for d in metrics]).cuda() for k in metrics[0].keys()}
        outputs_gathered = {}
        for k, v in outputs_soa.items():
            gathered = torch.empty((v.shape[0] * dp_world), dtype=v.dtype, device=v.device)
            torch.distributed.all_gather_into_tensor(gathered, v, group=dp_group)
            outputs_gathered[k] = gathered

        metrics = {f"val_metrics/{k}": v.mean() for k, v in outputs_gathered.items()}
        metrics = {**metrics, **{f"val_metrics_distributions/{k}": v for k, v in outputs_gathered.items()}}
        metrics["trainer/global_step"] = batch_idx
        metrics["val_samples"] = table

        if torch.distributed.get_rank() == 0:
            wandb.log(metrics)

        return metrics

    # Need to override this otherwise distributed fused adam won't work
    # with frozen layers
    def parameters(self):
        return (p for p in unwrap_float16_module(self.model).language_model.parameters() if p.requires_grad)

    def offload_reference_model(self):
        unwrap_float16_module(self.model).offload_reference_model()

    def free_kv_cache(self):
        def clear(m):
            if hasattr(m, "inference_key_memory"):
                m.inference_key_memory = None
                m.inference_value_memory = None

        unwrap_float16_module(self.model).apply(clear)

    def get_forward_output_and_loss_func(self, validation_step=False):
        def fwd_output_and_loss_func(flat_batch: List[torch.Tensor], model, checkpoint_activations_all_layers=None):
            # On first and last pipeline stages, the input data is passed in
            if flat_batch is not None:
                batch = unflatten_dataclass(PPORLBatch)(flat_batch)
                batch = to_device(batch, torch.cuda.current_device(), non_blocking=True)

                inputs = torch.cat((batch.query_tensors, batch.response_tensors), dim=1)
                pad_by = ceil(inputs.shape[1] / 8) * 8 - inputs.shape[1]
                inputs = torch.nn.functional.pad(inputs, (0, pad_by), value=self.tokenizer.eos_id)

                # Note that the inputs to this are left padded as well as right padded
                # Due to combining the left padded queries and right padded responses
                # `get_ltor_masks_and_position_ids` will 0 mask all tokens == eos_token_id
                (
                    attention_mask,
                    loss_mask,
                    position_ids,
                ) = get_ltor_masks_and_position_ids(
                    data=inputs,
                    eod_token=self.tokenizer.eos_id,
                    reset_position_ids=True,
                    reset_attention_mask=True,
                    eod_mask_loss=True,
                )

                if isinstance(unwrap_float16_module(self.model), RefLMHeads):
                    extra_args = dict(run_reference_model=False, run_value_head=True)
                else:
                    extra_args = dict()

                model_output = model(
                    input_ids=inputs,
                    position_ids=position_ids.long(),
                    attention_mask=attention_mask,
                    **extra_args,
                )
            else:
                # In-between stages are given data via the pipeline engine
                # Still need to specify thes arguments to avoid errors
                model_output = model(input_ids=None, position_ids=None, attention_mask=None)

            def loss_func(model_output):
                # # TODO: implement this in a sequence parallel way
                logits, vs, _ = model_output

                response_length = batch.rewards.shape[1]

                start = batch.query_tensors.shape[1]
                end = start + response_length

                label_logprobs = logprobs_of_labels(logits[:, :-1, :], inputs[:, 1:])
                label_logprobs = label_logprobs[:, start:end]

                advantages, returns = self.ppo_config.get_advantages_and_returns(
                    batch.values, batch.rewards, response_length, use_whitening=False
                )

                advantages = whiten(advantages, group=parallel_state.get_data_parallel_group())

                values_pred = vs[:, :-1][:, start:end]

                loss_for_mb, stats = self.ppo_config.loss(
                    logprobs=label_logprobs,
                    values=values_pred,
                    old_logprobs=batch.logprobs,
                    old_values=batch.values,
                    advantages=advantages,
                    returns=returns,
                    mask=loss_mask[:, start:end],
                )

                reduced_loss = average_losses_across_data_parallel_group([loss_for_mb])

                # Needed for async grad allreduce
                torch.cuda.synchronize()

                return loss_for_mb, {"avg_loss": reduced_loss, **stats}

            return model_output, loss_func

        return fwd_output_and_loss_func

    def get_forward_output_only_func(
        self,
        run_policy_model=True,
        run_reference_model=False,
        run_value_head=False,
        compute_logprobs=False,
    ):
        def fwd_output_only_func(
            batch: torch.Tensor,
            model,
        ):
            if batch is not None:
                batch = to_device(batch, torch.cuda.current_device(), non_blocking=True)

                if parallel_state.is_pipeline_last_stage():
                    extra_arg = dict(
                        run_reference_model=run_reference_model,
                        run_value_head=run_value_head,
                        run_policy_model=run_policy_model,
                    )
                else:
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

                    extra_arg["set_inference_key_value_memory"] = set_inference_key_value_memory[0].item()
                    extra_arg["inference_max_sequence_len"] = inference_max_sequence_len[0].item()

                model_output = model(
                    input_ids=tokens,
                    position_ids=position_ids.long(),
                    attention_mask=attention_mask,
                    **extra_arg,
                )
            else:
                model_output = model(input_ids=None, position_ids=None, attention_mask=None)

            def ppo_postprocess(model_output):
                model_output = tree_map(lambda t: t.float() if t is not None else t, model_output)
                logits, values, ref_logits = model_output

                # Logits can become large so best to pick out the logprobs per microbatch here
                # to save memory

                if run_policy_model and compute_logprobs:
                    logprobs = logprobs_of_labels(logits[:, :-1, :], tokens[:, 1:])
                    return logprobs, dict(logprobs=logprobs, values=values)

                if run_reference_model and compute_logprobs:
                    ref_logprobs = logprobs_of_labels(ref_logits[:, :-1, :], tokens[:, 1:])
                    return ref_logprobs, dict(ref_logprobs=ref_logprobs)

                return logits, {"logits": logits, "values": values, "ref_logits": ref_logits}

            return model_output, ppo_postprocess

        return fwd_output_only_func

    def infer_logprobs_and_values(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        fwd_bwd_func = self._get_fwd_bwd_function()
        infer_policy = self.get_forward_output_only_func(
            run_policy_model=True, run_reference_model=False, run_value_head=True, compute_logprobs=True
        )
        infer_ref = self.get_forward_output_only_func(
            run_policy_model=False, run_reference_model=True, run_value_head=False, compute_logprobs=True
        )

        self.model.eval()

        bs, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        # Using get_ltor_masks_and_position_ids
        attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
            data=input_ids,
            eod_token=self.tokenizer.eos_id,
            reset_position_ids=True,
            reset_attention_mask=True,
            eod_mask_loss=True,
        )

        with self.inference_mode():
            _reconfigure_microbatch_calculator(
                rank=AppState().global_rank,
                rampup_batch_size=None,
                global_batch_size=input_ids.shape[0] * AppState().data_parallel_size,
                micro_batch_size=self.cfg.micro_batch_size,
                data_parallel_size=AppState().data_parallel_size,
            )

            with torch.no_grad():
                policy_stage_output = fwd_bwd_func(
                    forward_step_func=infer_policy,
                    batch=[input_ids, attention_mask, position_ids],
                    model=self.model,
                    forward_only=True,
                    dtype=self.autocast_dtype,
                    sequence_parallel_enabled=self.cfg.get("sequence_parallel", False),
                    sync_batch_comm=self.cfg.get("sync_batch_comm", False),
                )
                ref_stage_output = fwd_bwd_func(
                    forward_step_func=infer_ref,
                    batch=[input_ids, attention_mask, position_ids],
                    model=self.model,
                    forward_only=True,
                    dtype=self.autocast_dtype,
                    sequence_parallel_enabled=self.cfg.get("sequence_parallel", False),
                    sync_batch_comm=self.cfg.get("sync_batch_comm", False),
                )
                if policy_stage_output is not None and ref_stage_output is not None:
                    stage_output = [{**policy, **ref} for policy, ref in zip(policy_stage_output, ref_stage_output)]
                else:
                    stage_output = None

        if stage_output is not None:
            outputs = {k: [output[k] for output in stage_output] for k in stage_output[0].keys()}
            return {k: torch.cat(v, dim=0) for k, v in outputs.items()}
        else:
            return None

    def generate(
        self,
        inputs: Union[Sequence[str], Tuple[torch.Tensor, torch.Tensor]],
        length_params: LengthParam,
        sampling_params: SamplingParam = None,
    ) -> OutputType:
        default_sampling_params = {
            "use_greedy": False,
            "temperature": self.ppo_config.gen_kwargs.get("temperature", 1.0),
            "top_k": self.ppo_config.gen_kwargs.get("top_k", 0),
            "top_p": 1.0,
            "repetition_penalty": 1.2,
            "add_BOS": False,
            "all_probs": False,
            "compute_logprob": False,
        }

        if sampling_params is None:
            sampling_params = default_sampling_params
        else:
            sampling_params = {**default_sampling_params, **sampling_params}

        self.maybe_initalize_per_dp_rng()

        if isinstance(inputs[0], str):
            inputs = self.tokenizer.tokenizer(inputs)
            context_tokens = inputs["input_ids"]
            max_new_tokens = length_params["max_length"]
            context_lengths = [len(x) for x in context_tokens]
            max_context_length = max(context_lengths)

            pad_id = self.tokenizer.tokenizer.eos_token_id
            padded = [x + [pad_id] * (max_context_length + max_new_tokens - len(x)) for x in context_tokens]

            inputs = (torch.as_tensor(padded).cuda(), torch.as_tensor(context_lengths).cuda())

        with self.inference_mode():
            # Fork the RNG per dp rank to ensure that each dp rank generates different samples
            with tensor_parallel.random.get_cuda_rng_tracker().fork(_PER_DP_RANK_RNG):
                output = super().generate(inputs, length_params, sampling_params)

            if output is None:
                return None

            _, lengths = inputs
            prompts = [
                sentence[: offset[l]] for sentence, offset, l in zip(output["sentences"], output["offsets"], lengths)
            ]
            responses = [
                sentence[offset[l] :] for sentence, offset, l in zip(output["sentences"], output["offsets"], lengths)
            ]

            if self.stop_sequences:
                for stop in self.stop_sequences:
                    responses = [response.split(stop)[0] for response in responses]
                output["sentences"] = [prompt + response for prompt, response in zip(prompts, responses)]
                output["token_ids"] = self.tokenizer.tokenizer(output["sentences"])["input_ids"]
                # Offsets and str tokens are no longer correct after slicing at stop sequences
                del output["offsets"]
                del output["tokens"]

            output["prompts"] = prompts
            output["responses"] = responses

            return output
