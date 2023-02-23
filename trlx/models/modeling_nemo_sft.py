# Extensible version of the GPT model
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed
from apex.transformer import parallel_state
from apex.transformer.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    gather_from_tensor_model_parallel_region,
)
from einops import rearrange
from nemo.collections.common.losses import CrossEntropyLoss
from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_samplers import (
    MegatronPretrainingBatchSampler,
)
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import (
    MegatronGPTModel,
)
from nemo.collections.nlp.modules.common.megatron.module import (
    Float16Module,
    # MegatronModule,
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

import trlx.utils as utils
from trlx.trainer.accelerate_sft_trainer import SFTConfig
from trlx.utils import to_device

# from nemo.utils import logging



try:
    from apex.transformer import parallel_state
    from apex.transformer.pipeline_parallel.utils import (
        _reconfigure_microbatch_calculator,
        get_num_microbatches,
    )

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False


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


class SFTGPT(MegatronGPTModel):
    sft_config: SFTConfig

    def __init__(self, sft_config, metric_fn=None, **kwargs):
        self.sft_config = sft_config
        self.metric_fn = metric_fn
        super().__init__(**kwargs)
        if len(list(self.parameters())) == 0:
            raise ValueError("No parameters in model")

        self.loss = self.setup_loss()

        self._ori_activations_checkpoint_granularity = self.cfg.get("activations_checkpoint_granularity", None)
        self._ori_activations_checkpoint_method = self.cfg.get("activations_checkpoint_method", None)
        self._ori_activations_checkpoint_num_layers = self.cfg.get("activations_checkpoint_num_layers", None)
        utils.print_rank_0("vocab size", self.tokenizer.tokenizer.vocab_size)

    def setup_loss(self):
        return CrossEntropyLoss(logits_ndim=3, ignore_index=-100)  # self.tokenizer.tokenizer.eos_token_id)

    def build_train_valid_test_datasets(self):
        pass

    def build_data_loader(self, dataset, collate_fn, consumed_samples=0):
        dp_rank = parallel_state.get_data_parallel_rank()
        dp_size = parallel_state.get_data_parallel_world_size()
        logging.info(
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

    def set_train_dataset(self, train_dataset, collate_fn: Optional[callable] = None):
        self._train_dataset = train_dataset
        self._train_collate_fn = collate_fn

    def set_valid_dataset(self, valid_dataset, collate_fn: Optional[callable] = None):
        self._valid_dataset = valid_dataset
        self._valid_collate_fn = collate_fn

    # Called by superclass to build data loaders
    def setup_training_data(self, _):
        if hasattr(self, "_train_dataset"):
            self._train_dl = self.build_data_loader(self._train_dataset, self._train_collate_fn)

    def setup_validation_data(self, _):
        if hasattr(self, "_valid_dataset"):
            self._validation_dl = self.build_data_loader(self._valid_dataset, self._valid_collate_fn)

    def load_from_pretrained(self, checkpoint_dir):
        mp_rank = parallel_state.get_tensor_model_parallel_rank()
        checkpoint_path = Path(checkpoint_dir)

        # Check if there are rank subfolders
        rank_subfolder = f"mp_rank_{mp_rank:02d}"
        if (checkpoint_path / rank_subfolder).exists():
            rank_params = checkpoint_path / rank_subfolder / "model_weights.ckpt"
        else:
            rank_params = checkpoint_path / "model_weights.ckpt"
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
        # # This enables the final layernorm in the GPT model if there is one
        gpt.language_model.post_process = post_process
        return gpt

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

    # Adapted from NeMo
    # https://github.com/NVIDIA/NeMo/blob/r1.13.0/nemo/collections/nlp/models/language_modeling/megatron_gpt_model.py#L259
    def training_step(self, batch: List[torch.Tensor], batch_idx: int):  # noqa: C901
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
            self.cfg.encoder_seq_length,
            self.cfg.micro_batch_size,
            self.cfg.hidden_size,
        ]

        # handle asynchronous grad reduction
        if self.with_distributed_adam:
            if self.megatron_amp_o2:
                # copy grads to main grad
                def custom_sync_context_handler():
                    return self._optimizer.no_sync(greedy_grad_copy=True)

            else:
                # keep grad tensors around
                def custom_sync_context_handler():
                    return self._optimizer.no_sync(greedy_grad_copy=False)

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
        elif self.megatron_amp_o2:
            # when using pipeline parallelism grads must be all-reduced after the pipeline (not asynchronously)
            if self.cfg.get("pipeline_model_parallel_size", 1) > 1 or self.cfg.get("sequence_parallel", False):
                # main grads are stored in the MainParamsOptimizer wrapper
                self._optimizer.allreduce_main_grads()
        else:
            # async grad allreduce is not currently implemented for O1/autocasting mixed precision training
            # so we all-reduce gradients after the pipeline
            self.allreduce_gradients()  # @sangkug we think this is causing memory to blow up (hurts perf)

        if self.cfg.get("pipeline_model_parallel_size", 1) > 1:
            # when using pipeline parallelism the first and last stage must keep embeddings in sync
            self.allreduce_first_last_embeddings()

        # we can only log on one rank if it is rank zero so we broadcast from last rank
        # we can avoid this broadcast by updating the PTL log function to accept specific ranks
        torch.distributed.broadcast(loss_mean, get_last_rank())

        if self.cfg.precision == 16:
            loss_scale = self.trainer.precision_plugin.scaler._scale
            if loss_scale is not None:
                self.log("loss_scale", loss_scale)

        self.log(
            "reduced_train_loss",
            loss_mean,
            prog_bar=True,
            rank_zero_only=True,
        )

        for k, v in mean_outputs.items():
            if k != "avg_loss":
                self.log(k, v)

        self.log(
            "global_step",
            float(self.trainer.global_step),
            prog_bar=True,
            rank_zero_only=True,
        )

        if self.trainer.global_step % self.ilql_config.steps_for_target_q_sync == 0 and self.trainer.global_step > 0:
            if parallel_state.is_pipeline_last_stage():
                unwrap_float16_module(self.model).other_heads.sync_target_q_heads()

        return loss_mean

    def validation_step(self, batch: Tuple[List[int], List[int]], batch_idx: int):
        if self.metric_fn is None:
            raise ValueError("Must set metric_fn to use validation")

        sp_was_enabled = self.cfg.get("sequence_parallel", False)
        if sp_was_enabled:
            self.sequence_parallel_(False)

        activations_checkpointing_was_enabled = self.cfg.get("activations_checkpoint_granularity", None) is not None

        if activations_checkpointing_was_enabled:
            self.activation_checkpointing_(False)

        input_ids, lengths = batch
        input_ids, lengths = torch.as_tensor(input_ids), torch.as_tensor(lengths)

        input_ids, lengths = to_device((input_ids, lengths), torch.cuda.current_device(), non_blocking=True)

        max_new_tokens = self.sft_config.gen_kwargs.get("max_new_tokens", 64)

        gen = self.generate(
            (input_ids, lengths),
            dict(max_length=max_new_tokens, min_length=0)
        )

        metrics = self.metric_fn(gen["sentences"])

        metric_keys, metric_values = zip(*metrics.items())

        columns = ["sentences", *metric_keys]
        rows = list(zip(gen["sentences"], *metric_values))

        avg_metrics = {f"avg_{k}": torch.as_tensor(v).mean() for k, v in metrics.items()}

        if activations_checkpointing_was_enabled:
            self.activation_checkpointing_(True)

        if sp_was_enabled:
            self.sequence_parallel_(True)

        # NeMo generate resets the microbatch calculator
        from apex.transformer.pipeline_parallel.utils import (
            _reconfigure_microbatch_calculator,
        )
        from nemo.utils import AppState

        _reconfigure_microbatch_calculator(
            rank=AppState().global_rank,
            rampup_batch_size=None,
            global_batch_size=self.cfg.global_batch_size,
            micro_batch_size=self.cfg.micro_batch_size,
            data_parallel_size=AppState().data_parallel_size,
        )

        return avg_metrics, (rows, columns)

    def validation_epoch_end(self, outputs: List[Tuple[dict, Tuple[List[str], List[str]]]]):
        metrics, tables = zip(*outputs)
        _, columns = tables[0]
        rows = [r for trows, _ in tables for r in trows]

        self.logger.log_text(key="samples", columns=columns, data=rows)

        outputs_soa = {k: torch.as_tensor([d[k] for d in metrics]) for k in metrics[0].keys()}
        # this assumes all validation microbatches are the same size
        avg_outputs = {k: v.mean() for k, v in outputs_soa.items()}
        for k, v in avg_outputs.items():
            self.log(
                f"val_metrics/{k}",
                v,
                prog_bar=True,
                rank_zero_only=True,
                sync_dist=True,
            )

    def get_forward_output_and_loss_func(self, validation_step=False):
        def fwd_output_and_loss_func(batch: List[torch.Tensor], model, checkpoint_activations_all_layers=None):
            # On first and last pipeline stages, the input data is passed in
            if batch is not None:
                batch = to_device(batch, torch.cuda.current_device(), non_blocking=True)
                utils.print_rank_0(f"batch: {batch}")
                inputs = torch.vstack(batch) if isinstance(batch, list) else batch[0]
                utils.print_rank_0("inputs: ", inputs.shape, inputs.dtype, inputs.device)
                pad_by = self.cfg.encoder_seq_length - inputs.shape[1]
                inputs = torch.nn.functional.pad(inputs, (0, pad_by), value=self.tokenizer.eos_id)
                utils.print_rank_0("inputs: ", inputs.shape, inputs.dtype, inputs.device)
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
                utils.print_rank_0("attention_mask: ", attention_mask.shape, attention_mask.dtype, attention_mask.device)
                utils.print_rank_0("position_ids: ", position_ids.shape, position_ids.dtype, position_ids.device)

                model_output = model(
                    input_ids=inputs,
                    position_ids=position_ids.long(),
                    attention_mask=attention_mask,
                )
                utils.print_rank_0("model_output: ", model_output.shape, model_output.dtype, model_output.device)
            else:
                # In-between stages are given data via the pipeline engine
                # Still need to specify these arguments to avoid errors
                model_output = model(input_ids=None, position_ids=None, attention_mask=None)

            def gather_ntc(t: torch.Tensor):
                """Gather sequence parallel tensor [batch, seq, hidden]"""
                t = rearrange(t, "N T ... -> T N ...")
                t = gather_from_sequence_parallel_region(t, to_model_parallel=False)
                t = rearrange(t, "T N ... -> N T ...")
                return t

            def loss_func(model_output):
                # TODO: implement this in a sequence parallel way
                logits = model_output
                utils.print_rank_0(f"loss logits: {logits}\n logits shape: {logits.shape}")
                logits = gather_from_tensor_model_parallel_region(logits)
                utils.print_rank_0(f"gathered loss logits: {logits}\n logits shape: {logits.shape}")
                utils.print_rank_0(f"loss labels: {inputs}\n labels shape: {inputs.shape}")
                loss_func = CrossEntropyLoss(logits_ndim=3, ignore_index=self.tokenizer.tokenizer.eos_token_id)
                loss_for_mb = loss_func(logits=logits, labels=inputs, loss_mask=loss_mask)
                reduced_loss = average_losses_across_data_parallel_group([loss_for_mb])

                # TODO: figure out why this sync is needed (crashes otherwise)
                torch.cuda.synchronize()

                return loss_for_mb, {"avg_loss": reduced_loss}

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

            def sft_postprocess(model_output):
                model_output = utils.tree_map(lambda t: t.float(), model_output)
                logits = model_output
                return logits, {"logits": logits}

            return model_output, sft_postprocess

        return fwd_output_only_func

    # Need to override this otherwise distributed fused adam won't work
    # with frozen layers
    def parameters(self):
        return (p for p in self.model.parameters() if p.requires_grad)

    def generate(
        self,
        inputs: Union[List[str], torch.Tensor, List[dict]],
        length_params: LengthParam,
        sampling_params: SamplingParam = None,
    ) -> OutputType:
        if sampling_params is None:
            sampling_params = {
                "use_greedy": self.sft_config.gen_kwargs.get("use_greedy", False),
                "temperature": self.sft_config.gen_kwargs.get("temperature", 1.0),
                "top_k": self.sft_config.gen_kwargs.get("top_k", 0),
                "top_p": self.sft_config.gen_kwargs.get("top_p", 0.9),
                "repetition_penalty": self.sft_config.gen_kwargs.get("repetition_penalty", 1.0),
                "add_BOS": False,
                "all_probs": False,
                "compute_logprob": False,
            }

        return super().generate(inputs, length_params, sampling_params)
