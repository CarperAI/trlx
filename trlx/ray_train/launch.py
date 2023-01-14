#!/usr/bin/env python

# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import warnings
from unittest.mock import patch

from accelerate.commands.config.config_utils import DYNAMO_BACKENDS
from accelerate.utils import (
    DynamoBackend,
    PrecisionType,
    is_deepspeed_available,
    is_torch_version,
)
from accelerate.utils.launch import env_var_path_add
from accelerate.commands.launch import (
    launch_command_parser,
    launch_command as original_launch_command,
)

logger = logging.getLogger(__name__)


def simple_launcher(args):
    current_env = {}
    current_env["ACCELERATE_USE_CPU"] = str(args.cpu or args.use_cpu)
    if args.use_mps_device:
        warnings.warn(
            '`use_mps_device` flag is deprecated and will be removed in version 0.15.0 of 🤗 Accelerate. Use "--mps" instead.',
            FutureWarning,
        )
        args.mps = True
    current_env["ACCELERATE_USE_MPS_DEVICE"] = str(args.mps)
    if args.mps:
        current_env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    try:
        mixed_precision = PrecisionType(args.mixed_precision.lower())
    except ValueError:
        raise ValueError(
            f"Unknown mixed_precision mode: {args.mixed_precision.lower()}. Choose between {PrecisionType.list()}."
        )

    if args.fp16:
        warnings.warn(
            "`fp16` is deprecated and will be removed in version 0.15.0 of 🤗 Accelerate. Use `mixed_precision fp16` instead.",
            FutureWarning,
        )
        mixed_precision = "fp16"

    current_env["ACCELERATE_MIXED_PRECISION"] = str(mixed_precision)

    try:
        dynamo_backend = DynamoBackend(args.dynamo_backend.upper())
    except ValueError:
        raise ValueError(
            f"Unknown dynamo backend: {args.dynamo_backend.upper()}. Choose between {DYNAMO_BACKENDS}."
        )
    current_env["ACCELERATE_DYNAMO_BACKEND"] = dynamo_backend.value

    current_env["OMP_NUM_THREADS"] = str(args.num_cpu_threads_per_process)

    os.environ.update(current_env)


def multi_gpu_launcher(args):
    current_env = {}
    mixed_precision = args.mixed_precision.lower()
    try:
        mixed_precision = PrecisionType(mixed_precision)
    except ValueError:
        raise ValueError(
            f"Unknown mixed_precision mode: {mixed_precision}. Choose between {PrecisionType.list()}."
        )

    if args.fp16:
        warnings.warn(
            "`fp16` is deprecated and will be removed in version 0.15.0 of 🤗 Accelerate. Use `mixed_precision fp16` instead.",
            FutureWarning,
        )
        mixed_precision = "fp16"

    current_env["ACCELERATE_MIXED_PRECISION"] = str(mixed_precision)

    try:
        dynamo_backend = DynamoBackend(args.dynamo_backend.upper())
    except ValueError:
        raise ValueError(
            f"Unknown dynamo backend: {args.dynamo_backend.upper()}. Choose between {DYNAMO_BACKENDS}."
        )
    current_env["ACCELERATE_DYNAMO_BACKEND"] = dynamo_backend.value

    if args.use_fsdp:
        current_env["ACCELERATE_USE_FSDP"] = "true"
        current_env["FSDP_SHARDING_STRATEGY"] = str(args.fsdp_sharding_strategy)
        current_env["FSDP_OFFLOAD_PARAMS"] = str(args.fsdp_offload_params).lower()
        current_env["FSDP_MIN_NUM_PARAMS"] = str(args.fsdp_min_num_params)
        if args.fsdp_auto_wrap_policy is not None:
            current_env["FSDP_AUTO_WRAP_POLICY"] = str(args.fsdp_auto_wrap_policy)
        if args.fsdp_transformer_layer_cls_to_wrap is not None:
            current_env["FSDP_TRANSFORMER_CLS_TO_WRAP"] = str(
                args.fsdp_transformer_layer_cls_to_wrap
            )
        if args.fsdp_backward_prefetch_policy is not None:
            current_env["FSDP_BACKWARD_PREFETCH"] = str(
                args.fsdp_backward_prefetch_policy
            )
        if args.fsdp_state_dict_type is not None:
            current_env["FSDP_STATE_DICT_TYPE"] = str(args.fsdp_state_dict_type)

    if args.use_megatron_lm:
        prefix = "MEGATRON_LM_"
        current_env["ACCELERATE_USE_MEGATRON_LM"] = "true"
        current_env[prefix + "TP_DEGREE"] = str(args.megatron_lm_tp_degree)
        current_env[prefix + "PP_DEGREE"] = str(args.megatron_lm_pp_degree)
        current_env[prefix + "GRADIENT_CLIPPING"] = str(
            args.megatron_lm_gradient_clipping
        )
        if args.megatron_lm_num_micro_batches is not None:
            current_env[prefix + "NUM_MICRO_BATCHES"] = str(
                args.megatron_lm_num_micro_batches
            )
        if args.megatron_lm_sequence_parallelism is not None:
            current_env[prefix + "SEQUENCE_PARALLELISM"] = str(
                args.megatron_lm_sequence_parallelism
            )
        if args.megatron_lm_recompute_activations is not None:
            current_env[prefix + "RECOMPUTE_ACTIVATIONS"] = str(
                args.megatron_lm_recompute_activations
            )
        if args.megatron_lm_use_distributed_optimizer is not None:
            current_env[prefix + "USE_DISTRIBUTED_OPTIMIZER"] = str(
                args.megatron_lm_use_distributed_optimizer
            )

    current_env["OMP_NUM_THREADS"] = str(args.num_cpu_threads_per_process)
    if is_torch_version("<", "1.9.0"):
        raise NotImplementedError("Multi-node training requires pytorch>=1.9.0")

    os.environ.update(current_env)


def deepspeed_launcher(args):
    if not is_deepspeed_available():
        raise ImportError(
            "DeepSpeed is not installed => run `pip3 install deepspeed` or build it from source."
        )

    current_env = {}
    try:
        mixed_precision = PrecisionType(args.mixed_precision.lower())
    except ValueError:
        raise ValueError(
            f"Unknown mixed_precision mode: {args.mixed_precision.lower()}. Choose between {PrecisionType.list()}."
        )

    if args.fp16:
        warnings.warn(
            '--fp16 flag is deprecated and will be removed in version 0.15.0 of 🤗 Accelerate. Use "--mixed_precision fp16" instead.',
            FutureWarning,
        )
        mixed_precision = "fp16"

    current_env["PYTHONPATH"] = env_var_path_add("PYTHONPATH", os.path.abspath("."))
    current_env["ACCELERATE_MIXED_PRECISION"] = str(mixed_precision)
    current_env["ACCELERATE_USE_DEEPSPEED"] = "true"
    current_env["DEEPSPEED_ZERO_STAGE"] = str(args.zero_stage)
    current_env["GRADIENT_ACCUMULATION_STEPS"] = str(args.gradient_accumulation_steps)
    current_env["GRADIENT_CLIPPING"] = str(args.gradient_clipping).lower()
    current_env["DEEPSPEED_OFFLOAD_OPTIMIZER_DEVICE"] = str(
        args.offload_optimizer_device
    ).lower()
    current_env["DEEPSPEED_OFFLOAD_PARAM_DEVICE"] = str(
        args.offload_param_device
    ).lower()
    current_env["DEEPSPEED_ZERO3_INIT"] = str(args.zero3_init_flag).lower()
    current_env["DEEPSPEED_ZERO3_SAVE_16BIT_MODEL"] = str(
        args.zero3_save_16bit_model
    ).lower()
    if args.deepspeed_config_file is not None:
        current_env["DEEPSPEED_CONFIG_FILE"] = str(args.deepspeed_config_file)

    with open(".deepspeed_env", "a") as f:
        for key, value in current_env.items():
            if ";" in value or " " in value:
                continue
            f.write(f"{key}={value}\n")

    os.environ.update(current_env)


def _raise_notimplementederror(*args, **kwargs):
    raise NotImplementedError()


def launch_command(args):
    with patch(
        "accelerate.commands.launch.deepspeed_launcher", deepspeed_launcher
    ), patch(
        "accelerate.commands.launch.multi_gpu_launcher", multi_gpu_launcher
    ), patch(
        "accelerate.commands.launch.simple_launcher", simple_launcher
    ), patch(
        "accelerate.commands.launch.tpu_launcher", _raise_notimplementederror
    ), patch(
        "accelerate.commands.launch.sagemaker_launcher", _raise_notimplementederror
    ):
        return original_launch_command(args)


def main():
    parser = launch_command_parser()
    args = parser.parse_args()
    launch_command(args)


if __name__ == "__main__":
    main()
