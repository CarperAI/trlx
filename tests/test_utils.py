import accelerate
import pytest
import torch
import transformers

import trlx.utils as utils
import trlx.utils.modeling as modeling_utils

# Test general utils


@pytest.mark.parametrize(
    "optimizer_name",
    [o.value for o in utils.OptimizerName],
)
def test_optimizer_class_getters(optimizer_name: str):
    try:
        _class = utils.get_optimizer_class(optimizer_name)
    except Exception as e:
        assert False, "Failed to get optimizer class with error: " + str(e)

    # Hard-check for one of the optimizers
    _class = utils.get_optimizer_class("adamw")
    assert _class == torch.optim.AdamW


@pytest.mark.parametrize(
    "scheduler_name",
    [o.value for o in utils.SchedulerName],
)
def test_scheduler_class_getters(scheduler_name: str):
    try:
        _class = utils.get_scheduler_class(scheduler_name)
    except Exception as e:
        assert False, "Failed to get scheduler class with error: " + str(e)

    # Hard-check for one of the schedulers
    _class = utils.get_scheduler_class("cosine_annealing")
    assert _class == torch.optim.lr_scheduler.CosineAnnealingLR


# Test modeling utils


@pytest.mark.parametrize(
    "model_name",
    [
        "EleutherAI/gpt-j-6B",
        "EleutherAI/gpt-neox-20b",
        "gpt2",
        "facebook/opt-1.3b",
    ],
)
def test_hf_attr_getters(model_name: str):
    with accelerate.init_empty_weights():
        config = transformers.AutoConfig.from_pretrained(model_name)
        arch = transformers.AutoModelForCausalLM.from_config(config)

    arch_getters = [
        modeling_utils.hf_get_causal_base_model,
        modeling_utils.hf_get_causal_final_norm,
        modeling_utils.hf_get_causal_hidden_layers,
        modeling_utils.hf_get_lm_head,
    ]
    for get in arch_getters:
        try:
            get(arch)
        except Exception as e:
            assert False, "Failed to get model attribute with error: " + str(e)

    config_getters = [
        modeling_utils.hf_get_hidden_size,
        modeling_utils.hf_get_num_hidden_layers,
    ]
    for get in config_getters:
        try:
            get(config)
        except Exception as e:
            assert False, "Failed to get config attribute with error: " + str(e)
