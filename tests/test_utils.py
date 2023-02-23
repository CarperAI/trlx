import unittest

import accelerate
import pytest
import torch
import transformers

import trlx.utils as utils
import trlx.utils.modeling as modeling_utils

try:
    import bitsandbytes

    HAS_BNB = True
except ImportError:
    HAS_BNB = False


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
    if HAS_BNB:
        _bnb_class = utils.get_optimizer_class("adamw_8bit_bnb")
        assert _bnb_class == bitsandbytes.optim.AdamW8bit


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
        modeling_utils.hf_get_decoder,
        modeling_utils.hf_get_decoder_final_norm,
        modeling_utils.hf_get_decoder_blocks,
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


@pytest.mark.parametrize(
    "model_name",
    [
        "EleutherAI/gpt-j-6B",
        "EleutherAI/gpt-neox-20b",
        "facebook/opt-1.3b",
        "bigscience/bloom-560m",
        "google/flan-t5-large",
    ],
)
def test_parse_delta_kwargs(model_name):
    config = transformers.AutoConfig.from_pretrained(model_name)

    modified_modules_dict = modeling_utils.MODIFIED_MODULES_DICT[config.model_type]
    for default_modifier, default_modified_modules in modified_modules_dict.items():
        delta_type, delta_kwargs = modeling_utils.parse_delta_kwargs(
            delta_kwargs={"delta_type": "lora", "modified_modules": default_modifier},
            config=config,
            num_layers_unfrozen=4,
        )
        # Ensure the parsed module regex patterns capture the default module names
        for kwarg_mod, default_mod in zip(delta_kwargs["modified_modules"], default_modified_modules):
            assert kwarg_mod.endswith(
                default_mod
            ), f"Parsed modified module `{kwarg_mod}` should contain the trlx default `{default_mod}`"
        assert delta_type == "lora", "Delta type should be lora"

    # Ensure the defaults don't get used if the user specifies a list of `modified_modules`
    delta_type, delta_kwargs = modeling_utils.parse_delta_kwargs(
        delta_kwargs={"delta_type": "lora", "modified_modules": ["a", "b"]},
        config=config,
        num_layers_unfrozen=2,
    )
    for kwarg_mod in delta_kwargs["modified_modules"]:
        assert kwarg_mod.endswith("a") or kwarg_mod.endswith("b"), "Parsed modified module should contain ['a', 'b']"


class TestStatistics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = modeling_utils.RunningMoments()
        cls.a1 = torch.arange(100, dtype=float)
        cls.a2 = torch.ones(100, dtype=float)
        cls.a3 = torch.exp(torch.arange(10, dtype=float))
        cls.a4 = torch.tensor([-10, -1, 0, 1, 10], dtype=float)

    def test_running_moments(self):
        assert torch.isclose(self.m.update(self.a1)[1], self.a1.std(unbiased=True), atol=1e-6)
        assert torch.isclose(self.m.update(self.a2)[1], self.a2.std(unbiased=True), atol=1e-6)
        assert torch.isclose(self.m.update(self.a3)[1], self.a3.std(unbiased=True), atol=1e-6)
        assert torch.isclose(self.m.update(self.a4)[1], self.a4.std(unbiased=True), atol=1e-6)

        a = torch.hstack((self.a1, self.a2, self.a3, self.a4))
        assert torch.isclose(self.m.mean, a.mean(), atol=1e-6)
        assert torch.isclose(self.m.std, a.std(unbiased=True), atol=1e-6)
