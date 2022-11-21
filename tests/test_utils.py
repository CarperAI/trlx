import accelerate
import pytest
import transformers

import trlx.utils.modeling as modeling_utils


# Test modeling utils


@pytest.mark.parametrize(
    "model_name",
    [
        "bigscience/bloom-3b",
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
        modeling_utils.get_causal_base_model,
        modeling_utils.get_causal_lm_head,
        modeling_utils.get_final_norm,
        modeling_utils.get_hidden_layers,
    ]
    for get in arch_getters:
        try:
            get(arch)
        except Exception as e:
            assert False, "Failed to get model attribute with error: " + str(e)

    config_getters = [modeling_utils.get_hidden_size]
    for get in config_getters:
        try:
            get(config)
        except Exception as e:
            assert False, "Failed to get config attribute with error: " + str(e)
