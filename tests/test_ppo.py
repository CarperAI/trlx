import unittest

import torch
from transformers import AutoTokenizer

from trlx.data.configs import TRLConfig
from trlx.trainer.nn.ppo_models import CausalLMHydraWithValueHead
from trlx.utils.modeling import RunningMoments


# Note tests must start with "test_"
class TestHydraHead(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Testing Hydra model...")
        config = TRLConfig.load_yaml("configs/test_config.yml")
        cls.hydra_model = CausalLMHydraWithValueHead(
            config.model.model_path, config.model.num_layers_unfrozen
        )

        tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        cls.dummy_inputs = tokenizer(
            "Once upon a time there was a happy goose named Louis. He liked to eat bananas.",
            truncation=True,
            padding="max_length",
            max_length=4,
            return_tensors="pt",
        )

    def test_lm_heads(self):
        with torch.no_grad():
            unfrozen_outputs = TestHydraHead.hydra_model(
                **TestHydraHead.dummy_inputs,
                return_dict=True,
                output_hidden_states=True
            )
            unfrozen_logits = unfrozen_outputs.logits
            last_hidden_states = unfrozen_outputs.hidden_states[-1].to(torch.float32)
            frozen_logits = TestHydraHead.hydra_model.frozen_head.lm_head(
                last_hidden_states
            )
            diff = torch.sum(unfrozen_logits - frozen_logits).item()
            self.assertEqual(diff, 0)

    def test_frozen_head(self):
        # Ensure that all parameters of the `hydra_model.frozen_head` are actually frozen
        for parameter in TestHydraHead.hydra_model.frozen_head.parameters():
            self.assertTrue(parameter.requires_grad is False)

    def test_forward(self):
        with torch.no_grad():
            unfrozen_outputs = TestHydraHead.hydra_model(
                **TestHydraHead.dummy_inputs,
                return_dict=True,
                output_hidden_states=True
            )
            unfrozen_last_hidden_states = unfrozen_outputs.hidden_states[-1]
            unfrozen_logits = unfrozen_outputs.logits

            frozen_outputs = TestHydraHead.hydra_model.forward_hydra(
                **TestHydraHead.dummy_inputs,
                return_dict=True,
                output_hidden_states=True
            )
            frozen_last_hidden_states = frozen_outputs.hidden_states[-1]
            frozen_logits = frozen_outputs.logits

            hs_diff = torch.sum(
                unfrozen_last_hidden_states - frozen_last_hidden_states
            ).item()
            logits_diff = torch.sum(unfrozen_logits - frozen_logits).item()
            self.assertEqual(hs_diff, 0)
            self.assertEqual(logits_diff, 0)


class TestStatistics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = RunningMoments()
        cls.a1 = torch.arange(100, dtype=float)
        cls.a2 = torch.ones(100, dtype=float)
        cls.a3 = torch.exp(torch.arange(10, dtype=float))
        cls.a4 = torch.tensor([-10, -1, 0, 1, 10], dtype=float)

    def test_running_moments(self):
        assert torch.isclose(
            self.m.update(self.a1)[1], self.a1.std(unbiased=True), atol=1e-6
        )
        assert torch.isclose(
            self.m.update(self.a2)[1], self.a2.std(unbiased=True), atol=1e-6
        )
        assert torch.isclose(
            self.m.update(self.a3)[1], self.a3.std(unbiased=True), atol=1e-6
        )
        assert torch.isclose(
            self.m.update(self.a4)[1], self.a4.std(unbiased=True), atol=1e-6
        )

        a = torch.hstack((self.a1, self.a2, self.a3, self.a4))
        assert torch.isclose(self.m.mean, a.mean(), atol=1e-6)
        assert torch.isclose(self.m.std, a.std(unbiased=True), atol=1e-6)
