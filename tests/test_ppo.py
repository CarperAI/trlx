import unittest
from trlx.data.configs import TRLConfig
from trlx.model.accelerate_ppo_model import AcceleratePPOModel
from trlx.orchestrator.ppo_orchestrator import PPOOrchestrator
from trlx.pipeline.ppo_pipeline import PPOPipeline
from trlx.utils.loading import get_model, get_orchestrator, get_pipeline
from trlx.model.nn.ppo_models import GPTHeadWithValueModel, GPTHydraHeadWithValueModel
from trlx.data.accelerate_base_datatypes import PromptBatch
from transformers import AutoConfig, AutoTokenizer
import torch


# Note tests must start with "test_"
class TestHydraHead(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Testing Hydra model...")
        config = TRLConfig.load_yaml("configs/test_config.yml")
        cls.hydra_model = GPTHydraHeadWithValueModel(
            config.model.model_path, config.model.num_layers_unfrozen
        )

        tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        text = "Once upon a time there was a happy goose named Louis. He liked to eat bananas."
        cls.dummy_inputs = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=config.train.input_size,
            return_tensors="pt",
        )

        cls.dummy_batch = PromptBatch([text], cls.dummy_inputs["input_ids"])

    def test_lm_heads(self):
        with torch.no_grad():
            unfrozen_outputs = TestHydraHead.hydra_model(
                **TestHydraHead.dummy_inputs,
                return_dict=True,
                output_hidden_states=True
            )
            unfrozen_logits = unfrozen_outputs.logits
            last_hidden_states = unfrozen_outputs.hidden_states[-1].to(torch.float32)
            frozen_logits = TestHydraHead.hydra_model.ref_model.lm_head(
                last_hidden_states
            )
            diff = torch.sum(unfrozen_logits - frozen_logits).item()
            self.assertEqual(diff, 0)

    def test_forward(self):
        with torch.no_grad():
            unfrozen_outputs = TestHydraHead.hydra_model(
                **TestHydraHead.dummy_inputs,
                return_dict=True,
                output_hidden_states=True
            )
            unfrozen_last_hidden_states = unfrozen_outputs.hidden_states[-1]
            unfrozen_logits = unfrozen_outputs.logits

            frozen_outputs = TestHydraHead.hydra_model.ref_model(
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
