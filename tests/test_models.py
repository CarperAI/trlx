import copy
import gc
import tempfile
import unittest

import torch
import transformers

from trlx.models.modeling_ilql import (
    AutoModelForCausalLMWithILQLHeads,
    AutoModelForSeq2SeqLMWithILQLHeads,
)
from trlx.models.modeling_ppo import (
    AutoModelForCausalLMWithHydraValueHead,
    AutoModelForCausalLMWithValueHead,
    AutoModelForSeq2SeqLMWithHydraValueHead,
    AutoModelForSeq2SeqLMWithValueHead,
)

AUTO_CAUSAL_LM_PATHS = ["gpt2", "EleutherAI/pythia-160m", "facebook/opt-125m"]
AUTO_SEQ2SEQ_LM_PATHS = ["t5-small", "google/flan-t5-small"]


# Value Head Modeling Tests


class TestAutoModelForCausalLMWithValueHead(unittest.TestCase):
    _auto_model_class = AutoModelForCausalLMWithValueHead
    _supported_args = {}

    def setUp(self):
        self.text = "Once upon a time there was a happy goose named Louis. He liked to eat bananas."

    def tearDown(self):
        gc.collect()  # Try to free up memory

    def _create_inputs(self, model_path):
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return tokenizer(self.text, truncation=True, padding="max_length", max_length=4, return_tensors="pt")

    def test_forward(self):
        for model_path in AUTO_CAUSAL_LM_PATHS:
            model = self._auto_model_class.from_pretrained(model_path, **self._supported_args)
            inputs = self._create_inputs(model_path)

            # Ensure that the `forward` method doesn't throw an error on generic inputs
            try:
                model(**inputs)
            except Exception as e:
                self.assertFalse(True, msg=e)

    def test_generate(self):
        for model_path in AUTO_CAUSAL_LM_PATHS:
            model = self._auto_model_class.from_pretrained(model_path, **self._supported_args)
            inputs = self._create_inputs(model_path)

            # Ensure that the `generate` method doesn't throw an error on generic inputs
            try:
                model.generate(**inputs, return_dict=True, output_hidden_states=True)
            except Exception as e:
                self.assertFalse(True, msg=e)

    def test_save_load(self):
        for model_path in AUTO_CAUSAL_LM_PATHS:
            model = self._auto_model_class.from_pretrained(model_path, **self._supported_args)
            modified_model = copy.deepcopy(model)

            # Manually modify value head parameters
            modified_model.v_head[-1].bias = torch.nn.Parameter(torch.tensor([6000053.33]))

            with tempfile.TemporaryDirectory() as tmpdirname:
                modified_model.save_pretrained(tmpdirname)
                loaded_model = self._auto_model_class.from_pretrained(tmpdirname)

            # Check that the loaded model state dict is the same as the saved model state dict
            loaded_state_dict = loaded_model.state_dict()
            self.assertEqual(modified_model.state_dict().keys(), loaded_state_dict.keys())
            for name, saved_state in modified_model.state_dict().items():
                self.assertTrue(torch.all(torch.isclose(saved_state, loaded_state_dict[name])))

            # Assert loaded states are not the same as the original unmodified pretrained model
            self.assertFalse(torch.all(torch.isclose(modified_model.v_head[-1].bias, model.v_head[-1].bias)))

    def test_from_config(self):
        for model_path in AUTO_CAUSAL_LM_PATHS:
            config = transformers.AutoConfig.from_pretrained(model_path)
            # Modify the config to ensure the model is initialized from the custom config
            config.vocab_size = 2
            model = self._auto_model_class.from_config(config, **self._supported_args)
            self.assertEqual(model.base_model.get_output_embeddings().out_features, config.vocab_size)


class TestAutoModelForCausalLMWithHydraValueHead(TestAutoModelForCausalLMWithValueHead):
    _auto_model_class = AutoModelForCausalLMWithHydraValueHead
    _supported_args = {"num_layers_unfrozen": 2}  # TODO: Test various values

    def test_forward(self):
        for model_path in AUTO_CAUSAL_LM_PATHS:
            model = self._auto_model_class.from_pretrained(model_path, **self._supported_args)
            inputs = self._create_inputs(model_path)

            with torch.no_grad():
                # Compare logits and hidden states from frozen and unfrozen heads
                unfrozen_outputs = model(**inputs, return_dict=True, output_hidden_states=True)
                unfrozen_last_hidden_state = unfrozen_outputs.hidden_states[-1]
                unfrozen_logits = unfrozen_outputs.logits

                frozen_outputs = model.forward_hydra(**inputs, return_dict=True, output_hidden_states=True)
                frozen_last_hidden_state = frozen_outputs.hidden_states[-1]
                frozen_logits = frozen_outputs.logits

                hs_diff = torch.sum(unfrozen_last_hidden_state - frozen_last_hidden_state).item()
                logits_diff = torch.sum(unfrozen_logits - frozen_logits).item()

                self.assertEqual(hs_diff, 0)
                self.assertEqual(logits_diff, 0)

    def test_lm_heads(self):
        for model_path in AUTO_CAUSAL_LM_PATHS:
            model = self._auto_model_class.from_pretrained(model_path, **self._supported_args)
            inputs = self._create_inputs(model_path)

            # Compare frozen and unfrozen logits
            with torch.no_grad():
                unfrozen_outputs = model(**inputs, return_dict=True, output_hidden_states=True)
                unfrozen_logits = unfrozen_outputs.logits
                frozen_logits = model.frozen_head.lm_head(unfrozen_outputs.hidden_states[-1].to(torch.float32))
                diff = torch.sum(unfrozen_logits - frozen_logits).item()
                self.assertEqual(diff, 0)

    def test_frozen_head(self):
        for model_path in AUTO_CAUSAL_LM_PATHS:
            model = self._auto_model_class.from_pretrained(model_path, **self._supported_args)
            # Ensure that all parameters of the hyrda `model.frozen_head` are actually frozen
            for parameter in model.frozen_head.parameters():
                self.assertTrue(parameter.requires_grad is False)


class TestAutoModelForSeq2SeqLMWithValueHead(unittest.TestCase):
    _auto_model_class = AutoModelForSeq2SeqLMWithValueHead
    _supported_args = {}

    def setUp(self):
        self.encoder_text = "Translate this text to French: Hello, my dog is cute"
        self.decoder_text = "Bonjour, mon chien est mignon"

    def tearDown(self):
        gc.collect()  # Try to free up memory

    def _create_inputs(self, model_path):
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        tokenizer.sep_token = "<sep>"
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        encoder_inputs = tokenizer(
            self.encoder_text, truncation=True, padding="max_length", max_length=10, return_tensors="pt"
        )
        decoder_inputs = tokenizer(self.decoder_text, return_tensors="pt")
        return {
            **encoder_inputs,
            "decoder_input_ids": decoder_inputs.input_ids,
            "decoder_attention_mask": decoder_inputs.attention_mask,
        }

    def test_forward(self):
        for model_path in AUTO_SEQ2SEQ_LM_PATHS:
            model = self._auto_model_class.from_pretrained(model_path, **self._supported_args)
            inputs = self._create_inputs(model_path)

            # Ensure that the `forward` method doesn't throw an error on generic inputs
            try:
                model(**inputs)
            except Exception as e:
                self.assertFalse(True, msg=e)

    def test_generate(self):
        for model_path in AUTO_SEQ2SEQ_LM_PATHS:
            model = self._auto_model_class.from_pretrained(model_path, **self._supported_args)
            inputs = self._create_inputs(model_path)

            # Ensure that the `generate` method doesn't throw an error on generic inputs
            try:
                model.generate(inputs["input_ids"])
            except Exception as e:
                self.assertFalse(True, msg=e)

    def test_save_load(self):
        for model_path in AUTO_SEQ2SEQ_LM_PATHS:
            model = self._auto_model_class.from_pretrained(model_path, **self._supported_args)
            modified_model = copy.deepcopy(model)

            # Manually modify value head parameters
            modified_model.v_head[-1].bias = torch.nn.Parameter(torch.tensor([6000053.33]))

            with tempfile.TemporaryDirectory() as tmpdirname:
                modified_model.save_pretrained(tmpdirname)
                loaded_model = self._auto_model_class.from_pretrained(tmpdirname)

            # Check that the loaded model state dict is the same as the saved model state dict
            loaded_state_dict = loaded_model.state_dict()
            self.assertEqual(modified_model.state_dict().keys(), loaded_state_dict.keys())
            for name, saved_state in modified_model.state_dict().items():
                self.assertTrue(torch.all(torch.isclose(saved_state, loaded_state_dict[name])))

            # Assert loaded states are not the same as the original unmodified pretrained model
            self.assertFalse(torch.all(torch.isclose(modified_model.v_head[-1].bias, model.v_head[-1].bias)))

    def test_from_config(self):
        for model_path in AUTO_SEQ2SEQ_LM_PATHS:
            config = transformers.AutoConfig.from_pretrained(model_path)
            # Modify the config to ensure the model is initialized from the custom config
            config.vocab_size = 2
            model = self._auto_model_class.from_config(config, **self._supported_args)
            self.assertEqual(model.base_model.get_output_embeddings().out_features, config.vocab_size)


class TestAutoModelForSeq2SeqLMWithHydraValueHead(TestAutoModelForSeq2SeqLMWithValueHead):
    _auto_model_class = AutoModelForSeq2SeqLMWithHydraValueHead
    _supported_args = {"num_layers_unfrozen": 2}  # TODO: Test various values

    @unittest.skip("TODO: Final hidden states are not the same for frozen and unfrozen T5 heads")
    def test_forward(self):
        for model_path in AUTO_SEQ2SEQ_LM_PATHS:
            model = self._auto_model_class.from_pretrained(model_path, **self._supported_args)
            inputs = self._create_inputs(model_path)

            with torch.no_grad():
                # Compare logits and hidden states from frozen and unfrozen heads
                unfrozen_outputs = model(**inputs, return_dict=True, output_hidden_states=True)
                unfrozen_last_hidden_state = unfrozen_outputs.decoder_hidden_states[-1]
                unfrozen_logits = unfrozen_outputs.logits

                frozen_outputs = model.forward_hydra(**inputs, return_dict=True, output_hidden_states=True)
                frozen_last_hidden_state = frozen_outputs.decoder_hidden_states[-1]
                frozen_logits = frozen_outputs.logits

                hs_diff = torch.sum(unfrozen_last_hidden_state - frozen_last_hidden_state).item()
                logits_diff = torch.sum(unfrozen_logits - frozen_logits).item()

                self.assertEqual(hs_diff, 0)
                self.assertEqual(logits_diff, 0)

    @unittest.skip("TODO: Final hidden states are not the same for frozen and unfrozen T5 heads")
    def test_lm_heads(self):
        for model_path in AUTO_SEQ2SEQ_LM_PATHS:
            model = self._auto_model_class.from_pretrained(model_path, **self._supported_args)
            inputs = self._create_inputs(model_path)

            # Compare frozen and unfrozen logits
            with torch.no_grad():
                unfrozen_outputs = model(**inputs, return_dict=True, output_hidden_states=True)
                unfrozen_logits = unfrozen_outputs.logits
                last_hidden_state = unfrozen_outputs.decoder_hidden_states[-1]
                frozen_logits = model.frozen_head.lm_head(last_hidden_state)
                diff = torch.sum(unfrozen_logits - frozen_logits).item()
                self.assertEqual(diff, 0)

    def test_frozen_head(self):
        for model_path in AUTO_SEQ2SEQ_LM_PATHS:
            model = self._auto_model_class.from_pretrained(model_path, **self._supported_args)
            # Ensure that all parameters of the hyrda `model.frozen_head` are actually frozen
            for parameter in model.frozen_head.parameters():
                self.assertTrue(parameter.requires_grad is False)


# ILQL Heads Modeling Tests


class TestAutoModelForCausalLMWithILQLHeads(unittest.TestCase):
    _auto_model_class = AutoModelForCausalLMWithILQLHeads
    _supported_args = {"two_qs": True, "alpha": 0.8}  # TODO: Test various values

    def setUp(self):
        self.text = "Once upon a time there was a happy goose named Louis. He liked to eat bananas."

    def tearDown(self):
        gc.collect()  # Try to free up memory

    def _create_inputs(self, model_path):
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return tokenizer(self.text, truncation=True, padding="max_length", max_length=4, return_tensors="pt")

    def test_forward(self):
        for model_path in AUTO_CAUSAL_LM_PATHS:
            model = self._auto_model_class.from_pretrained(model_path, **self._supported_args)
            inputs = self._create_inputs(model_path)

            # Ensure that the `forward` method doesn't throw an error on generic inputs
            try:
                model(**inputs)
            except Exception as e:
                self.assertFalse(True, msg=e)

    def test_generate(self):
        for model_path in AUTO_CAUSAL_LM_PATHS:
            model = self._auto_model_class.from_pretrained(model_path, **self._supported_args)
            inputs = self._create_inputs(model_path)

            # Ensure that the `generate` method doesn't throw an error on generic inputs
            try:
                model.generate(**inputs)
            except Exception as e:
                self.assertFalse(True, msg=e)

    def test_save_load(self):
        for model_path in AUTO_CAUSAL_LM_PATHS:
            model = self._auto_model_class.from_pretrained(model_path, **self._supported_args)
            modified_model = copy.deepcopy(model)

            # Manually modify value head parameters
            modified_model.ilql_heads.q_heads[0][0].bias = torch.nn.Parameter(
                torch.ones_like(modified_model.ilql_heads.q_heads[0][0].bias) * 600053.34
            )

            with tempfile.TemporaryDirectory() as tmpdirname:
                modified_model.save_pretrained(tmpdirname)
                loaded_model = self._auto_model_class.from_pretrained(tmpdirname)

            # Check that the loaded model state dict is the same as the saved model state dict
            loaded_state_dict = loaded_model.state_dict()
            self.assertEqual(modified_model.state_dict().keys(), loaded_state_dict.keys())
            for name, saved_state in modified_model.state_dict().items():
                self.assertTrue(torch.all(torch.isclose(saved_state, loaded_state_dict[name])))

            # Assert loaded states are not the same as the original unmodified pretrained model
            self.assertFalse(
                torch.all(
                    torch.isclose(modified_model.ilql_heads.q_heads[0][0].bias, model.ilql_heads.q_heads[0][0].bias)
                )
            )

    def test_from_config(self):
        for model_path in AUTO_CAUSAL_LM_PATHS:
            config = transformers.AutoConfig.from_pretrained(model_path)
            # Modify the config to ensure the model is initialized from the custom config
            config.vocab_size = 2
            model = self._auto_model_class.from_config(config, **self._supported_args)
            self.assertEqual(model.base_model.get_output_embeddings().out_features, config.vocab_size)


class TestAutoModelForSeq2SeqLMWithILQLHeads(unittest.TestCase):
    _auto_model_class = AutoModelForSeq2SeqLMWithILQLHeads
    _supported_args = {"two_qs": True, "alpha": 0.8}  # TODO: Test various values

    def setUp(self):
        self.text = "Once upon a time there was a happy goose named Louis. He liked to eat bananas."

    def tearDown(self):
        gc.collect()  # Try to free up memory

    def _create_inputs(self, model_path):
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return tokenizer(self.text, truncation=True, padding="max_length", max_length=4, return_tensors="pt")

    def test_forward(self):
        for model_path in AUTO_SEQ2SEQ_LM_PATHS:
            model = self._auto_model_class.from_pretrained(model_path, **self._supported_args)
            inputs = self._create_inputs(model_path)

            # Ensure that the `forward` method doesn't throw an error on generic inputs
            try:
                model(**inputs)
            except Exception as e:
                self.assertFalse(True, msg=e)

    def test_generate(self):
        for model_path in AUTO_SEQ2SEQ_LM_PATHS:
            model = self._auto_model_class.from_pretrained(model_path, **self._supported_args)
            inputs = self._create_inputs(model_path)

            # Ensure that the `generate` method doesn't throw an error on generic inputs
            try:
                model.generate(**inputs)
            except Exception as e:
                self.assertFalse(True, msg=e)

    def test_save_load(self):
        for model_path in AUTO_SEQ2SEQ_LM_PATHS:
            model = self._auto_model_class.from_pretrained(model_path, **self._supported_args)
            modified_model = copy.deepcopy(model)

            # Manually modify value head parameters
            modified_model.ilql_heads.q_heads[0][0].bias = torch.nn.Parameter(
                torch.ones_like(modified_model.ilql_heads.q_heads[0][0].bias) * 600053.34
            )

            with tempfile.TemporaryDirectory() as tmpdirname:
                modified_model.save_pretrained(tmpdirname)
                loaded_model = self._auto_model_class.from_pretrained(tmpdirname)

            # Check that the loaded model state dict is the same as the saved model state dict
            loaded_state_dict = loaded_model.state_dict()
            self.assertEqual(modified_model.state_dict().keys(), loaded_state_dict.keys())
            for name, saved_state in modified_model.state_dict().items():
                self.assertTrue(torch.all(torch.isclose(saved_state, loaded_state_dict[name])))
            # Assert loaded states are not the same as the original unmodified pretrained model
            self.assertFalse(
                torch.all(
                    torch.isclose(modified_model.ilql_heads.q_heads[0][0].bias, model.ilql_heads.q_heads[0][0].bias)
                )
            )

    def test_from_config(self):
        for model_path in AUTO_SEQ2SEQ_LM_PATHS:
            config = transformers.AutoConfig.from_pretrained(model_path)
            # Modify the config to ensure the model is initialized from the custom config
            config.vocab_size = 2
            model = self._auto_model_class.from_config(config, **self._supported_args)
            self.assertEqual(model.base_model.get_output_embeddings().out_features, config.vocab_size)
