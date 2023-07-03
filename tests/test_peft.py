import copy
import gc
import os
import sys
import tempfile
import unittest
from typing import Optional

import numpy as np
import torch
import transformers
from peft import get_peft_config, get_peft_model
from peft.utils.config import PeftType, TaskType
from transformers import AutoConfig, AutoModelForCausalLM

from trlx.data.configs import TokenizerConfig
from trlx.data.default_configs import (
    ModelConfig,
    default_ilql_config,
    default_ppo_config,
    default_sft_config,
)
from trlx.models.modeling_ilql import (
    AutoModelForCausalLMWithILQLHeads,
    AutoModelForSeq2SeqLMWithILQLHeads,
)
from trlx.models.modeling_ppo import (
    AutoModelForCausalLMWithHydraValueHead,
    AutoModelForCausalLMWithValueHead,
    AutoModelForSeq2SeqLMWithHydraValueHead,
)
from trlx.trainer.accelerate_ilql_trainer import AccelerateILQLTrainer
from trlx.trainer.accelerate_ppo_trainer import AcceleratePPOTrainer
from trlx.trainer.accelerate_sft_trainer import AccelerateSFTTrainer

PPO = "ppo"
ILQL = "ilql"
SFT = "sft"
TRAINING_TYPES = [PPO, ILQL, SFT]

CAUSAL = "causal"
SEQ2SEQ = "seq2seq"

MODEL_TASK_TYPE = {
    "gpt2": CAUSAL,
    "google/t5-efficient-tiny": SEQ2SEQ,
    # "EleutherAI/pythia-160m": CAUSAL,
    # "facebook/opt-125m": CAUSAL,
}
MODELS_TO_TEST = list(MODEL_TASK_TYPE.keys())

PEFT_CONFIGS_TO_TEST = [PeftType.LORA, PeftType.PROMPT_TUNING, PeftType.PREFIX_TUNING]

ALL_TEST_COMBINATIONS = [
    [training_type, model_path, peft_type]
    for training_type in TRAINING_TYPES
    for model_path in MODELS_TO_TEST
    for peft_type in PEFT_CONFIGS_TO_TEST
    if [training_type, MODEL_TASK_TYPE[model_path]] != [SFT, SEQ2SEQ]  # Seq2Seq SFT not implemented
    and (MODEL_TASK_TYPE[model_path] != SEQ2SEQ or peft_type == PeftType.LORA)
    # Skip some tests due to implementation problems of peft 0.3.0 with Seq2Seq
]


class TestPeft(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    def tearDown(self):
        gc.collect()  # Try to free up memory

    def _create_model(
        self,
        training_type: str,
        model_path: str,
        task_type: str,
        peft_type: Optional[str],
        create_trainer: bool = False,
    ):
        self.peft_config = self._get_peft_config(peft_type, task_type) if peft_type else None
        if create_trainer:
            self.trainer = self._get_trainer(training_type, model_path, task_type, self.peft_config)
            self.model = self.trainer.model.to("cpu")
        else:
            # Should be a bit faster to execute than creating a trainer.
            if training_type == SFT:
                self.model = AutoModelForCausalLM.from_pretrained(model_path)
                if self.peft_config:
                    self.model = get_peft_model(self.model, self.peft_config)
            else:
                self.model = self._get_auto_model_type(training_type, task_type).from_pretrained(
                    model_path,
                    peft_config=self.peft_config,
                )

        self._create_inputs(model_path, task_type)

    def _create_inputs(self, tokenizer_path, task_type):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)

        if task_type == CAUSAL:
            self.inputs = self.tokenizer(
                "Once upon a time there was a happy goose named Louis. He liked to eat bananas and",
                return_tensors="pt",
            )
        elif task_type == SEQ2SEQ:
            self.encoder_text = "Translate this text to French: Hello, my dog is cute"
            self.decoder_text = "Bonjour, mon chien est mignon"
            encoder_inputs = self.tokenizer(self.encoder_text, return_tensors="pt")
            decoder_inputs = self.tokenizer(self.decoder_text, return_tensors="pt")
            self.inputs = {
                **encoder_inputs,
                "decoder_input_ids": decoder_inputs.input_ids,
                "decoder_attention_mask": decoder_inputs.attention_mask,
            }
        else:
            # Classification tasks not implemented
            raise NotImplementedError

    def _get_trainer(self, training_type, model_path: str, task_type: str, peft_config, tokenizer_path: str = None):
        if training_type == PPO:
            config = default_ppo_config()
            trainer_type = AcceleratePPOTrainer
        elif training_type == ILQL:
            config = default_ilql_config()
            trainer_type = AccelerateILQLTrainer
        elif training_type == SFT:
            config = default_sft_config()
            trainer_type = AccelerateSFTTrainer
        else:
            raise ValueError(f"Training type {training_type} not recognized.")

        config.tokenizer = TokenizerConfig(tokenizer_path=tokenizer_path if tokenizer_path else model_path)
        config.model = ModelConfig(model_path=model_path, peft_config=peft_config, model_arch_type=task_type)
        config.train.tracker = None

        return trainer_type(config)

    def _get_auto_model_type(self, training_type, task_type):
        if training_type == PPO:
            if task_type == CAUSAL:
                return AutoModelForCausalLMWithHydraValueHead
            elif task_type == SEQ2SEQ:
                return AutoModelForSeq2SeqLMWithHydraValueHead
        elif training_type == ILQL:
            if task_type == CAUSAL:
                return AutoModelForCausalLMWithILQLHeads
            elif task_type == SEQ2SEQ:
                return AutoModelForSeq2SeqLMWithILQLHeads
        elif training_type == SFT and task_type == CAUSAL:
            return AutoModelForCausalLM

        raise ValueError(f"Training type {training_type} for the task {task_type} not recognized.")

    def _get_peft_config(self, peft_type: str, task_type: str):
        assert task_type in [CAUSAL, SEQ2SEQ]
        task_type = TaskType.CAUSAL_LM if task_type == "causal" else TaskType.SEQ_2_SEQ_LM

        if peft_type == PeftType.LORA:
            return get_peft_config(
                {
                    "peft_type": peft_type,
                    "task_type": task_type,
                    "r": 8,
                    "lora_alpha": 32,
                    "lora_dropout": 0.0,
                }
            )
        elif peft_type == PeftType.PREFIX_TUNING:
            return get_peft_config(
                {
                    "peft_type": peft_type,
                    "task_type": task_type,
                    "num_virtual_tokens": 10,
                }
            )
        elif peft_type == PeftType.PROMPT_TUNING:
            return get_peft_config(
                {
                    "peft_type": peft_type,
                    "task_type": task_type,
                    "prompt_tuning_init": "RANDOM",
                    "num_virtual_tokens": 10,
                }
            )
        else:
            raise NotImplementedError

    def _backprop(self, model):
        output = model(**self.inputs, return_dict=True)
        # Just apply an arbitrary loss to cause whatever change in the model's parameters.
        # This loss doesn't make sense, but it causes a gradient, so it's fine.
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            output.logits[0][-1][:1],
            torch.tensor([0.53]),
        )

        if hasattr(output, "value"):
            loss += torch.nn.functional.binary_cross_entropy_with_logits(
                output.value.squeeze()[-1:],
                torch.tensor([0.53]),
            )

        loss.backward()

        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer.step()

        return model

    def _check_that_models_are_equivalent(self, model1, model2, training_type, test_hydra=False):
        self.assertTrue(
            torch.equal(model1(**self.inputs, return_dict=True).logits, model2(**self.inputs, return_dict=True).logits)
        )

        state_dict1 = model1.state_dict()
        state_dict2 = model2.state_dict()
        self.assertEqual(state_dict1.keys(), state_dict2.keys())
        for name in state_dict1.keys():
            self.assertTrue(torch.equal(state_dict1[name], state_dict2[name]))

        if training_type != SFT:
            self.assertTrue(
                torch.equal(
                    model1(**self.inputs, return_dict=True).value,
                    model2(**self.inputs, return_dict=True).value,
                )
            )

        if training_type == PPO and test_hydra:
            self.assertTrue(
                torch.equal(
                    model1.forward_hydra(**self.inputs, return_dict=True).logits,
                    model2.forward_hydra(**self.inputs, return_dict=True).logits,
                )
            )

    def test_save_and_load(self):
        for training_type in [PPO, ILQL]:
            for model_path in MODELS_TO_TEST:
                peft_type = PeftType.LORA
                task_type = MODEL_TASK_TYPE[model_path]
                self._create_model(training_type, model_path, task_type, peft_type)
                self._backprop(self.model)

                with tempfile.TemporaryDirectory() as tmp_dir:
                    self.model.save_pretrained(tmp_dir)

                    self.assertTrue(os.path.isfile(f"{tmp_dir}/adapter_model.bin"))
                    self.assertTrue(os.path.isfile(f"{tmp_dir}/adapter_config.json"))
                    self.assertTrue(os.path.isfile(f"{tmp_dir}/pytorch_model.bin"))

                    # Check that it didn't save the whole model (which weights around 500MB)
                    # pytorch_model.bin should only contain the other trained parts like the value heads.
                    # ILQL heads are very big though (around 1.1GB for gpt2).
                    self.assertLess(os.path.getsize(f"{tmp_dir}/pytorch_model.bin"), 1.3e9 if ILQL else 1e7)

                    auto_model_type = self._get_auto_model_type(training_type, task_type)

                    loaded_model = auto_model_type.from_pretrained(tmp_dir)
                    self._check_that_models_are_equivalent(loaded_model, self.model, training_type, True)

    def test_from_config(self):
        """Check that from_config will add a peft adapter if given the argument peft_config"""
        for training_type in TRAINING_TYPES:
            peft_config = self._get_peft_config(PeftType.LORA, CAUSAL)
            gpt2_config = AutoConfig.from_pretrained("gpt2")
            trainer = self._get_trainer(training_type, gpt2_config, CAUSAL, peft_config, tokenizer_path="gpt2")
            state_dict = trainer.model.state_dict()

            self.assertTrue(any(["lora" in layer_name for layer_name in state_dict.keys()]))

    def test_save_and_load_without_peft(self):
        """Similar to test_save_load, but with peft not installed. Should not raise any error."""
        with unittest.mock.patch.dict(sys.modules, {"peft": None}):
            for training_type in [PPO, ILQL]:
                for model_path in MODELS_TO_TEST:
                    task_type = MODEL_TASK_TYPE[model_path]
                    self._create_model(training_type, model_path, task_type, peft_type=None)
                    self._backprop(self.model)

                    with tempfile.TemporaryDirectory() as tmp_dir:
                        self.model.save_pretrained(tmp_dir)
                        auto_model_type = self._get_auto_model_type(training_type, task_type)

                        loaded_model = auto_model_type.from_pretrained(tmp_dir)
                        self._check_that_models_are_equivalent(loaded_model, self.model, training_type)

    def test_backpropagation_and_disabling(self):
        for training_type, model_path, peft_type in ALL_TEST_COMBINATIONS:
            task_type = MODEL_TASK_TYPE[model_path]
            self._create_model(training_type, model_path, task_type, peft_type, create_trainer=True)
            old_logits = self.model(**self.inputs, return_dict=True).logits
            initial_model_state_dict = copy.deepcopy(self.model.state_dict())

            self._backprop(self.model)
            self._backprop(self.model)
            new_logits = self.model(**self.inputs, return_dict=True).logits
            new_model_state_dict = self.model.state_dict()

            # Check that the backpropagation affected the predictions
            self.assertFalse(torch.equal(old_logits, new_logits))

            # Check that only the peft adapter layers are modified by the backpropagation
            self.assertEqual(initial_model_state_dict.keys(), new_model_state_dict.keys())
            for name in initial_model_state_dict.keys():
                parameters_equal = torch.equal(initial_model_state_dict[name], new_model_state_dict[name])
                if "lora" in name or "prompt" in name or "v_head" in name:
                    self.assertFalse(parameters_equal)
                else:
                    self.assertTrue(parameters_equal)

            # Check Lora enabling and disabling
            if "LORA" in peft_type:
                # If disabling the Lora adapter restores the original logits,
                # this shows that the backpropagation only affected the Lora adapter
                self.lora_model = self.model.base_model if training_type != SFT else self.model
                self.lora_model.disable_adapter_layers()
                new_logits = self.model(**self.inputs, return_dict=True).logits
                self.assertTrue(torch.equal(old_logits, new_logits))

                # Re-enabling the Lora adapter should make the 2 models different again
                self.lora_model.enable_adapter_layers()
                new_logits = self.model(**self.inputs, return_dict=True).logits
                self.assertFalse(torch.equal(old_logits, new_logits))

    def test_forward_hydra(self):
        """Test that PPO hydra heads work and give similar logits to the model without any fine-tuning."""
        for model_path in MODELS_TO_TEST:
            for peft_type in PEFT_CONFIGS_TO_TEST:
                task_type = MODEL_TASK_TYPE[model_path]
                if task_type == SEQ2SEQ and peft_type != PeftType.LORA:
                    continue  # TODO: pass some tests due to some bugs in peft 0.3.0 with Seq2Seq

                self._create_model(PPO, model_path, task_type, peft_type)

                logits_without_peft = self.model.base_model.base_model(**self.inputs, return_dict=True).logits
                logits_before_backpropagation = self.model(**self.inputs, return_dict=True).logits

                self._backprop(self.model)

                # forward_hydra should return the same logits as the original model
                new_logits_from_hydra = self.model.forward_hydra(**self.inputs, return_dict=True).logits
                self.assertTrue(torch.equal(logits_without_peft, new_logits_from_hydra))

                if "LORA" in peft_type:
                    # True because the Lora adapter initially does not modify the output
                    self.assertTrue(torch.equal(logits_before_backpropagation, new_logits_from_hydra))
                else:
                    # False because the initial prompt before backpropagation
                    # was used to calculate logits_before_backpropagation, but not for new_logits_from_hydra.
                    self.assertFalse(torch.equal(logits_before_backpropagation, new_logits_from_hydra))

    def test_generate(self):
        """
        Check that generate works, and that it's deterministic when the temperature is very low.
        """
        temperature = 0.0

        for training_type, model_path, peft_type in ALL_TEST_COMBINATIONS:
            task_type = MODEL_TASK_TYPE[model_path]
            self._create_model(training_type, model_path, task_type, peft_type)
            self._backprop(self.model)
            with torch.no_grad():
                output1 = self.model.generate(
                    **self.inputs,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

                output2 = self.model.generate(
                    **self.inputs,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                self.assertTrue(torch.equal(output1, output2))

    def test_peft_not_installed_error(self):
        """If the argument peft_config is used but peft is not installed, expect a ModuleNotFoundError"""
        with unittest.mock.patch.dict(sys.modules, {"peft": None}):
            peft_config = {"peft_type": "LORA"}

            with self.assertRaises(ModuleNotFoundError):
                self._get_trainer(PPO, "gpt2", CAUSAL, peft_config)

            with self.assertRaises(ModuleNotFoundError):
                AutoModelForCausalLMWithHydraValueHead.from_pretrained("gpt2", peft_config=peft_config)

    def test_lora_modules_to_save(self):
        """
        Test the special Lora config option 'modules_to_save'.
        It allows also train some non-lora modules, and its implementation is a bit tricky.
        """
        for training_type in [PPO, ILQL]:
            trainable_layer_name = "base_model.model.transformer.h.3.mlp"

            peft_config = {
                "peft_type": PeftType.LORA,
                "task_type": CAUSAL,
                "r": 8,
                "lora_alpha": 32,
                "lora_dropout": 0.0,
                "modules_to_save": [trainable_layer_name],
            }

            model = self._get_auto_model_type(training_type, CAUSAL).from_pretrained("gpt2", peft_config=peft_config)
            initial_state_dict = copy.deepcopy(model.state_dict())
            self._create_inputs("gpt2", CAUSAL)
            # initial_logits = model(**self.inputs, return_dict=True).logits

            self._backprop(model)
            self._backprop(model)
            new_state_dict = model.state_dict()

            self.assertEqual(initial_state_dict.keys(), new_state_dict.keys())
            for name in initial_state_dict.keys():
                parameters_equal = torch.equal(initial_state_dict[name], new_state_dict[name])
                if trainable_layer_name + ".modules_to_save" in name or "lora" in name or "v_head" in name:
                    self.assertFalse(parameters_equal)
                else:
                    self.assertTrue(parameters_equal)

            # TODO: deactivated until the issue (https://github.com/huggingface/peft/issues/493) is fixed
            # if training_type == PPO:
            #     forward_hydra_logits = model.forward_hydra(**self.inputs, return_dict=True).logits
            #     self.assertTrue(torch.equal(initial_logits, forward_hydra_logits))

            trained_model_logits = model(**self.inputs, return_dict=True).logits
            with tempfile.TemporaryDirectory() as tmp_dir:
                model.save_pretrained(tmp_dir)
                loaded_model = self._get_auto_model_type(training_type, CAUSAL).from_pretrained(tmp_dir)
                loaded_model_logits = loaded_model(**self.inputs, return_dict=True).logits
                self.assertTrue(torch.equal(trained_model_logits, loaded_model_logits))

    # @unittest.skipUnless(
    #     importlib.util.find_spec("bitsandbytes") and torch.cuda.is_available(),
    #     "bitsandbytes and GPU needed to execute test_8bits",
    # )
    @unittest.skip("`8-bit` model loading support is not yet fully implemented")
    def test_8bits(self):
        """Test the behaviour of from_pretrained with 8 bits models"""
        from bitsandbytes.nn import Linear8bitLt

        # gpt2 uses Conv1D instead of Linear, so use pythia-160m instead.
        model_id = "EleutherAI/pythia-160m"

        peft_config = {
            "peft_type": PeftType.LORA,
            "task_type": TaskType.CAUSAL_LM,
            "lora_dropout": 0.0,
            "lora_alpha": 32,
        }
        reference_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_id,
            peft_config=peft_config,
        )
        initial_nb_trainable_params = sum(p.numel() for p in reference_model.parameters() if p.requires_grad)

        model_8bit = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_id,
            peft_config=peft_config,
            load_in_8bit=True,
            peft_int8_kwargs={"use_gradient_checkpointing": True},
            device_map="auto",
        )

        new_nb_trainable_params = sum(p.numel() for p in model_8bit.parameters() if p.requires_grad)
        self.assertEqual(new_nb_trainable_params, initial_nb_trainable_params)

        self.assertIsInstance(reference_model.base_model.model.gpt_neox.layers[0].mlp.dense_h_to_4h, torch.nn.Linear)
        self.assertIsInstance(model_8bit.base_model.model.gpt_neox.layers[0].mlp.dense_h_to_4h, Linear8bitLt)

        base_model = AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit=True, device_map="auto")
        model_8bit = AutoModelForCausalLMWithValueHead.from_pretrained(
            base_model,
            peft_config=peft_config,
            load_in_8bit=True,
            peft_int8_kwargs={"use_gradient_checkpointing": False},
            device_map="auto",
        )

        new_nb_trainable_params = sum(p.numel() for p in model_8bit.parameters() if p.requires_grad)
        self.assertEqual(new_nb_trainable_params, initial_nb_trainable_params)

        self.assertIsInstance(model_8bit.base_model.model.gpt_neox.layers[0].mlp.dense_h_to_4h, Linear8bitLt)
