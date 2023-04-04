import copy
import gc
import tempfile
import unittest
from functools import lru_cache

import torch
import transformers
from hypothesis import given, settings
from hypothesis import strategies as st

from trlx.data.default_configs import default_ilql_config
from trlx.models.modeling_ilql import (
    AutoModelForCausalLMWithILQLHeads,
    AutoModelForSeq2SeqLMWithILQLHeads,
    ILQLBatch,
    ILQLConfig,
    ILQLHeads,
    batched_index_select,
)
from trlx.models.modeling_ppo import (
    AutoModelForCausalLMWithHydraValueHead,
    AutoModelForCausalLMWithValueHead,
    AutoModelForSeq2SeqLMWithHydraValueHead,
    AutoModelForSeq2SeqLMWithValueHead,
)
from trlx.trainer.accelerate_ilql_trainer import make_experience

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
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.padding_side = "left"
        tokenized = tokenizer(self.text, truncation=True, padding="max_length", max_length=4, return_tensors="pt")
        return dict(input_ids=tokenized.input_ids, attention_mask=tokenized.attention_mask)

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
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
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
        tokenizer.padding_side = "left"
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        inputs = tokenizer(self.text, truncation=True, padding="max_length", max_length=4, return_tensors="pt")
        inputs["decoder_input_ids"] = torch.tensor([[tokenizer.pad_token_id]])
        return inputs

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
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
            inputs = self._create_inputs(model_path)

            inputs["pad_token_id"] = tokenizer.pad_token_id
            inputs["eos_token_id"] = tokenizer.eos_token_id

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


@given(st.integers(1, 100), st.integers(1, 100), st.integers(0, 100), st.integers(1, 100))
def test_batched_index_select(batch, seq_len, num_idxes, hidden):
    x = torch.randn(batch, seq_len, hidden)
    if seq_len > 0:
        idxs = torch.randint(0, seq_len, (batch, num_idxes))
    else:
        idxs = torch.zeros(batch, num_idxes, dtype=torch.long)
    out = batched_index_select(x, idxs, dim=1)

    # Compute output using for loop
    out2 = torch.zeros(batch, num_idxes, hidden)
    for i in range(batch):
        out2[i] = x[i, idxs[i]]

    assert (out == out2).all()


@given(
    st.integers(1, 32),
    st.integers(1, 32),
    st.integers(0, 32),
    st.integers(0, 32),
    st.integers(1, 32),
    st.integers(1, 32),
    st.booleans(),
)
def test_ilql_heads_indexing(batch_size, seq_len, num_action_idxs, num_state_idxs, hidden_size, vocab_size, two_qs):
    heads = ILQLHeads(hidden_size, vocab_size, two_qs, alpha=1.0, dtype=torch.float32)

    # heads(hidden_states, states_ixs, actions_ixs) should
    # == heads(hidden_states) followed by indexing with states_ixs and actions_ixs

    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    states_ixs = torch.randint(0, seq_len, (batch_size, num_state_idxs))
    actions_ixs = torch.randint(0, seq_len, (batch_size, num_action_idxs))

    qs, target_qs, vs = heads(hidden_states, states_ixs, actions_ixs)
    qs2, target_qs2, vs2 = heads(hidden_states)

    assert len(qs2) == len(target_qs2) == len(qs)

    qs2 = tuple(batched_index_select(q, actions_ixs, dim=1) for q in qs2)
    target_qs2 = tuple(batched_index_select(q, actions_ixs, dim=1) for q in target_qs2)
    vs2 = batched_index_select(vs2, states_ixs, dim=1)

    assert all(torch.allclose(q, q2, atol=1e-06) for q, q2 in zip(qs, qs2))
    assert all(torch.allclose(q, q2, atol=1e-06) for q, q2 in zip(target_qs, target_qs2))
    assert torch.allclose(vs, vs2, atol=1e-06)


@given(
    st.integers(1, 32),
    st.integers(1, 32),
    st.integers(0, 32),
    st.integers(0, 32),
    st.integers(1, 32),
    st.integers(1, 32),
    st.booleans(),
)
def test_ilql_heads_output_count_and_shape(
    batch_size, seq_len, num_action_idxs, num_state_idxs, hidden_size, vocab_size, two_qs
):
    heads = ILQLHeads(hidden_size, vocab_size, two_qs, alpha=1.0, dtype=torch.float32)

    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    states_ixs = torch.randint(0, seq_len, (batch_size, num_state_idxs))
    actions_ixs = torch.randint(0, seq_len, (batch_size, num_action_idxs))

    qs, target_qs, vs = heads(hidden_states, states_ixs, actions_ixs)

    assert len(qs) == len(target_qs)

    assert qs[0].shape == (batch_size, num_action_idxs, vocab_size)
    assert target_qs[0].shape == (batch_size, num_action_idxs, vocab_size)
    assert vs.shape == (batch_size, num_state_idxs, 1)

    if two_qs:
        assert len(qs) == 2
        assert qs[1].shape == (batch_size, num_action_idxs, vocab_size)
        assert target_qs[1].shape == (batch_size, num_action_idxs, vocab_size)
    else:
        assert len(qs) == 1


@given(
    st.integers(1, 32),
    st.integers(1, 32),
    st.floats(0.0, 1.0),
    st.booleans(),
)
def test_ilql_heads_alpha(hidden_size, vocab_size, alpha, two_qs):
    heads = ILQLHeads(hidden_size, vocab_size, two_qs, alpha=alpha, dtype=torch.float32)

    for q_head in heads.q_heads:
        for param in q_head.parameters():
            param.data.copy_(torch.ones_like(param.data))

    for target_q_head in heads.target_q_heads:
        for param in target_q_head.parameters():
            param.data.copy_(torch.zeros_like(param.data))

    heads.sync_target_q_heads()

    for target_q_head in heads.target_q_heads:
        for param in target_q_head.parameters():
            assert torch.allclose(param.data, alpha * torch.ones_like(param.data), atol=1e-06)


@given(
    st.integers(1, 32),
    st.integers(1, 32),
    st.integers(1, 32),
    st.integers(1, 32),
    st.integers(1, 32),
    st.booleans(),
)
def test_ilql_loss_doesnt_crash(batch_size, seq_len, num_action_idxs, hidden_size, vocab_size, two_qs):
    ilql_config: ILQLConfig = default_ilql_config().method
    ilql_config.two_qs = two_qs

    num_state_idxs = num_action_idxs + 1

    heads = ILQLHeads(hidden_size, vocab_size, two_qs, alpha=1.0, dtype=torch.float32)

    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    states_ixs = torch.randint(0, seq_len, (batch_size, num_state_idxs))
    actions_ixs = torch.randint(0, seq_len, (batch_size, num_action_idxs))

    qs, target_qs, vs = heads(hidden_states, states_ixs, actions_ixs)
    logits = torch.randn(batch_size, seq_len, vocab_size)

    labels = ILQLBatch(
        input_ids=torch.randint(0, vocab_size, (batch_size, seq_len + 1)),
        attention_mask=torch.ones(batch_size, seq_len, dtype=torch.bool),
        rewards=torch.randn(batch_size, num_action_idxs),
        states_ixs=states_ixs,
        actions_ixs=actions_ixs,
        dones=torch.randint(0, 2, (batch_size, num_state_idxs), dtype=torch.bool),
    )

    loss_input = logits, (qs, target_qs, vs)
    loss, stats = ilql_config.loss(loss_input, labels)


@lru_cache
def cached_tokenizer():
    return transformers.AutoTokenizer.from_pretrained("gpt2")


@given(
    st.lists(st.tuples(st.text(min_size=1), st.floats(0.0, 1.0)), min_size=1),
    st.integers(1, 32),
    st.booleans(),
)
@settings(deadline=None)
def test_ilql_loss_make_experience_single_turn(samples_rewards, hidden_size, two_qs):
    samples, rewards = zip(*samples_rewards)
    batch_size = len(samples)
    rollouts = make_experience(samples, rewards, tokenizer=cached_tokenizer(), verbose=False)
    ilql_config: ILQLConfig = default_ilql_config().method

    loader = rollouts.create_loader(batch_size)
    ilql_batch = next(iter(loader))
    seq_len = ilql_batch.input_ids.shape[1]
    heads = ILQLHeads(hidden_size, 50257, two_qs, alpha=1.0, dtype=torch.float32)

    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    qs, target_qs, vs = heads(hidden_states, states_ixs=ilql_batch.states_ixs, actions_ixs=ilql_batch.actions_ixs)
    logits = torch.randn(batch_size, seq_len, 50257)

    loss, stats = ilql_config.loss((logits, (qs, target_qs, vs)), ilql_batch)
