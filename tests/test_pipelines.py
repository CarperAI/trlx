from unittest import TestCase

from hypothesis import given
from hypothesis import strategies as st
from transformers import AutoTokenizer

from trlx.pipeline.offline_pipeline import DialogMessage, tokenize_dialogue


class TestTokenizeDialog(TestCase):
    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

    @given(st.lists(st.text(), max_size=32))
    def test_tokenize_dialogue_single_turn(self, response_words):
        response = " ".join(response_words)  # space seperate to make it multiple tokens
        tokenized_response = tuple(self.tokenizer(response, add_special_tokens=False).input_ids)
        tokenized_response = tokenized_response + (self.tokenizer.eos_token_id,)
        dialog = tokenize_dialogue(response, self.tokenizer)

        assert len(dialog) == 2
        user_dm, bot_dm = dialog

        assert user_dm == DialogMessage(is_output=False, tokens=(self.tokenizer.bos_token_id,))
        assert bot_dm == DialogMessage(is_output=True, tokens=tokenized_response)

    @given(st.lists(st.text(), max_size=32), st.integers(min_value=2, max_value=16))
    def test_tokenize_dialogue_single_turn_truncation_right(self, response_words, max_length):
        response = " ".join(response_words)  # space seperate to make it multiple tokens
        self.tokenizer.truncation_side = "right"
        tokenized_response = tuple(self.tokenizer(response, add_special_tokens=False).input_ids)
        tokenized_response = tokenized_response + (self.tokenizer.eos_token_id,)
        dialog = tokenize_dialogue(response, self.tokenizer, max_length=max_length)

        assert len(dialog) == 2
        user_dm, bot_dm = dialog

        assert user_dm == DialogMessage(is_output=False, tokens=(self.tokenizer.bos_token_id,))
        assert bot_dm == DialogMessage(is_output=True, tokens=tokenized_response[: max_length - 1])

        all_tokens = sum((dm.tokens for dm in dialog), ())
        assert len(all_tokens) <= max_length

    @given(st.lists(st.text(), max_size=32), st.integers(min_value=2, max_value=16))
    def test_tokenize_dialogue_single_turn_truncation_left(self, response_words, max_length):
        response = " ".join(response_words)  # space seperate to make it multiple tokens
        self.tokenizer.truncation_side = "left"
        tokenized_response = tuple(self.tokenizer(response, add_special_tokens=False).input_ids)
        tokenized_response = tokenized_response + (self.tokenizer.eos_token_id,)
        dialog = tokenize_dialogue(response, self.tokenizer, max_length=max_length)

        # if no truncation should have happened, then the user BOS prompt should be present
        if len(tokenized_response) + 1 <= max_length:
            assert len(dialog) == 2
            user_dm, bot_dm = dialog

            assert user_dm == DialogMessage(is_output=False, tokens=(self.tokenizer.bos_token_id,))
            assert bot_dm == DialogMessage(is_output=True, tokens=tokenized_response)
        else:
            assert len(dialog) == 1
            bot_dm = dialog[0]
            assert bot_dm == DialogMessage(is_output=True, tokens=tokenized_response[-max_length:])

        all_tokens = sum((dm.tokens for dm in dialog), ())
        assert len(all_tokens) <= max_length

    @given(st.lists(st.tuples(st.text(), st.text()), min_size=1, max_size=32))
    def test_tokenize_dialogue_multi_turn(self, user_response_pairs):
        convo = [[" ".join(user_words), " ".join(response_words)] for user_words, response_words in user_response_pairs]
        flat_convo = sum(convo, [])
        tokenized_flat_convo = tuple(
            tuple(self.tokenizer(turn, add_special_tokens=False).input_ids) for turn in flat_convo
        )
        tokenized_flat_convo = (*tokenized_flat_convo[:-1], (*tokenized_flat_convo[-1], self.tokenizer.eos_token_id))
        dialog = tokenize_dialogue(flat_convo, self.tokenizer)

        dm_convo = [DialogMessage(is_output=i % 2 == 1, tokens=tokens) for i, tokens in enumerate(tokenized_flat_convo)]
        nonempty_dm_convo = [dm for dm in dm_convo if dm.tokens]
        assert dialog == nonempty_dm_convo

    @given(st.lists(st.tuples(st.text(), st.text()), min_size=1, max_size=32), st.integers(min_value=2, max_value=16))
    def test_tokenize_dialogue_multi_turn_truncation_right(self, user_response_pairs, max_length):
        convo = [[" ".join(user_words), " ".join(response_words)] for user_words, response_words in user_response_pairs]
        flat_convo = sum(convo, [])
        self.tokenizer.truncation_side = "right"
        tokenized_flat_convo = tuple(
            tuple(self.tokenizer(turn, add_special_tokens=False).input_ids) for turn in flat_convo
        )
        tokenized_flat_convo = (*tokenized_flat_convo[:-1], (*tokenized_flat_convo[-1], self.tokenizer.eos_token_id))
        dialog = tokenize_dialogue(flat_convo, self.tokenizer, max_length=max_length)

        all_tokens = sum((dm.tokens for dm in dialog), ())
        should_be_tokens = sum(tokenized_flat_convo, ())[:max_length]
        assert all_tokens == should_be_tokens
        assert len(all_tokens) <= max_length

    @given(st.lists(st.tuples(st.text(), st.text()), min_size=1, max_size=32), st.integers(min_value=2, max_value=16))
    def test_tokenize_dialogue_multi_turn_truncation_left(self, user_response_pairs, max_length):
        convo = [[" ".join(user_words), " ".join(response_words)] for user_words, response_words in user_response_pairs]
        flat_convo = sum(convo, [])
        self.tokenizer.truncation_side = "left"
        tokenized_flat_convo = tuple(
            tuple(self.tokenizer(turn, add_special_tokens=False).input_ids) for turn in flat_convo
        )
        tokenized_flat_convo = (*tokenized_flat_convo[:-1], (*tokenized_flat_convo[-1], self.tokenizer.eos_token_id))
        dialog = tokenize_dialogue(flat_convo, self.tokenizer, max_length=max_length)

        all_tokens = sum((dm.tokens for dm in dialog), ())

        should_be_tokens = sum(tokenized_flat_convo, ())[-max_length:]
        assert all_tokens == should_be_tokens

        assert len(all_tokens) <= max_length
