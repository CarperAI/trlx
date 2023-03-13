from hypothesis import given
from hypothesis import strategies as st
from transformers import AutoTokenizer

from trlx.pipeline.offline_pipeline import tokenize_dialogue


@given(st.lists(st.text(), max_size=32))
def test_tokenize_dialogue_single_turn(response_words):
    response = " ".join(response_words)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenized_response = tokenizer(response, add_special_tokens=False).input_ids
    dialog_tokens = tokenize_dialogue(response, tokenizer)

    assert len(dialog_tokens) == 2
    assert dialog_tokens[0] == [tokenizer.bos_token_id]
    assert dialog_tokens[1] == tokenized_response + [tokenizer.eos_token_id]


@given(st.lists(st.text(), max_size=32), st.integers(min_value=1, max_value=16))
def test_tokenize_dialogue_single_turn_truncation_right(response_words, max_length):
    response = " ".join(response_words)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.truncation_side = "right"
    tokenized_response = tokenizer(response, add_special_tokens=False).input_ids
    dialog_tokens = tokenize_dialogue(response, tokenizer, max_length=max_length)

    assert len(dialog_tokens) == 2
    assert dialog_tokens[0] == [tokenizer.bos_token_id]
    assert dialog_tokens[1] == tokenized_response[: max_length - 1] + [tokenizer.eos_token_id]

    all_tokens = sum(dialog_tokens, [])
    assert len(all_tokens) <= max_length


@given(st.lists(st.text(), max_size=32), st.integers(min_value=1, max_value=16))
def test_tokenize_dialogue_single_turn_truncation_left(response_words, max_length):
    response = " ".join(response_words)  # space seperate to make it multiple tokens
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.truncation_side = "left"
    tokenized_response = tokenizer(response, add_special_tokens=False).input_ids
    dialog_tokens = tokenize_dialogue(response, tokenizer, max_length=max_length)

    assert len(dialog_tokens) == 2
    assert dialog_tokens[0] == [tokenizer.bos_token_id]
    assert dialog_tokens[1] == tokenized_response[-max_length + 1 :] + [tokenizer.eos_token_id]

    all_tokens = sum(dialog_tokens, [])
    assert len(all_tokens) <= max_length
