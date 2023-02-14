from typing import Iterable, List, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, PreTrainedTokenizer

from trlx.data.ilql_types import (
    ILQLBatch,
    ILQLElement,
    ILQLSeq2SeqBatch,
    ILQLSeq2SeqElement,
)
from trlx.pipeline import BasePipeline, BaseRolloutStore, register_datapipeline


def tokenize_dialogue(dialogue: Union[str, List[str]], tokenizer, max_length=2048) -> List[int]:  # noqa: C901
    """
    Tokenize sample with the interleaved form of (prompt_1, output_1, prompt_2, output_2...)
    """
    if isinstance(dialogue, str):
        dialogue = [tokenizer.bos_token, dialogue]
    elif isinstance(dialogue, tuple):
        dialogue = list(dialogue)
    dialogue[-1] += tokenizer.eos_token

    out = []
    ctx_length = max_length
    if tokenizer.truncation_side == "left":
        for phrase in reversed(dialogue):
            # Manually added BOS and EOS above so we don't want to add special tokens here
            tokens = tokenizer(phrase, add_special_tokens=False).input_ids[-ctx_length:]
            ctx_length -= len(tokens)
            out.insert(0, tokens)
            if ctx_length == 0:
                break

        # in case of odd number of phrases (possibly due to truncation)
        # since the first phrase always has to be a prompt, force it to be <bos>
        if len(out) % 2 == 1:
            if sum(map(len, out)) == max_length:
                out[0].pop(0)
            out.insert(0, [tokenizer.bos_token_id])

    elif tokenizer.truncation_side == "right":
        for phrase in dialogue:
            # Manually added BOS and EOS above so we don't want to add special tokens here
            tokens = tokenizer(phrase, add_special_tokens=False).input_ids[:ctx_length]
            ctx_length -= len(tokens)
            out.append(tokens)
            if ctx_length == 0:
                break
    return out


@register_datapipeline
class PromptPipeline(BasePipeline):
    """
    Tokenizes prompts, unless they are already tokenized, and truncates them to `max_prompt_length` from the right
    """

    def __init__(self, prompts: List[str], max_prompt_length: int, tokenizer: PreTrainedTokenizer):
        super().__init__()

        model_inputs = tokenizer(
            prompts, truncation=True, padding=False, max_length=max_prompt_length, add_special_tokens=False
        )

        prompts_tokens = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

        self.tokenizer = tokenizer
        self.prompts = [
            {"input_ids": tokens, "attention_mask": mask} for tokens, mask in zip(prompts_tokens, attention_mask)
        ]

    def __getitem__(self, ix: int):
        return self.prompts[ix]

    def __len__(self) -> int:
        return len(self.prompts)

    def create_loader(self, batch_size: int, shuffle=False) -> DataLoader:
        collate_fn = DataCollatorWithPadding(self.tokenizer) if self.tokenizer else torch.vstack
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)


def ilql_collate_fn(elems: Iterable[ILQLElement]):
    return ILQLBatch(
        pad_sequence([x.input_ids for x in elems], batch_first=True, padding_value=0),
        pad_sequence([x.attention_mask for x in elems], batch_first=True, padding_value=0),
        pad_sequence([x.rewards for x in elems], batch_first=True, padding_value=0.0),
        pad_sequence([x.states_ixs for x in elems], batch_first=True, padding_value=0),
        pad_sequence([x.actions_ixs for x in elems], batch_first=True, padding_value=0),
        pad_sequence([x.dones for x in elems], batch_first=True, padding_value=0),
    )


class ILQLRolloutStorage(BaseRolloutStore):
    """
    Rollout storage for training ILQL
    """

    def __init__(self, input_ids, attention_mask, rewards, states_ixs, actions_ixs, dones):
        super().__init__()

        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.rewards = rewards
        self.states_ixs = states_ixs
        self.actions_ixs = actions_ixs
        self.dones = dones

    def __getitem__(self, ix: int) -> ILQLElement:
        return ILQLElement(
            self.input_ids[ix],
            self.attention_mask[ix],
            self.rewards[ix],
            self.states_ixs[ix],
            self.actions_ixs[ix],
            self.dones[ix],
        )

    def __len__(self) -> int:
        return len(self.input_ids)

    def create_loader(self, batch_size: int, drop_last=True):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=ilql_collate_fn,
            drop_last=drop_last,
        )


def ilql_seq2seq_collate_fn(elems: Iterable[ILQLElement]):
    return ILQLSeq2SeqBatch(
        pad_sequence([x.input_ids for x in elems], batch_first=True, padding_value=0),
        pad_sequence([x.attention_mask for x in elems], batch_first=True, padding_value=0),
        pad_sequence([x.decoder_input_ids for x in elems], batch_first=True, padding_value=0),
        pad_sequence([x.rewards for x in elems], batch_first=True, padding_value=0.0),
        pad_sequence([x.states_ixs for x in elems], batch_first=True, padding_value=0),
        pad_sequence([x.actions_ixs for x in elems], batch_first=True, padding_value=0),
        pad_sequence([x.dones for x in elems], batch_first=True, padding_value=0),
    )


class ILQLSeq2SeqRolloutStorage(BaseRolloutStore):
    """
    Rollout storage for training ILQL
    """

    def __init__(self, input_ids, attention_mask, decoder_input_ids, rewards, states_ixs, actions_ixs, dones):
        super().__init__()

        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.decoder_input_ids = decoder_input_ids
        self.rewards = rewards
        self.states_ixs = states_ixs
        self.actions_ixs = actions_ixs
        self.dones = dones

    def __getitem__(self, ix: int) -> ILQLElement:
        return ILQLSeq2SeqElement(
            self.input_ids[ix],
            self.attention_mask[ix],
            self.decoder_input_ids[ix],
            self.rewards[ix],
            self.states_ixs[ix],
            self.actions_ixs[ix],
            self.dones[ix],
        )

    def __len__(self) -> int:
        return len(self.input_ids)

    def create_loader(self, batch_size: int, drop_last=True):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=ilql_seq2seq_collate_fn,
            drop_last=drop_last,
        )
