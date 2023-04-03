from dataclasses import dataclass
from typing import Iterable, List, Tuple, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorWithPadding,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from trlx.data.ilql_types import (
    ILQLBatch,
    ILQLElement,
    ILQLSeq2SeqBatch,
    ILQLSeq2SeqElement,
)
from trlx.pipeline import BasePipeline, BaseRolloutStore, register_datapipeline


@dataclass
class DialogMessage:
    is_output: bool
    tokens: Tuple[int]


def tokenize_dialogue(  # noqa: C901
    dialogue: Union[str, List[str]], tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], max_length=2048
) -> List[DialogMessage]:
    """
    Tokenize sample with the interleaved form of (prompt_1, output_1, prompt_2, output_2...)
    """
    if isinstance(dialogue, str):
        bos_token = tokenizer.bos_token or tokenizer.eos_token
        dialogue = [bos_token, dialogue]
    elif isinstance(dialogue, tuple):
        if len(dialogue) % 2 != 0:
            raise ValueError("Dialogue must have an even number of phrases, alternating prompt and output")
        dialogue = list(dialogue)

    if not dialogue[-1].endswith(tokenizer.eos_token):
        dialogue[-1] = dialogue[-1] + tokenizer.eos_token

    tokenized = [
        DialogMessage(is_output=i % 2 == 1, tokens=tuple(tokenizer(dialogue[i], add_special_tokens=False).input_ids))
        for i in range(len(dialogue))
    ]

    # flip to truncate from the left
    if tokenizer.truncation_side == "left":
        tokenized = [DialogMessage(is_output=m.is_output, tokens=m.tokens[::-1]) for m in tokenized[::-1]]

    # truncate if necessary
    lengths = [len(t.tokens) for t in tokenized]
    cumsum_lengths = [sum(lengths[:i]) for i in range(len(lengths))]
    truncated = [
        DialogMessage(is_output=t.is_output, tokens=t.tokens[: max(max_length - cl, 0)])
        for t, cl in zip(tokenized, cumsum_lengths)
    ]

    # flip back if was fliped to left truncate
    if tokenizer.truncation_side == "left":
        truncated = [DialogMessage(is_output=m.is_output, tokens=m.tokens[::-1]) for m in truncated[::-1]]

    # remove empty messages
    truncated = [t for t in truncated if len(t.tokens) > 0]

    return truncated


class DialogStore(BaseRolloutStore):
    def __init__(self, dialogs: List[List[DialogMessage]], tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        attention_masks = [torch.ones(sum(len(m.tokens) for m in d), dtype=torch.bool) for d in dialogs]
        input_ids = [torch.tensor([t for m in d for t in m.tokens], dtype=torch.long) for d in dialogs]
        # -100 is the ignore index for CrossEntropyLoss
        labels = [
            torch.tensor([t if m.is_output else -100 for m in d for t in m.tokens], dtype=torch.long) for d in dialogs
        ]
        self.history = [
            dict(input_ids=i, attention_mask=a, labels=l) for i, a, l in zip(input_ids, attention_masks, labels)
        ]

    def create_loader(self, batch_size: int, shuffle=False) -> DataLoader:
        hf_collate_fn = DataCollatorWithPadding(self.tokenizer)

        def collate_fn(elems: Iterable[dict]):
            batch = hf_collate_fn(
                {"input_ids": [e["input_ids"] for e in elems], "attention_mask": [e["attention_mask"] for e in elems]}
            )
            labels = hf_collate_fn([{"input_ids": e["labels"]} for e in elems])["input_ids"]
            batch["labels"] = labels
            return batch

        return DataLoader(self, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)


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
