from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple, Union

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
    dialogue: Union[str, Iterable[str]], tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], max_length=2048
) -> List[DialogMessage]:
    """
    Tokenize sample with the interleaved form of (prompt_1, output_1, prompt_2, output_2...)
    """
    if isinstance(dialogue, str):
        bos_token = tokenizer.bos_token or tokenizer.eos_token
        dialogue = [bos_token, dialogue]
    elif isinstance(dialogue, Iterable):
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
    out = [t for t in truncated if len(t.tokens) > 0]

    if out[0].is_output:
        if sum(map(lambda msg: len(msg.tokens), out)) == max_length:
            if tokenizer.truncation_side == "left":
                out[0].tokens = out[0].tokens[1:]
            else:
                out[-1].tokens = out[-1].tokens[:-1]

        out.insert(0, DialogMessage(False, (tokenizer.bos_token_id,)))
    return out


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
    Dataloader which is used to supply prompts for either training or evaluation

    Args:
        prompts (`List[str]` or `List[Dict[str, Any]]`): list of raw text prompts or a dictionary with a required
            key `"prompt"` and extra information, that would be passed along the generation for that prompt as a
            keyword argument to a reward function.
        max_prompt_length (`int`): max length of the prompt, if exceeded the prompt will be truncated according to
            tokenizer's truncation setting.
        tokenizer (`transformers.PreTrainedTokenizer`): a tokenizer to tokenize prompts with.
        add_special_tokens (`bool`): whether to encode prompts with tokenizer's special tokens (passed directly
            into `tokenizer.encode`)
    """

    def __init__(
        self,
        prompts: Union[List[Dict[str, Any]], List[str]],
        max_prompt_length: int,
        tokenizer: PreTrainedTokenizer,
        add_special_tokens: bool = False,
    ):
        super().__init__()

        if isinstance(prompts[0], dict):
            metadata = prompts
            prompts = [x.pop("prompt") for x in metadata]
        else:
            metadata = [{}] * len(prompts)

        model_inputs = tokenizer(
            prompts, truncation=True, padding=False, max_length=max_prompt_length, add_special_tokens=add_special_tokens
        )

        prompts_tokens = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

        self.tokenizer = tokenizer
        self.prompts = [
            {"input_ids": tokens, "attention_mask": mask, **metadata}
            for tokens, mask, metadata in zip(prompts_tokens, attention_mask, metadata)
        ]

    def __getitem__(self, ix: int):
        return self.prompts[ix]

    def __len__(self) -> int:
        return len(self.prompts)

    def create_loader(self, batch_size: int, shuffle=False, sampler=None, drop_last=False) -> DataLoader:
        def collate_fn(xs):
            out = self.tokenizer.pad([{"input_ids": x["input_ids"]} for x in xs], return_tensors="pt")

            for key in xs[0]:
                if key != "input_ids" and key != "attention_mask":
                    out[key] = [x[key] for x in xs]

            return out

        # Since all data is already pre-processed, no need to have
        # multi-process data loading
        return DataLoader(
            self,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=0,
            drop_last=drop_last,
        )


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

    def create_loader(self, batch_size: int):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=ilql_collate_fn,
            drop_last=torch.distributed.is_initialized(),
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

    def create_loader(self, batch_size: int):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=ilql_seq2seq_collate_fn,
            drop_last=torch.distributed.is_initialized(),
        )


@dataclass
class DPOPreferences:
    prompt_tokens: Tuple
    chosen_tokens: Tuple
    rejected_tokens: Tuple


class DPOStore(BaseRolloutStore):
    # Adapted from TRL
    def __init__(self, preferences: List[DPOPreferences], tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.tokenizer = tokenizer

        self.history = [
            self._build_batch_from_preference_tokens(preference_element) for preference_element in preferences
        ]

    @staticmethod
    def tokenize_preferences(samples, tokenizer, max_length=2048):
        chosen_tokens = tokenizer(samples[0], add_special_tokens=False)
        rejected_tokens = tokenizer(samples[1], add_special_tokens=False)
        prompt_tokens = tokenizer(samples[2], add_special_tokens=False)

        chosen_tokens["input_ids"].append(tokenizer.eos_token_id)
        chosen_tokens["attention_mask"].append(1)

        rejected_tokens["input_ids"].append(tokenizer.eos_token_id)
        rejected_tokens["attention_mask"].append(1)

        longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

        # if combined sequence is too long, truncate the prompt only
        if len(prompt_tokens["input_ids"]) + longer_response_length > max_length:
            if tokenizer.truncation_side == "right":
                prompt_tokens = {k: v[: self.max_prompt_length] for k, v in prompt_tokens.items()}
            elif tokenizer.truncation_side == "left":
                prompt_tokens = {k: v[-self.max_prompt_length :] for k, v in prompt_tokens.items()}

        # if that's still too long, truncate the response
        if len(prompt_tokens["input_ids"]) + longer_response_length > max_length:
            chosen_tokens = {k: v[: max_length - max_prompt_length] for k, v in chosen_tokens.items()}
            rejected_tokens = {k: v[: max_length - max_prompt_length] for k, v in rejected_tokens.items()}

        return DPOPreferences(prompt_tokens=prompt_tokens, chosen_tokens=chosen_tokens, rejected_tokens=rejected_tokens)

    def _build_batch_from_preference_tokens(self, preference_tokens: DPOPreferences):
        # Create labels
        chosen_sequence_tokens = {
            k: preference_tokens.prompt_tokens[k] + preference_tokens.chosen_tokens[k]
            for k in preference_tokens.chosen_tokens
        }
        rejected_sequence_tokens = {
            k: preference_tokens.prompt_tokens[k] + preference_tokens.rejected_tokens[k]
            for k in preference_tokens.rejected_tokens
        }
        chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
        chosen_sequence_tokens["labels"][: len(preference_tokens.prompt_tokens["input_ids"])] = [
            self.label_pad_token_id
        ] * len(preference_tokens.prompt_tokens["input_ids"])
        rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
        rejected_sequence_tokens["labels"][: len(preference_tokens.prompt_tokens["input_ids"])] = [
            self.label_pad_token_id
        ] * len(preference_tokens.prompt_tokens["input_ids"])

        batch = {}

        for k, toks in {
            "chosen": chosen_sequence_tokens,
            "rejected": rejected_sequence_tokens,
            "prompt": preference_tokens.prompt_tokens,
        }.items():
            for type_key, tokens in toks.items():
                if type_key == "token_type_ids":
                    continue
                batch[f"{k}_{type_key}"] = tokens

        return batch

    def create_loader(self, batch_size: int, shuffle=False) -> DataLoader:
        def collate_fn(batch):
            # first, pad everything to the same length
            padded_batch = {}
            for k in batch[0].keys():
                if k.endswith("_input_ids") or k.endswith("_attention_mask") or k.endswith("_labels"):
                    # adapted from https://stackoverflow.com/questions/73256206
                    if "prompt" in k:
                        to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                    else:
                        to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                    if k.endswith("_input_ids"):
                        padding_value = self.tokenizer.pad_token_id
                    elif k.endswith("_labels"):
                        padding_value = self.label_pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = self.padding_value
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")

                    padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                    # for the prompt, flip back so padding is on left side
                    if "prompt" in k:
                        padded_batch[k] = padded_batch[k].flip(dims=[1])
                else:
                    padded_batch[k] = [ex[k] for ex in batch]

            return padded_batch

        return DataLoader(self, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)
