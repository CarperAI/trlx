from torch.utils.data import dataset
from datasets import load_dataset
import torch
from datasets import load_metric
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq
)

MAX_LENGTH_INPUT = 16
MAX_LENGTH_OUTPUT = 128

from tqdm import tqdm

class MethodDataset(dataset.Dataset):

    def __init__(
        self, tokenizer, mode, block_size: int = 256, overwrite_cache=False, local_rank=-1,
    ):
        self.tokenizer = tokenizer
        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)
        self.examples = []
        if mode == 'train':
            imdb = load_dataset("imdb", split="train+test")
        else:
            imdb = load_dataset("imdb", split="test").select(range(100))
        for input_text in tqdm(list(imdb["text"])):
            model_input = self.tokenizer(
                input_text,
                max_length=block_size, 
                padding='max_length',
                truncation=True
            )
            encodings = {
                'input_ids': torch.tensor(model_input['input_ids'] + [tokenizer.eos_token_id], dtype = torch.long), 
                'attention_mask': torch.tensor(model_input['attention_mask'] + [tokenizer.eos_token_id], dtype = torch.long),
             }
            self.examples.append(encodings)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return self.examples[i]

import os
import logging
import pickle
import random
import time

from dataclasses import dataclass, field

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import (
    BatchEncoding,
    DataCollator,
    DataCollatorForLanguageModeling,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from typing import Any, Callable, Dict, List, NewType, Tuple, Union

# Translated from: https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py
@dataclass
class DataCollatorForSeq2SeqMaskLanguageModeling:
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """

    tokenizer: PreTrainedTokenizer
    mlm: bool = True
    mlm_probability: float = 0.15

    def __call__(self, examples: List[Union[torch.Tensor, Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        if isinstance(examples[0], (dict, BatchEncoding)):
            examples = [e["input_ids"] for e in examples]
        batch = self._tensorize_batch(examples)
        if self.mlm:
            inputs, labels = self.mask_tokens(batch)
            return {"input_ids": inputs, "labels": labels}
        else:
            labels = batch.clone().detach()
            labels[labels == self.tokenizer.pad_token_id] = -100
            return {"input_ids": batch, "labels": labels}

    def _tensorize_batch(self, examples: List[torch.Tensor]) -> torch.Tensor:
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)

    def _noise_span_to_unique_sentinel(self, tokens, mask, max_sentinels, sentinel_id):
        sentineled_toks = tokens.clone()
        prev_tok_noise = torch.nn.functional.pad(mask[:-1], [1, 0])

        first_noise_toks = torch.logical_and(mask, ~prev_tok_noise)
        subse_noise_toks = torch.logical_and(mask, prev_tok_noise)
        
        sentinels = torch.arange(start = sentinel_id, end = sentinel_id - max_sentinels, step = -1)
        sentineled_toks[first_noise_toks] = sentinels[:first_noise_toks.sum().item()]
        return sentineled_toks[~subse_noise_toks]

    def mask_tokens(self, inputs: torch.Tensor, mlm_probability = 0.15, min_span_length = 1, max_span_length = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        device = inputs.device
        inpts = inputs.clone()
        span_lengths = torch.randint(low = min_span_length, high = max_span_length + 1, size = (inpts.shape[0],), device = device)
        periods = torch.round(span_lengths / mlm_probability)
        offsets = torch.tensor([random.randint(0, period.item()) for period in periods], device = device)
        masks = torch.stack([(torch.arange(start = 0, end = inpts.shape[1]) + offset) % period < span for offset, period, span in zip(offsets, periods, span_lengths)])

        if self.tokenizer._pad_token is not None:
            padding_mask = inpts.eq(self.tokenizer.pad_token_id)
            masks.masked_fill_(padding_mask, value = False)
        num_masks = torch.floor_divide(masks.sum(axis = 1), span_lengths)
        new_inpts = []
        lbls = []
        for inpt, mask in zip(inpts, masks):
            new_inpts.append(
                self._noise_span_to_unique_sentinel(inpt, mask, 100, tokenizer.convert_tokens_to_ids(['<extra_id_0>'])[0])
            )
            lbls.append(
                self._noise_span_to_unique_sentinel(inpt, ~mask, 100, tokenizer.convert_tokens_to_ids(['<extra_id_0>'])[0])
            )

        new_inpts = pad_sequence(new_inpts, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        lbls = pad_sequence(lbls, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        return new_inpts, lbls



if __name__=="__main__":
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("flan-t5-review")
    train_dataset = MethodDataset(tokenizer,'train')
    valid_dataset = MethodDataset(tokenizer,'val')
    
    
    class Hparam():
        def __init__(self):
            self.model_name_or_path = 't5-small'
            self.model_type = 't5'
            self.tokenizer_name = 't5-small'
            self.cache_dir = './cached'
            self.learning_rate = 5e-5
            self.weight_decay = 0.0
            self.adam_epsilon = 1e-8
            self.warmup_steps = 0
            self.num_train_epochs = 1
            self.trn_bs = 8
            self.val_bs = 8
            self.overwrite_cache = False
            self.fp16 = False
            self.fp16_opt_level = 'O1'
            self.n_tpu_cores = 0
            self.n_gpu = 1
            self.max_grad_norm = 1.0
            self.do_train = True
            self.do_predict = True
            self.gradient_accumulation_steps = 1
            self.seed = 42
            self.block_size = 128

    hparams = Hparam()
    
    training_args = TrainingArguments(
        logging_steps=50,
        output_dir="./code_encoder",
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=hparams.trn_bs,
        per_device_eval_batch_size=hparams.val_bs,
        gradient_accumulation_steps = hparams.gradient_accumulation_steps,
        save_steps=1000,
        save_total_limit=2,
        do_train = True,
        do_eval = True
    )

    trainer = Trainer(
        model = model,
        args = training_args,
        data_collator = DataCollatorForSeq2SeqMaskLanguageModeling(tokenizer),
        train_dataset = train_dataset,
        eval_dataset = valid_dataset,
    )
    
    trainer.train()
