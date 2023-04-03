import os
from typing import Union, cast

import numpy as np
import torch
import transformers
from rich.console import Console
from rich.table import Table

import trlx.utils.logging as logging
from trlx.data.configs import TRLConfig
from trlx.data.ilql_types import ILQLBatch, ILQLSeq2SeqBatch
from trlx.models.modeling_ilql import (
    AutoModelForCausalLMWithILQLHeads,
    AutoModelForSeq2SeqLMWithILQLHeads,
    ILQLConfig,
)
from trlx.pipeline.offline_pipeline import (
    ILQLRolloutStorage,
    ILQLSeq2SeqRolloutStorage,
    tokenize_dialogue,
)
from trlx.trainer import register_trainer
from trlx.trainer.accelerate_base_trainer import AccelerateRLTrainer
from trlx.utils import to_device

logger = logging.get_logger(__name__)


def make_experience(samples, rewards, tokenizer=None, max_length=2048, verbose=True):  # noqa: C901
    """
    Tokenizes samples and shapes rewards into proper tensors and then inserts the resulting dataset into the trainer
    """

    if verbose:
        logger.info("Collecting rollouts")
    if tokenizer is not None:
        samples = [tokenize_dialogue(s, tokenizer, max_length) for s in samples]

    all_input_ids = []
    all_actions_ixs = []
    all_states_ixs = []
    all_dones = []
    for sample in samples:
        length = 0
        all_input_ids.append(torch.tensor(sum((s.tokens for s in sample), ())))
        actions_ixs = []
        for dm in sample:
            if dm.is_output:
                actions_ixs.append(torch.arange(length - 1, length + len(dm.tokens) - 1))

            length += len(dm.tokens)

        states_ixs = torch.hstack((*actions_ixs, torch.tensor(length - 1)))
        all_dones.append(torch.tensor([1] * (len(states_ixs) - 1) + [0], dtype=int))
        all_actions_ixs.append(torch.hstack(actions_ixs))
        all_states_ixs.append(states_ixs)

    if tokenizer is not None and os.environ.get("RANK", "0") == "0" and verbose:
        logger.info("Logging sample example")
        prompt = tokenizer.decode(all_input_ids[0][: all_states_ixs[0][1]])
        response = tokenizer.decode(all_input_ids[0][all_states_ixs[0][1] :])
        columns = ["Prompt", "Response", "Reward"]
        table = Table(*columns, title="Sample Example", show_lines=True)
        table.add_row(prompt, response, str(rewards[0]))
        Console().print(table)

    sample_lengths = np.array(list(map(len, all_input_ids)))
    output_lengths = np.array(list(map(len, all_actions_ixs)))
    prompt_lengths = sample_lengths - output_lengths
    returns = torch.tensor(rewards, dtype=float)

    if os.environ.get("RANK", "0") == "0" and verbose:
        logger.info("Logging experience string statistics")
        columns = ["Prompt Length", "Output Length", "Sample Length"]
        table = Table(*columns, title="Experience String Stats (mean ∈ \[min, max])", show_lines=True)
        row = []
        for lengths in [prompt_lengths, output_lengths, sample_lengths]:
            row.append(f"{lengths.mean():.2f} ∈ [{min(lengths)}, {max(lengths)}]")
        table.add_row(*row)
        Console().print(table)

    returns = returns - returns.mean()
    std_returns = returns.std()
    if not torch.isnan(std_returns):
        returns = returns / (std_returns + torch.finfo(returns.dtype).eps)
    rewards = [torch.zeros(len(x)) for x in all_actions_ixs]
    for rs, ret in zip(rewards, returns):
        rs[-1] = ret

    attention_mask = [torch.ones(len(x), dtype=int) for x in all_input_ids]

    return ILQLRolloutStorage(
        all_input_ids,
        attention_mask,
        rewards,
        all_states_ixs,
        all_actions_ixs,
        all_dones,
    )


@register_trainer
class AccelerateILQLTrainer(AccelerateRLTrainer):
    def __init__(self, config: TRLConfig, **kwargs):
        super().__init__(config, **kwargs)

        if not isinstance(config.method, ILQLConfig):
            raise ValueError("config.method must be ILQLConfig")

        self.ilql: ILQLConfig = cast(ILQLConfig, config.method)

        self.generate_kwargs = dict(
            config.method.gen_kwargs,
            max_length=self.max_length,
            logit_mask=self.logit_mask,
            eos_token_id=self.tokenizer.eos_token_id if self.tokenizer else 0,
            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer else 0,
        )

    def get_arch(self, config):
        if config.model.model_arch_type == "seq2seq":
            from_fn = AutoModelForSeq2SeqLMWithILQLHeads.from_pretrained
            if issubclass(type(config.model.model_path), transformers.PretrainedConfig):
                from_fn = AutoModelForSeq2SeqLMWithILQLHeads.from_config
        else:
            from_fn = AutoModelForCausalLMWithILQLHeads.from_pretrained
            if issubclass(type(config.model.model_path), transformers.PretrainedConfig):
                from_fn = AutoModelForCausalLMWithILQLHeads.from_config
        return from_fn(
            config.model.model_path,
            two_qs=config.method.two_qs,
            alpha=config.method.alpha,
        )

    def post_backward_callback(self):
        if self.iter_count % self.config.method.steps_for_target_q_sync == 0:
            self.accelerator.unwrap_model(self.model).sync_target_q_heads()

    def loss(self, batch: Union[ILQLBatch, ILQLSeq2SeqBatch]):
        batch = to_device(batch, self.accelerator.device)
        if self.config.model.model_arch_type == "seq2seq":
            logits, qs, target_qs, vs, _, _ = self.model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                actions_ixs=batch.actions_ixs,
                states_ixs=batch.states_ixs,
                decoder_input_ids=batch.decoder_input_ids,
            )
        else:
            logits, qs, target_qs, vs, _ = self.model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                actions_ixs=batch.actions_ixs,
                states_ixs=batch.states_ixs,
            )

        return self.ilql.loss((logits, (qs, target_qs, vs)), batch)

    def prepare_learning(self):
        train_dataloader = self.store.create_loader(self.config.train.batch_size)
        eval_dataloader = self.eval_pipeline.create_loader(self.config.train.batch_size)

        (
            self.model,
            self.opt,
            self.train_dataloader,
            self.eval_dataloader,
        ) = self.accelerator.prepare(self.model, self.opt, train_dataloader, eval_dataloader)

        self.n_updates_per_batch = 1
        self.total_steps = self.config.train.epochs * len(self.train_dataloader)
        self.total_steps = min(self.total_steps, self.config.train.total_steps)

    def make_experience_seq2seq(self, samples, rewards, max_length=2048):
        """
        Tokenizes samples and shapes rewards into proper tensors and then inserts the resulting dataset into the trainer
        """
        logger.info("Collecting rollouts")
        if self.tokenizer:
            samples = [tokenize_dialogue(s, self.tokenizer, max_length) for s in samples]

        all_input_ids = []
        all_output_ids = []
        all_actions_ixs = []
        all_states_ixs = []
        all_dones = []
        for sample in samples:
            all_input_ids.append(torch.tensor(sample[0].tokens))
            all_output_ids.append(torch.tensor(sample[1].tokens))
            actions_ixs = []
            length = 0
            for phrase in sample:
                if phrase.is_output:
                    length = len(phrase.tokens)
                    actions_ixs.append(torch.arange(0, length - 1))
            states_ixs = torch.hstack((*actions_ixs, torch.tensor(length - 1)))
            all_dones.append(torch.tensor([1] * (len(states_ixs) - 1) + [0], dtype=int))
            all_actions_ixs.append(torch.hstack(actions_ixs))
            all_states_ixs.append(states_ixs)

        if self.tokenizer and os.environ.get("RANK", "0") == "0":
            logger.info("Logging sample example")
            prompt = self.tokenizer.decode(all_input_ids[0])
            response = self.tokenizer.decode(all_output_ids[0])
            columns = ["Prompt", "Response", "Reward"]
            table = Table(*columns, title="Sample Example", show_lines=True)
            table.add_row(prompt, response, str(rewards[0]))
            Console().print(table)

        sample_lengths = np.array(list(map(len, all_input_ids))) + np.array(list(map(len, all_output_ids)))
        output_lengths = np.array(list(map(len, all_output_ids)))
        prompt_lengths = sample_lengths - output_lengths
        returns = torch.tensor(rewards, dtype=float)

        if os.environ.get("RANK", "0") == "0":
            logger.info("Logging experience string statistics")
            columns = ["Prompt Length", "Output Length", "Sample Length"]
            table = Table(*columns, title="Experience String Stats (mean ∈ \[min, max])", show_lines=True)
            row = []
            for lengths in [prompt_lengths, output_lengths, sample_lengths]:
                row.append(f"{lengths.mean():.2f} ∈ [{min(lengths)}, {max(lengths)}]")
            table.add_row(*row)
            Console().print(table)

        returns = (returns - returns.mean()) / (returns.std() + torch.finfo(returns.dtype).eps)
        rewards = [torch.zeros(len(x)) for x in all_actions_ixs]
        for rs, ret in zip(rewards, returns):
            rs[-1] = ret

        attention_mask = [torch.ones(len(x), dtype=int) for x in all_input_ids]
        self.store = ILQLSeq2SeqRolloutStorage(
            all_input_ids,
            attention_mask,
            all_output_ids,
            rewards,
            all_states_ixs,
            all_actions_ixs,
            all_dones,
        )

    def make_experience(self, samples, rewards, max_length=2048):
        """
        Tokenizes samples and shapes rewards into proper tensors and then inserts the resulting dataset into the trainer
        """

        if self.config.model.model_arch_type == "seq2seq":
            return self.make_experience_seq2seq(samples, rewards, max_length)

        self.store = make_experience(samples, rewards, self.tokenizer, max_length=max_length, verbose=True)
