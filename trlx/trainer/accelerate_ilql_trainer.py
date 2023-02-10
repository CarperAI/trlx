import os
from typing import Optional, Sequence, Union, cast

import numpy as np
import torch
from rich.console import Console
from rich.table import Table

import trlx.utils.logging as logging
from trlx.data.configs import TRLConfig
from trlx.data.ilql_types import ILQLBatch
from trlx.pipeline.offline_pipeline import ILQLRolloutStorage, tokenize_dialogue
from trlx.trainer import register_trainer
from trlx.trainer.accelerate_base_trainer import AccelerateRLTrainer
from trlx.trainer.nn.ilql_models import CausalLMWithValueHeads, ILQLConfig
from trlx.utils import to_device

logger = logging.get_logger(__name__)


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
        return CausalLMWithValueHeads(
            config.model.model_path,
            ilql_config=config.method,
            num_layers_unfrozen=config.model.num_layers_unfrozen,
        )

    def tokenize(self, texts: Union[Sequence[str], Sequence[torch.LongTensor]]):
        if isinstance(texts[0], torch.LongTensor):
            return texts

        tokenized = self.tokenizer(
            [self.tokenizer.bos_token + x + self.tokenizer.eos_token for x in texts],
            max_length=self.max_length,
            truncation=True,
            # NOTE: We manually add special tokens (bos) above so we set this False
            # to avoid models that automatically add special tokens (e.g. OPT)
            # adding them twice more.
            add_special_tokens=False,
        )
        input_ids = list(map(torch.as_tensor, tokenized.input_ids))
        return input_ids

    def post_backward_callback(self):
        if self.iter_count % self.config.method.steps_for_target_q_sync == 0:
            self.accelerator.unwrap_model(self.model).sync_target_q_heads()

    def loss(self, batch: ILQLBatch):
        batch = to_device(batch, self.accelerator.device)

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
        self.total_steps = self.config.train.epochs * len(train_dataloader)
        self.total_steps = min(self.total_steps, self.config.train.total_steps)

    def save_pretrained(self, directory: Optional[str] = None):
        """NOTE: If a `directory` is not provided, the model will be saved to a sub-directory
        of the Trainer config checkpoint dir named "hf_model" (e.g. `/ckpts/hf_model`).
        """
        # TODO: Support saving with `transformers.PreTrainedModel.save_pretrained`.
        # This is currently not supported becasue `nn.ilql_models.CausalLMWithValueHeads`
        # requires a custom `generate` method using its (value/q) heads to steer
        # sampling - something that is not possible with the default
        # `transformers.PreTrainedModel.generate`.
        raise NotImplementedError(
            "`AccelerateILQLTrainer` does not currently support automatic saving "
            "with `transformers.PreTrainedModel.save_pretrained`."
        )

    def make_experience(self, samples, rewards, max_length=2048):
        """
        Tokenizes samples and shapes rewards into proper tensors and then inserts the resulting dataset into the trainer
        """
        logger.info("Collecting rollouts")

        if self.tokenizer:
            samples = [tokenize_dialogue(s, self.tokenizer, max_length) for s in samples]

        all_input_ids = []
        all_actions_ixs = []
        all_states_ixs = []
        all_dones = []
        for sample in samples:
            length = 0
            all_input_ids.append(torch.tensor(sum(sample, [])))
            isoutput = False
            actions_ixs = []
            for phrase in sample:
                if isoutput:
                    actions_ixs.append(torch.arange(length - 1, length + len(phrase) - 1))

                length += len(phrase)
                isoutput = not isoutput

            states_ixs = torch.hstack((*actions_ixs, torch.tensor(length - 1)))
            all_dones.append(torch.tensor([1] * (len(states_ixs) - 1) + [0], dtype=int))
            all_actions_ixs.append(torch.hstack(actions_ixs))
            all_states_ixs.append(states_ixs)

        if self.tokenizer and os.environ.get("RANK", "0") == "0":
            logger.info("Logging sample example")
            prompt = self.tokenizer.decode(all_input_ids[0][: all_states_ixs[0][1]])
            response = self.tokenizer.decode(all_input_ids[0][all_states_ixs[0][1] :])
            columns = ["Prompt", "Response", "Reward"]
            table = Table(*columns, title="Sample Example", show_lines=True)
            table.add_row(prompt, response, str(rewards[0]))
            Console().print(table)

        sample_lengths = np.array(list(map(len, all_input_ids)))
        output_lengths = np.array(list(map(len, all_actions_ixs)))
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

        returns = (returns - returns.mean()) / (returns.std() + 1e-30)
        rewards = [torch.zeros(len(x)) for x in all_actions_ixs]
        for rs, ret in zip(rewards, returns):
            rs[-1] = ret

        attention_mask = [torch.ones(len(x), dtype=int) for x in all_input_ids]

        self.store = ILQLRolloutStorage(
            all_input_ids,
            attention_mask,
            rewards,
            all_states_ixs,
            all_actions_ixs,
            all_dones,
        )
