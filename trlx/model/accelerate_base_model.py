import os
from abc import abstractmethod
from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchtyping import TensorType
from transformers import AutoConfig, AutoTokenizer

from trlx.data import BatchElement, RLElement
from trlx.data.accelerate_base_datatypes import (AccelerateRLBatchElement,
                                                 PromptBatch)
from trlx.data.configs import TRLConfig
from trlx.model import BaseRLModel, register_model
from trlx.utils import Clock, rampup_decay, safe_mkdir, topk_mask


@register_model
class AccelerateRLModel(BaseRLModel):
    """
    RL Model that uses accelerate for training
    """
    def __init__(self, config, train_mode=True):
        super().__init__(config, train_mode)

        self.store = rollout_storage  # Need to pass in rollout_storage to be loaded into accelerate object

        self.model = self.get_arch(
            self.config
        )  # Retrieves model equipped for ppo, ilql, etc
        if self.config.model.num_layers_unfrozen > 0:
            for block in self.model.gpt.transformer.h[:-self.config.model.num_layers_unfrozen]:
                for parameter in block.parameters():
                    parameter.requires_grad = False

        if config.model.tokenizer_path:
            self.tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer_path)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
        else:
            self.tokenizer = None

        self.max_length = config.train.gen_size
        config_dict = self.config.to_dict()
        if self.config.train.accelerate_config_path != "":
            with open(self.config.train.accelerate_config_path, mode="r") as file:
                accelerate_config = yaml.safe_load(file)
            config_dict.update(accelerate_config)

        self.accelerator = Accelerator(log_with="wandb")

        if int(os.environ.get("WORLD_SIZE", 1)) > 1:
            torch.distributed.barrier(device_ids=[int(os.environ.get("LOCAL_RANK", 0))])
        else:
            torch.random.manual_seed(1000)

        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                project_name=self.config.train.project_name,
                config=config_dict,
                init_kwargs={
                    "wandb": {
                        "name": f"trlx-{config.model.model_path}",
                        "mode": "disabled"
                        if os.environ.get("debug", False)
                        else "online",
                    }
                },
            )

        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.train.learning_rate_init,
            betas=self.config.train.opt_betas,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt,
            self.config.train.total_steps,
            eta_min=self.config.train.learning_rate_target,
        )

    def tokenize(self, text: Iterable[str]):
        """
        Tokenize a batch of text after adding bos token.
        """
        text = [self.tokenizer.bos_token + txt for txt in text]
        return self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.config.train.input_size,
            return_tensors="pt",
        )

    def act(
        self, data: PromptBatch
    ) -> Tuple[
        TensorType["chunk_size", "input_length"],
        TensorType["chunk_size", "gen_size"],
        Iterable[str],
    ]:
        query_tensors = data.tokens.to(
            self.accelerator.device
        )  # [B, N] #TODO(dahoas): This may need to be changed
        with torch.no_grad():
            # TODO(dahoas): swap this out for custom generate to if this fixes issue
            _ = self.model(
                self.dummy_input.to(self.accelerator.device)
            )  # Dummy pass to make things play nice with accelerate
            # Removed synced gpus
            response = self.model.generate(
                query_tensors,
                pad_token_id=self.tokenizer.eos_token_id,
                **self.config.method.gen_kwargs,
            )
            response_tensors = response[
                :,
                query_tensors.size()[1] : query_tensors.size()[1]
                + self.config.train.gen_size,
            ]
        response_text = self.tokenizer.batch_decode(response_tensors)
        return query_tensors, response_tensors, response_text

    @torch.inference_mode()
    def sample(self, prompts: PromptBatch, gen_kwargs: dict) -> Iterable[str]:
        _, _, response_text = self.act(prompts)
        return response_text

    def get_components(self) -> Dict[str, any]:
        components = (
            {"model": self.model, "opt": self.opt, "scheduler": self.scheduler}
            if self.train_mode
            else {"model": self.model}
        )
        return components

    def save(self, directory=None):
        self.accelerator.save_state(directory or self.config.train.checkpoint_dir)

    @abstractmethod
    def get_arch(config: TRLConfig):
        pass

    @abstractmethod
    def loss(input_tokens, attn, rewards):
        pass

    @abstractmethod
    def post_backward_callback(iter, batch, rewards):
        pass

    @abstractmethod
    def post_epoch_callback(iter, batch, rewards):
        """
        Additional exploration can happen here
        """
        pass

    def learn(self, log_fn = None, save_fn = None, eval_fn = None):
        """
        Learn from data in the rollout storage.
        """
        for epoch in range(self.config.train.epochs):
            for iter, (batch, rewards) in enumerate(self.rollout_loader):

                tokens = batch.tokens.to(self.accelerator.device)
                masks = batch.masks.to(self.accelerator.device)
                rewards = rewards.to(self.accelerator.device)
                loss = self.loss(tokens, masks, rewards)

                self.opt.zero_grad()
                self.accelerator.backward(loss)
                self.opt.step()
                self.scheduler.step()

                self.post_backward_callback(iter, batch, rewards)

                self.accelerator.wait_for_everyone()

            self.post_epoch_callback(epoch)
