import os
from abc import abstractmethod
from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchtyping import TensorType
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

import wandb
from trlx.data.accelerate_base_datatypes import PromptBatch
from trlx.data.configs import TRLConfig
from trlx.model import BaseRLModel, register_model
from trlx.model.accelerate_base_model import AccelerateRLModel
from trlx.model.accelerate_ppo_model import AcceleratePPOModel
from trlx.model.nn.ppo_models import GPTHeadWithValueModel
from trlx.pipeline.ppo_pipeline import PPORolloutStorage
from trlx.utils import Clock, rampup_decay, safe_mkdir, topk_mask
from trlx.utils.modeling import clip_by_value, logprobs_from_logits, whiten


class SoftEmbedding(nn.Module):
    def __init__(self, 
                wte: nn.Embedding,
                n_tokens: int = 10, 
                random_range: float = 0.5,
                initialize_from_vocab: bool = True):
        """appends learned embedding as prefix

        From: https://github.com/kipgparker/soft-prompt-tuning

        Args:
            wte (nn.Embedding): original transformer word embedding
            n_tokens (int, optional): number of tokens for task. Defaults to 10.
            random_range (float, optional): range to init embedding (if not initialize from vocab). Defaults to 0.5.
            initialize_from_vocab (bool, optional): initalizes from default vocab. Defaults to True.
        """
        super(SoftEmbedding, self).__init__()
        self.wte = wte
        self.n_tokens = n_tokens
        self.learned_embedding = nn.parameter.Parameter(self.initialize_embedding(wte,
                                                                               n_tokens, 
                                                                               random_range, 
                                                                               initialize_from_vocab))
            
    def initialize_embedding(self, 
                             wte: nn.Embedding,
                             n_tokens: int = 10, 
                             random_range: float = 0.5, 
                             initialize_from_vocab: bool = True):
        """initializes learned embedding
        Args:
            same as __init__
        Returns:
            torch.float: initialized using original schemes
        """
        if initialize_from_vocab:
            return self.wte.weight[:n_tokens].clone().detach()
        return torch.FloatTensor(n_tokens, wte.weight.size(1)).uniform_(-random_range, random_range)
            
    def forward(self, tokens):
        """run forward pass
        Args:
            tokens (torch.long): input tokens before encoding
        Returns:
            torch.float: encoding of text concatenated with learned task specifc embedding
        """
        input_embedding = self.wte(tokens[:, self.n_tokens:])
        learned_embedding = self.learned_embedding.repeat(input_embedding.size(0), 1, 1)
        return torch.cat([learned_embedding, input_embedding], 1)


@register_model
class AcceleratePPOSoftpromptModel(AcceleratePPOModel):
    def __init__(self, config, train_mode=True):
        self.store = PPORolloutStorage()
        super().__init__(config, self.store)

        assert config.model.n_soft_tokens > 0, "Number of soft prompt tokens should be >=1"
        
        self.soft_dummy_token_id = 50256 # dummy token for padding soft prompts
        self.n_soft_tokens = config.model.n_soft_tokens

        s_wte = SoftEmbedding(self.model.gpt.get_input_embeddings(), 
                      n_tokens=config.model.n_soft_tokens, 
                      initialize_from_vocab=config.model.initialize_from_vocab)

        self.model.gpt.set_input_embeddings(s_wte)

        # account for extra prefix tokens
        self.config.train.gen_size += self.n_soft_tokens
        self.config.method.gen_kwargs["max_length"] += self.n_soft_tokens
        self.config.method.gen_kwargs["min_length"] += self.n_soft_tokens

    def act(
        self, data: PromptBatch
    ) -> Tuple[
        TensorType["chunk_size", "input_length"],
        TensorType["chunk_size", "gen_size"],
        Iterable[str],
    ]:
        data.tokens = torch.cat([torch.full((data.tokens.shape[0],self.n_soft_tokens), self.soft_dummy_token_id), data.tokens], 1)
        query_tensors = data.tokens.to(
            self.accelerator.device
        )  # [B, N] #TODO(dahoas): This may need to be changed
        with torch.no_grad():
            # TODO(dahoas): swap this out for custom generate to if this fixes issue
            
            # need to pad attention_mask and input_ids to be full seq_len + n_learned_tokens
            # even though it does not matter what you pad input_ids with, it's just to make HF happy
            batch_dim = self.dummy_input.shape[0]
            self.dummy_input = torch.cat([torch.full((batch_dim,self.n_soft_tokens), self.soft_dummy_token_id), self.dummy_input], 1)
            _ = self.model(
                self.dummy_input.to(self.accelerator.device)
            )  # Dummy pass to make things play nice with accelerate
            # Removed synced gpus
            attn_mask = torch.full((query_tensors.shape[0],query_tensors.shape[1]), 1).to(
                self.accelerator.device
            )
            response = self.model.generate(
                query_tensors,
                pad_token_id=self.tokenizer.eos_token_id,
                attention_mask=attn_mask,
                use_cache=False, # needed for softprompt compatibility
                **self.config.method.gen_kwargs
            )
            response_tensors = response[
                :,
                query_tensors.size()[1] : query_tensors.size()[1]
                + self.config.train.gen_size,
            ]
            query_tensors = query_tensors[:, self.n_soft_tokens:] # remove softprompt padding tokens
        response_text = self.tokenizer.batch_decode(response_tensors)
        return query_tensors, response_tensors, response_text