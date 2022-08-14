from typing import Dict

from framework.data import RLElement
from framework.model import BaseRLModel, register_model
from framework.pipeline.sentiment import SentimentRolloutStorage
from framework.data.sentiment import SentimentRLElement
from framework.model.nn import QVModel
from framework.utils import rampup_decay, safe_mkdir

from transformers import AutoTokenizer
import os

import torch
from torch.utils.data import DataLoader

@register_model
class SentimentILQLModel(BaseRLModel):
    def __init__(self, config, train_mode = False):
        super().__init__(config, train_mode)

        self.store = SentimentRolloutStorage()

        self.model = QVModel("gpt2")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

        if self.train_mode:
            self.opt = torch.optim.AdamW(self.model.parameters(), lr = self.config.train.learning_rate_init)
            self.scheduler = rampup_decay(
                self.config.train.lr_ramp_steps,
                self.config.train.lr_decay_steps,
                self.config.train.learning_rate_target / self.config.train.learning_rate_init,
                self.opt
            )
    
    def act(self, data : SentimentRLElement) -> SentimentRLElement:
        tok_out = self.tokenizer(data.text)
        model_out = self.model(input_ids = tok_out["input_ids"], attention_mask = tok_out["attention_mask"])
        action = model_out["q_out"].argmax(-1)
        data.action = action
        return data


    def get_components(self) -> Dict[str, any]:
        components = {
            "model" : self.model,
            "opt" : self.opt,
            "scheduler" : self.scheduler} if self.train_mode else {"model" : self.model}
        return components
    
    def learn(self):
        loader = DataLoader(self.store, batch_size = self.config.train.batch_size, shuffle = True)

        return 1

