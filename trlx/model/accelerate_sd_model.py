import numpy as np
import torch
import diffusers

from trlx.data.configs import TRLConfig
from trlx.data.diffusion_types import DiffPPORLElement, DiffPPORLBatch
from trlx.model import register_model
from trlx.model.accelerate_base_model import AccelerateRLModel
from trlx.pipeline.diffusion_ppo_pipeline import DiffusionPPORolloutStorage

@register_model
class AccelerateSDModel(BaseRLModel):
    def __init__(self, config : TRLConfig):
        super().__init__(config)

        self.model = diffusers.ModelMixin.from_pretrained(config.model.model_path)
        self.store = DiffusionPPORolloutStorage()

        rollout_loader = self.store.create_loader(
            self.config.train.batch_size, shuffle=True
        )

        self.model, self.opt, self.scheduler, rollout_loader = self.accelerator.prepare(
            self.model, self.opt, self.scheduler, rollout_loader
        )

        self.store.clear_history()

    def forward(self, )
    
    def act(self, data : DiffPPORLElement) ->  DiffPPORLElement:
        