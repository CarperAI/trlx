import numpy as np
import torch
import diffusers
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

from trlx.data.configs import TRLConfig
from trlx.data.diffusion_types import DiffPPORLElement, DiffPPORLBatch
from trlx.model import register_model
from trlx.model import BaseRLModel
from trlx.pipeline.diffusion_ppo_pipeline import DiffusionPPORolloutStorage
from trlx.model.nn.sd_models import UNetActorCritic

@register_model
class SDModel(BaseRLModel):
    def __init__(self, config : TRLConfig, train_mode = True):
        super().__init__(config, train_mode)

        model_id = config.model.model_path

        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16)
        pipe = pipe.to("cuda")
        cfg = pipe.unet.config

        self.model = UNetActorCritic(pipe.vae, pipe.unet)

        # Pipe has:
        # - tokenizer (transformers.CLIPTokenizer)
        # - text_encoder (transformers.CLIPTextModel)
        # - unet (diffusers.UNet2DConditionModel)
        #   - forward([b, c, h, w] : sample, [t] : timestep, [b, n, d] : encoder hidden, return_dict = True)
        #   - return:  Unet2DConditionOutput([b, c, h, w] : sample)
        # - vae (diffusers.AutoEncoderKL)
        #   - forward([b, c, h, w] : sample, return_dict = True)
        #   - return:  DecoderOoutput([b, c, h, w] : sample)
        #   - encode([b, c, h, w])
        #   - return distribution
        self.store = DiffusionPPORolloutStorage()

        rollout_loader = self.store.create_loader(
            self.config.train.batch_size, shuffle=True
        )

        self.model, self.opt, self.scheduler, rollout_loader = self.accelerator.prepare(
            self.model, self.opt, self.scheduler, rollout_loader
        )

        self.store.clear_history()

    def forward(self, data):
        pass
    
    def act(self, data : DiffPPORLElement) ->  DiffPPORLElement:
        pass