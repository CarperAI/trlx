import unittest
from trlx.data.configs import TRLConfig
from trlx.model.accelerate_ppo_model import AcceleratePPOModel
from trlx.orchestrator.ppo_orchestrator import PPOOrchestrator
from trlx.pipeline.ppo_pipeline import PPOPipeline
from trlx.utils.loading import get_model, get_orchestrator, get_pipeline
from trlx.model.nn.ppo_models import GPTHeadWithValueModel, GPTHydraHeadWithValueModel
from transformers import AutoConfig, AutoTokenizer
import torch

config = TRLConfig.load_yaml("configs/test_config.yml")
hydra_model = GPTHydraHeadWithValueModel(
            config.model.model_path, config.model.num_layers_unfrozen
        )

tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

dummy_inputs = tokenizer(
            "Once upon a time there was a happy goose named Louis. He liked to eat bananas.",
            truncation=True,
            padding="max_length",
            max_length=config.train.input_size,
            return_tensors="pt",
        )

####
print(dummy_inputs)
response = hydra_model.ref_model.generate(**dummy_inputs, **config.method.gen_kwargs)
decoded_reponse = tokenizer.batch_decode(response)
print(decoded_reponse)