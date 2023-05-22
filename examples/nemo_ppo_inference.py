import os.path
import sys
from glob import glob

from nemo.collections.nlp.modules.common.megatron.megatron_init import (
    fake_initialize_model_parallel,
)
from nemo.utils.app_state import AppState
from omegaconf.omegaconf import OmegaConf

from trlx.data.default_configs import default_ppo_config
from trlx.trainer.nemo_ppo_trainer import PPOGPT, megatron_trainer

default_config = default_ppo_config()

trl_config = default_config.evolve(
    train=dict(
        default_config.train.__dict__,
        trainer="NeMoPPOTrainer",
        trainer_kwargs=dict(
            pretrained_model=None,
            megatron_cfg="megatron_20b.yaml",
        ),
    ),
)


def find_checkpoints(checkpoint_dir):
    checkpoints = glob(os.path.join(checkpoint_dir, "*", "*.ckpt"))
    names = [os.path.basename(c) for c in checkpoints]
    return set(names)


def main(megatron_cfg_path, checkpoint_path):
    ppo_config = trl_config.method

    megatron_cfg = OmegaConf.load(megatron_cfg_path)
    megatron_cfg.trainer.num_nodes = 1
    megatron_cfg.trainer.devices = 4
    megatron_cfg.model.resume_from_checkpoint = checkpoint_path
    megatron_cfg.exp_manager.create_wandb_logger = False
    megatron_cfg.exp_manager.create_checkpoint_callback = False

    trainer = megatron_trainer(megatron_cfg)

    # Manually set up the TP and PP groups
    app_state = AppState()
    app_state.model_parallel_size = (
        megatron_cfg.model.tensor_model_parallel_size * megatron_cfg.model.pipeline_model_parallel_size
    )
    app_state.tensor_model_parallel_size = megatron_cfg.model.tensor_model_parallel_size
    app_state.pipeline_model_parallel_size = megatron_cfg.model.pipeline_model_parallel_size
    (
        app_state.tensor_model_parallel_rank,
        app_state.pipeline_model_parallel_rank,
        app_state.model_parallel_size,
        app_state.data_parallel_size,
        app_state.pipeline_model_parallel_split_rank,
        app_state.virtual_pipeline_model_parallel_rank,
    ) = fake_initialize_model_parallel(
        world_size=app_state.model_parallel_size,
        rank=trainer.global_rank,
        tensor_model_parallel_size_=megatron_cfg.model.tensor_model_parallel_size,
        pipeline_model_parallel_size_=megatron_cfg.model.pipeline_model_parallel_size,
        pipeline_model_parallel_split_rank_=None,
    )

    model = PPOGPT(ppo_config=ppo_config, cfg=megatron_cfg.model, trainer=trainer, build_reference_model=False)
    model.load_from_pretrained(checkpoint_path)

    test = ["I don't know much about Hungarian underground"]
    test = [model.tokenizer.tokenizer.bos_token + t for t in test]

    print(model.generate(test, dict(max_length=40, min_length=0))["sentences"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
