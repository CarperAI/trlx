import os.path
import sys
from glob import glob

from nemo.collections.nlp.modules.common.megatron.megatron_init import (
    fake_initialize_model_parallel,
)
from nemo.utils.app_state import AppState
from nemo.utils.model_utils import inject_model_parallel_rank
from omegaconf.omegaconf import OmegaConf

from trlx.data.configs import TrainConfig
from trlx.data.default_configs import default_ilql_config
from trlx.trainer.nemo_ilql_trainer import ILQLGPT, megatron_trainer

default_config = default_ilql_config()

trl_config = default_config.evolve(
    train=TrainConfig(
        **dict(
            default_config.train.__dict__,
            trainer="NeMoILQLTrainer",
            trainer_kwargs=dict(
                pretrained_model=None,
                megatron_cfg="megatron_20b.yaml",
            ),
        ),
    )
)


def find_checkpoints(checkpoint_dir):
    checkpoints = glob(os.path.join(checkpoint_dir, "*", "*.ckpt"))
    names = [os.path.basename(c) for c in checkpoints]
    return set(names)


def main(megatron_cfg_path, checkpoint_path):
    ilql_config = trl_config.method

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

    checkpoint_names = find_checkpoints(checkpoint_path)
    checkpoint_name = next(iter(checkpoint_names))
    print(f"Loading checkpoint {checkpoint_name}, found {checkpoint_names} checkpoints")

    checkpoint_path = inject_model_parallel_rank(os.path.join(checkpoint_path, checkpoint_name))

    model = ILQLGPT.load_from_checkpoint(
        checkpoint_path,
        cfg=megatron_cfg.model,
        trainer=trainer,
        ilql_config=ilql_config,
    )

    model.sequence_parallel_(False)
    model.activation_checkpointing_(False)

    test = ["I don't know much about Hungarian underground"]
    test = [model.tokenizer.tokenizer.bos_token + t for t in test]

    print(model.generate(test, dict(max_length=40, min_length=0))["sentences"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
