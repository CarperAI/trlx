import os.path
import sys
from glob import glob

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
    megatron_cfg.trainer.devices = (
        megatron_cfg.model.tensor_model_parallel_size * megatron_cfg.model.pipeline_model_parallel_size
    )
    # Overriden in generate
    megatron_cfg.model.global_batch_size = megatron_cfg.model.micro_batch_size
    megatron_cfg.model.resume_from_checkpoint = checkpoint_path
    megatron_cfg.exp_manager.create_wandb_logger = False
    megatron_cfg.exp_manager.create_checkpoint_callback = False

    trainer = megatron_trainer(megatron_cfg)

    if trainer.world_size != megatron_cfg.trainer.devices:
        raise ValueError("Inference only supports data parallel world size of 1")

    # Initialize PyTorch Lightning DDP

    def dummy():
        return

    if trainer.strategy.launcher is not None:
        trainer.strategy.launcher.launch(dummy, trainer=trainer)
    trainer.strategy.setup_environment()

    model = PPOGPT(ppo_config=ppo_config, cfg=megatron_cfg.model, trainer=trainer, build_reference_model=False)
    model.load_from_pretrained(checkpoint_path)

    test = ["I don't know much about Hungarian underground"]
    test = [model.tokenizer.tokenizer.bos_token + t for t in test]

    print(model.generate(test, dict(max_length=40, min_length=0))["sentences"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
