from omegaconf.omegaconf import OmegaConf
def extract_config(pretrained_model):
    class Restorer(ILQLGPT):
        def __init__(self, cfg, trainer=None):
            super().__init__(ilql_config, cfg=cfg, trainer=trainer)

    pretrained_cfg = Restorer.restore_from(
        restore_path=pretrained_model,
        trainer=trainer,
        return_config=True,
    )
    OmegaConf.set_struct(pretrained_cfg, True)
