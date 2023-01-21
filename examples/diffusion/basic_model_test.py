import yaml

from trlx.model.diffusion.sd_rl_model import AccelerateSDModel
from trlx.data.configs import TRLConfig

if __name__ == "__main__":
    cfg = TRLConfig.load_yaml("configs/diffusion/isd_config.yml")
    model = AccelerateSDModel(cfg)

    