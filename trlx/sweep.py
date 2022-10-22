import wandb


class Sweep:
    def __init__(self, config, wandb_tracker):
        self.config = config
        self.wandb_tracker = wandb_tracker
