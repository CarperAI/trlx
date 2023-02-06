from typing import Optional

from transformers import AutoModelForCausalLM

from trlx.data.configs import TRLConfig
from trlx.trainer import register_trainer
from trlx.trainer.accelerate_base_trainer import AccelerateRLTrainer


@register_trainer
class AccelerateSFTTrainer(AccelerateRLTrainer):
    def __init__(self, config: TRLConfig, **kwargs):
        super().__init__(config, **kwargs)

        self.generate_kwargs = dict(
            config.method.gen_kwargs,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

    def get_arch(self, config):
        return AutoModelForCausalLM.from_pretrained(config.model.model_path)

    def loss(self, batch):
        loss = self.model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, labels=batch.input_ids).loss
        stats = {"loss": loss}

        return loss, stats

    def prepare_learning(self):
        train_dataloader = self.store.create_loader(self.config.train.batch_size)
        eval_dataloader = self.eval_pipeline.create_loader(self.config.train.batch_size)

        (
            self.model,
            self.opt,
            self.train_dataloader,
            self.eval_dataloader,
        ) = self.accelerator.prepare(self.model, self.opt, train_dataloader, eval_dataloader)

        self.n_updates_per_batch = 1
        self.total_steps = self.config.train.epochs * len(train_dataloader)
        self.total_steps = min(self.total_steps, self.config.train.total_steps)

    def save_pretrained(self, directory: Optional[str] = None):
        """NOTE: If a `directory` is not provided, the model will be saved to a sub-directory
        of the Trainer config checkpoint dir named "hf_model" (e.g. `/ckpts/hf_model`).
        """
        if directory is None:
            directory = f"{self.config.train.checkpoint_dir}/hf_model"
        self.accelerator.unwrap_model(self.model).base_model.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)
