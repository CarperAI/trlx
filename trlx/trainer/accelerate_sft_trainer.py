from dataclasses import dataclass

from transformers import AutoModelForCausalLM, PretrainedConfig

from trlx.data.configs import TRLConfig
from trlx.data.method_configs import MethodConfig, register_method
from trlx.pipeline.offline_pipeline import (
    DialogStore,
    PromptPipeline,
    tokenize_dialogue,
)
from trlx.trainer import register_trainer
from trlx.trainer.accelerate_base_trainer import AccelerateRLTrainer


@dataclass
@register_method
class SFTConfig(MethodConfig):
    """
    Config for SFT training

    :param gen_kwargs: kwargs for generation
    :type gen_kwargs: Dict[str, Any]
    """

    gen_kwargs: dict


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
        from_fn = AutoModelForCausalLM.from_pretrained
        if issubclass(type(config.model.model_path), PretrainedConfig):
            from_fn = AutoModelForCausalLM.from_config

        model = from_fn(config.model.model_path)

        if config.model.peft_config is not None:
            # Initialize the peft adapter
            import peft

            peft_config = config.model.peft_config
            if not isinstance(peft_config, peft.PeftConfig):
                if isinstance(peft_config, dict):
                    peft_config = peft.get_peft_config(peft_config)
                else:
                    raise ValueError("`peft_config` should be an instance of `peft.PeftConfig` or a dict.")
            model = peft.get_peft_model(model, peft_config)
            if self.accelerator.is_main_process:
                model.print_trainable_parameters()

        return model

    def loss(self, batch):
        if "labels" in batch:
            labels = batch.labels.clone()
        else:
            labels = batch.input_ids.clone()
        labels[~batch.attention_mask.bool()] = -100

        loss = self.model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, labels=labels).loss
        stats = {"loss": loss.item()}

        return loss, stats

    def create_train_dataloader(self):
        return self.accelerator.prepare(self.store.create_loader(self.config.train.batch_size))

    def prepare_learning(self):
        self.train_dataloader = self.create_train_dataloader()
        eval_dataloader = self.eval_pipeline.create_loader(self.config.train.batch_size)

        (
            self.model,
            self.opt,
            self.eval_dataloader,
        ) = self.accelerator.prepare(self.model, self.opt, eval_dataloader)

        self.n_inner_epochs = 1
        self.total_steps = self.config.train.epochs * len(self.train_dataloader)
        self.total_steps = min(self.total_steps, self.config.train.total_steps)

    def make_experience(self, samples, seq_length):
        if isinstance(samples[0], str):
            self.store = PromptPipeline(samples, seq_length, self.tokenizer)
        else:
            dialogs = [tokenize_dialogue(d, self.tokenizer, seq_length) for d in samples]
            self.store = DialogStore(dialogs, self.tokenizer)
