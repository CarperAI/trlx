from typing import Optional, cast

from trlx.data.configs import TRLConfig
from trlx.data.ilql_types import ILQLBatch
from trlx.trainer import register_trainer
from trlx.trainer.accelerate_base_trainer import AccelerateRLTrainer
from trlx.trainer.nn.ilql_models import CausalLMWithValueHeads, ILQLConfig
from trlx.utils import to_device


@register_trainer
class AccelerateILQLTrainer(AccelerateRLTrainer):
    def __init__(self, config: TRLConfig, **kwargs):
        super().__init__(config, **kwargs)

        if not isinstance(config.method, ILQLConfig):
            raise ValueError("config.method must be ILQLConfig")

        self.ilql: ILQLConfig = cast(ILQLConfig, config.method)

        self.generate_kwargs = dict(
            config.method.gen_kwargs,
            max_length=self.max_length,
            logit_mask=self.logit_mask,
            eos_token_id=self.tokenizer.eos_token_id if self.tokenizer else 0,
            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer else 0,
        )

    def get_arch(self, config):
        return CausalLMWithValueHeads(
            config.model.model_path,
            ilql_config=config.method,
            num_layers_unfrozen=config.model.num_layers_unfrozen,
        )

    def post_backward_callback(self):
        if self.iter_count % self.config.method.steps_for_target_q_sync == 0:
            self.accelerator.unwrap_model(self.model).sync_target_q_heads()

    def loss(self, batch: ILQLBatch):
        batch = to_device(batch, self.accelerator.device)

        logits, qs, target_qs, vs, _ = self.model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            actions_ixs=batch.actions_ixs,
            states_ixs=batch.states_ixs,
        )

        return self.ilql.loss((logits, (qs, target_qs, vs)), batch)

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
        # TODO: Support saving with `transformers.PreTrainedModel.save_pretrained`.
        # This is currently not supported becasue `nn.ilql_models.CausalLMWithValueHeads`
        # requires a custom `generate` method using its (value/q) heads to steer
        # sampling - something that is not possible with the default
        # `transformers.PreTrainedModel.generate`.
        raise NotImplementedError(
            "`AccelerateILQLTrainer` does not currently support automatic saving "
            "with `transformers.PreTrainedModel.save_pretrained`."
        )
