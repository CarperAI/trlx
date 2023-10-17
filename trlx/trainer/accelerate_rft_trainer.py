import itertools
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import wandb
from tqdm import tqdm
from transformers import AutoModelForCausalLM, PretrainedConfig

from trlx.data.configs import TRLConfig
from trlx.data.method_configs import MethodConfig, register_method
from trlx.pipeline.offline_pipeline import PromptPipeline
from trlx.trainer import register_trainer
from trlx.trainer.accelerate_base_trainer import AccelerateRLTrainer


@dataclass
@register_method
class RFTConfig(MethodConfig):
    """
    Config for RFT training

    :param gen_kwargs: kwargs for generation
    :type gen_kwargs: Dict[str, Any]

    :param start_percentile: percentile for the starting score threshold for each prompt used for the first improvement step
    :type start_percentile: float

    :param end_percentile: percentile for the final score threshold for each prompt
    :type end_percentile: float

    :param n_improve_steps: the number of improvement steps for each growth step with linearly increasing score threshold
    :type n_improve_steps: int

    :param n_generations_per_prompt: number of generations to sample per each prompt per each growth step
    :type n_generations_per_prompt: int
    """

    gen_kwargs: dict
    start_percentile: float = 0.7
    end_percentile: float = 0.95
    n_improve_steps: int = 4
    n_generations_per_prompt: int = 32


@register_trainer
class AccelerateRFTTrainer(AccelerateRLTrainer):
    def __init__(self, config: TRLConfig, **kwargs):
        super().__init__(config, **kwargs)

        self.generate_kwargs = dict(
            config.method.gen_kwargs,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        self.generate_experience_kwargs = None

    def get_arch(self, config):
        from_fn = AutoModelForCausalLM.from_pretrained
        if issubclass(type(config.model.model_path), PretrainedConfig):
            from_fn = AutoModelForCausalLM.from_config

        model = from_fn(config.model.model_path, **config.model.model_extra_configs)

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
        labels = batch.input_ids.clone()
        loss = self.model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, labels=labels).loss
        stats = {"loss": loss.item()}

        return loss, stats

    def create_train_dataloader(self):
        return self.accelerator.prepare(self.store.create_loader(self.config.train.batch_size))

    def prepare_learning(self):
        self.epoch_count = 0
        self.iter_count = 0
        self.n_inner_epochs = 1
        # because of variable number of samples per each improvement steps
        # there is no way to get the estimate, so here it's just copied from the config
        self.total_steps = self.config.train.total_steps

        self.generations_per_prompt = defaultdict(list)

        eval_dataloader = self.eval_pipeline.create_loader(self.config.train.batch_size)
        self.model, self.opt, self.eval_dataloader = self.accelerator.prepare(self.model, self.opt, eval_dataloader)

        self.make_experience()

    def add_prompt_pipeline(self, pipeline: PromptPipeline):
        """Add a prompt pipeline dataloader to a trainer instance for the `make_experience` stage"""
        prompt_dataloader = pipeline.create_loader(self.config.train.batch_size)
        self.prompt_dataloader = self.accelerator.prepare_data_loader(prompt_dataloader)

    def post_epoch_callback(self):
        self.make_experience()
        self.epoch_count += 1

    def make_experience(self):  # noqa:
        if self.epoch_count % self.config.method.n_improve_steps == 0:
            # generate n samples for each prompt in the prompt_dataloader
            generations = []
            for batch in tqdm(self.prompt_dataloader, desc="Generating", disable=not self.accelerator.is_main_process):
                for _ in range(self.config.method.n_generations_per_prompt):
                    samples = self.generate(**batch)
                    str_samples, str_prompts, str_outputs = self.decode(batch.input_ids, samples, append_eos_token=True)
                    generations.extend({"prompt": p, "output": o} for p, o in zip(str_prompts, str_outputs))

            if torch.distributed.is_initialized():
                all_generations = [None for _ in range(torch.distributed.get_world_size())]
                torch.distributed.all_gather_object(all_generations, generations)
                generations = list(itertools.chain(*all_generations))

            # score the generations
            if self.accelerator.is_main_process:
                all_scores = self.reward_fn(
                    samples=[x["prompt"] + x["output"] for x in generations],
                    prompts=[x["prompt"] for x in generations],
                    outputs=[x["output"] for x in generations],
                )

                all_scores = torch.tensor(all_scores, device=self.accelerator.device)
            else:
                all_scores = torch.zeros(len(generations), device=self.accelerator.device)
            if torch.distributed.is_initialized():
                torch.distributed.broadcast(all_scores, src=0)
                scores = all_scores
            else:
                scores = all_scores

            for g, s in zip(generations, scores):
                self.generations_per_prompt[g["prompt"]].append({"output": g["output"], "score": s.item()})

        scores = [[x["score"] for x in self.generations_per_prompt[p]] for p in self.generations_per_prompt]

        percentile_delta = (
            self.config.method.end_percentile - self.config.method.start_percentile
        ) / self.config.method.n_improve_steps
        percentile = self.config.method.start_percentile + percentile_delta * (
            self.epoch_count % self.config.method.n_improve_steps
        )
        thresholds = np.array([np.quantile(np.array(scores), percentile) for scores in scores])
        # corner case for quantized rewards: don't include the min values, but don't exclude the max values
        thresholds = np.clip(thresholds, thresholds.min() + 1e-3, thresholds.max() - 1e-3)

        # filter out the generations with a score below the percentile per prompt
        samples_selected = []
        for prompt, threshold in zip(self.generations_per_prompt, thresholds):
            for x in self.generations_per_prompt[prompt]:
                if x["score"] >= threshold:
                    samples_selected.append([prompt, x["output"]])

        # deduplicate the samples
        samples_selected = list({tuple(x) for x in samples_selected})

        self.accelerator.log(
            {
                "scores_per_single_prompt": wandb.Histogram(scores[0]),
                "thresholds": wandb.Histogram(thresholds),
                "scores_mean": np.mean(np.hstack(scores)),
                "scores_dist": wandb.Histogram(np.hstack(scores)),
                "len_samples_selected": len(samples_selected),
                "samples_per_single_prompt": wandb.Table(
                    data=list(
                        zip(
                            [x[0] for x in samples_selected[:128]],
                            [x[1] for x in samples_selected[:128]],
                        )
                    ),
                    columns=["prompt", "output"],
                ),
            },
            step=self.iter_count,
        )

        if len(samples_selected):
            self.store = PromptPipeline(
                samples_selected, max_prompt_length=2048, tokenizer=self.tokenizer, add_special_tokens=True
            )
