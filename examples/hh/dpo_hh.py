import json
import sys
from collections import defaultdict

import tqdm
from datasets import Dataset, load_dataset

import trlx
from trlx.data.default_configs import (
    DPOConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)

default_config = TRLConfig(
    train=TrainConfig(
        seq_length=1024,
        epochs=100,
        total_steps=1000,
        batch_size=1,
        checkpoint_interval=10000,
        eval_interval=100,
        pipeline="PromptPipeline",
        trainer="AccelerateDPOTrainer",
        checkpoint_dir="checkpoints/dpo_hh",
    ),
    model=ModelConfig(model_path="gpt2", num_layers_unfrozen=-1),
    tokenizer=TokenizerConfig(tokenizer_path="gpt2", truncation_side="right"),
    optimizer=OptimizerConfig(name="adamw", kwargs=dict(lr=1e-6, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)),
    scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=1e12, eta_min=1.0e-4)),  # train.total_steps
    method=DPOConfig(
        name="DPOConfig", gen_kwargs=dict(max_new_tokens=40, top_k=20, top_p=1.0, do_sample=True), beta=0.1
    ),
)


def get_hh(split: str, sanity_check=False, silent=False):
    dataset = load_dataset("Anthropic/hh-rlhf", split=split)
    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    def extract_anthropic_prompt(prompt_and_response):
        """Extract the anthropic prompt from a prompt and response pair."""
        search_term = "\n\nAssistant:"
        search_term_idx = prompt_and_response.rfind(search_term)
        assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
        return prompt_and_response[: search_term_idx + len(search_term)]

    def split_prompt_and_responses(ex):
        prompt = extract_anthropic_prompt(ex["chosen"])
        chosen_response = ex["chosen"][len(prompt) :]
        rejected_response = ex["rejected"][len(prompt) :]
        return prompt, chosen_response, rejected_response

    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(dataset, desc="Processing HH", disable=silent):
        prompt, chosen, rejected = split_prompt_and_responses(row)
        responses = [chosen, rejected]
        n_responses = len(data[prompt]["responses"])
        data[prompt]["pairs"].append((n_responses, n_responses + 1))
        data[prompt]["responses"].extend(responses)
        data[prompt]["sft_target"] = chosen

    def gen():
        for prompt, values in data.items():
            yield {
                "prompt": prompt,
                "responses": values["responses"],
                "pairs": values["pairs"],
            }

    return Dataset.from_generator(gen)


def preprocess(sample):
    pass


def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)

    dataset = load_dataset("Dahoas/full-hh-rlhf").map(preprocess)

    trlx.train(
        config=config,
        samples=dataset["train"],
        eval_prompts=dataset["test"]["prompt"][:280],
        # metric_fn=lambda **kwargs: {"reward": reward_fn(**kwargs)},
        stop_sequences=["Human:", "human:", "Assistant:", "assistant:"],
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)
