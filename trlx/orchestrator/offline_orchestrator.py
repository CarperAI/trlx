import torch

from trlx.model import BaseRLModel
from trlx.orchestrator import Orchestrator, register_orchestrator
from trlx.pipeline.offline_pipeline import (
    OfflinePipeline,
    OfflineRolloutStorage
)


@register_orchestrator
class OfflineOrchestrator(Orchestrator):
    def __init__(
        self, model: BaseRLModel, train_samples, eval_prompts, reward_fn, stats_fn=None
    ):
        self.model = model

        if isinstance(train_samples[0], str):
            train_samples = model.tokenize(train_samples)["input_ids"]

        attention_mask = [torch.ones(len(x), dtype=int) for x in train_samples]
        for mask in attention_mask:
            mask[-1] = 0

        returns = torch.as_tensor(reward_fn(train_samples)).float()
        returns = (returns - returns.mean()) / (returns.std() + 1e-30)

        rewards = [torch.zeros(len(x) - 1) for x in train_samples]
        for rs, G in zip(rewards, returns):
            rs[-1] = G

        assert len(train_samples[0]) == len(attention_mask[0]) == len(rewards[0]) + 1

        self.model.train_store = OfflineRolloutStorage(
            train_samples, attention_mask, rewards
        )
        self.model.eval_pipeline = OfflinePipeline(eval_prompts)
        self.model.reward_fn = reward_fn
        self.model.stats_fn = stats_fn

    def score(self, samples):
        return self.model.reward_fn(samples)
