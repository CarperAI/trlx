import torch

from trlx.model import BaseRLModel
from trlx.orchestrator import Orchestrator, register_orchestrator
from trlx.pipeline.offline_pipeline import (OfflinePipeline,
                                            OfflineRolloutStorage)


@register_orchestrator
class OfflineOrchestrator(Orchestrator):
    def __init__(
        self, model: BaseRLModel, train_samples, ratings, reward_fn=None, metric_fn=None
    ):
        self.model = model

        train_samples = model.tokenize(train_samples)

        returns = torch.as_tensor(ratings, dtype=torch.float)
        returns = (returns - returns.mean()) / (returns.std() + 1e-30)

        rewards = [torch.zeros(x.shape[0] - 1) for x in train_samples]
        for rs, G in zip(rewards, returns):
            rs[-1] = G

        attention_mask = [torch.ones(x.shape[0], dtype=int) for x in train_samples]
        for mask in attention_mask:
            mask[-1] = 0

        assert (
            train_samples[0].shape[0]
            == attention_mask[0].shape[0]
            == rewards[0].shape[0] + 1
        )

        self.model.train_store = OfflineRolloutStorage(
            train_samples, attention_mask, rewards
        )

        self.model.reward_fn = reward_fn
        self.model.metric_fn = metric_fn

    def score(self, samples):
        return self.model.reward_fn(samples)
