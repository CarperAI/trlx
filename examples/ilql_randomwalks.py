import networkx as nx
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, GPT2Config

from trlx.data.configs import TRLConfig
from trlx.model.accelerate_ilql_model import ILQLModel
from trlx.orchestrator.offline_orchestrator import OfflineOrchestrator


def randexclude(rng: np.random.RandomState, n: int, exclude: int) -> int:
    while True:
        x = rng.randint(n)
        if x != exclude:
            return x


# Toy dataset from Decision Transformer (Chen et. al 2021)
def generate_random_walks(
    n_nodes=21, max_length=10, n_walks=1000, p_edge=0.1, seed=1002
):
    rng = np.random.RandomState(seed)

    while True:
        adj = rng.rand(n_nodes, n_nodes) > (1 - p_edge)
        np.fill_diagonal(adj, 0)
        if np.all(adj.sum(1)):
            break

    # terminal state
    adj[0, :] = 0
    adj[0, 0] = 1

    goal = 0
    sample_walks = []
    for _ in range(n_walks):
        node = randexclude(rng, n_nodes, goal)
        walk = [node]

        for istep in range(max_length - 1):
            node = rng.choice(np.nonzero(adj[node])[0])
            walk.append(node)
            if node == goal:
                break

        sample_walks.append(torch.tensor(walk))

    worstlen = max_length

    bestlen = 0
    g = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    for start in set(range(n_nodes)) - {goal}:
        try:
            shortest_path = nx.shortest_path(g, start, goal)[:max_length]
            bestlen += len(shortest_path)
        except:
            bestlen += max_length

    bestlen /= n_nodes - 1

    def stats_fn(samples):
        actlen = 0

        for s in samples:
            for ix in range(len(s)):
                if s[ix] == 0:
                    break
            actlen += (ix + 1) / (n_nodes - 1)

        return {"percentage": 100 * (worstlen - actlen) / (worstlen - bestlen)}

    logit_mask = torch.tensor(~adj)
    return sample_walks, logit_mask, stats_fn


if __name__ == "__main__":
    config = TRLConfig.load_yaml("configs/ilql_config.yml")
    config.train.gen_size = 10
    config.train.epochs = 100

    train_samples, logit_mask, stats_fn = generate_random_walks(seed=1000)
    eval_prompts = torch.arange(1, logit_mask.shape[0]).view(-1, 1)

    def reward_fn(samples):
        rewards = []

        for s in samples:
            if s[-1] == 0:
                for ix in range(len(s)):
                    if s[ix] == 0:
                        rewards.append(-ix - 1)
                        break
            else:
                rewards.append(-100)

        return rewards

    config.model.model_path = GPT2Config(
        n_layer=4, n_embd=144, vocab_size=logit_mask.shape[0]
    )

    model = ILQLModel(config=config, logit_mask=logit_mask)

    orch = OfflineOrchestrator(
        model=model,
        train_samples=train_samples,
        eval_prompts=eval_prompts,
        reward_fn=reward_fn,
        stats_fn=stats_fn,
    )

    model.learn()
