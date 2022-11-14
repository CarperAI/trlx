# Toy dataset from Decision Transformer (Chen et al. 2021)
# finds graph shortest paths by learning from a dataset of sampled random walks

import networkx as nx
import numpy as np
import torch
from transformers import GPT2Config

import trlx
from trlx.data.configs import TRLConfig


def randexclude(rng: np.random.RandomState, n: int, exclude: int) -> int:
    while True:
        x = rng.randint(n)
        if x != exclude:
            return x


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
    best_lengths = []

    g = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    for start in set(range(n_nodes)) - {goal}:
        try:
            shortest_path = nx.shortest_path(g, start, goal)[:max_length]
            best_lengths.append(len(shortest_path))
        except Exception:
            best_lengths.append(max_length)

    best_lengths = torch.tensor(best_lengths)

    def metric_fn(samples):
        infty = -100
        lengths = []

        for s in samples:
            if s[-1] == 0:
                for ix in range(len(s)):
                    if s[ix] == 0:
                        lengths.append(-ix - 1)
                        break
            else:
                lengths.append(infty)

        lengths = torch.tensor(lengths, dtype=torch.float)
        bound_lengths = torch.where(lengths.eq(infty), worstlen, lengths).abs()
        return {
            "lengths": lengths,
            "optimality": (worstlen - bound_lengths)
            / (worstlen - best_lengths if lengths.shape == best_lengths.shape else 0),
        }

    logit_mask = torch.tensor(~adj)
    return sample_walks, logit_mask, metric_fn


if __name__ == "__main__":
    walks, logit_mask, metric_fn = generate_random_walks(seed=1000)
    eval_prompts = torch.arange(1, logit_mask.shape[0]).view(-1, 1)
    lengths = metric_fn(walks)["lengths"]

    config = TRLConfig.load_yaml("configs/ilql_config.yml")
    config.train.gen_size = 10
    config.train.epochs = 100
    config.train.lr_init = 1e-3
    config.method.alpha = 0.1

    config.model.tokenizer_path = ""
    config.model.model_path = GPT2Config(
        n_layer=2, n_embd=144, vocab_size=logit_mask.shape[0]
    )

    trlx.train(
        dataset=(walks, lengths),
        eval_prompts=eval_prompts,
        metric_fn=metric_fn,
        config=config,
        logit_mask=logit_mask,
    )
