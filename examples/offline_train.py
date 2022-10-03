from framework.data.accelerate_base_datatypes import PromptBatch

from framework.model.sentiment import SentimentILQLModel
from framework.orchestrator.sentiment import OfflineSentimentOrchestrator
from framework.pipeline.sentiment import SentimentPipeline
from framework.data.configs import TRLConfig

from framework.utils.loading import get_model, get_pipeline, get_orchestrator
from framework.eval.sentiment import sentiment_eval

import torch as th
from transformers import GPT2Config, AutoModel

import torch
import numpy as np
import torch as th
from torch import tensor
from torch.utils.data import TensorDataset
import torch.nn.functional as F
import networkx as nx

def randexclude(rng: np.random.RandomState, n: int, exclude: int) -> int:
    while True:
        x = rng.randint(n)
        if x != exclude:
            return x

# Toy dataset from Decision Transformer (Chen et. al 2021)
def randomwalks(n_nodes=21, max_length=10, n_walks=1000, p_edge=0.1, seed=1002):
    n_nodes = n_nodes
    n_walks = n_walks
    max_length = max_length
    rng = np.random.RandomState(seed)

    walks, rewards = [], []
    while True:
        adj = rng.rand(n_nodes, n_nodes) > (1 - p_edge)
        np.fill_diagonal(adj, 0)
        if np.all(adj.sum(1)): break

    # terminal state
    adj[0, :] = 0
    adj[0, 0] = 1

    goal = 0
    for _ in range(n_walks):
        node = randexclude(rng, n_nodes, goal)
        walk = [node]

        for istep in range(max_length-1):
            node = rng.choice(np.nonzero(adj[node])[0])
            walk.append(node)
            if node == goal:
                break

        r = th.zeros(max_length-1)
        r[:len(walk)-1] = -1 if walk[-1] == goal else -100

        rewards.append(r)
        walks.append(walk)

    states = []
    attention_masks = []

    for r, walk in zip(rewards, map(th.tensor, walks)):
        attention_mask = th.zeros(max_length, dtype=int)
        attention_mask[:len(walk)] = 1

        attention_masks.append(attention_mask)
        states.append(F.pad(walk, (0, max_length-len(walk))))

    worstlen = max_length
    avglen = sum(map(len, walks)) / n_walks
    bestlen = 0
    g = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    for start in set(range(n_nodes)) - {goal}:
        try:
            shortest_path = nx.shortest_path(g, start, goal)[:max_length]
            bestlen += len(shortest_path)
        except:
            bestlen += max_length

    bestlen /= n_nodes - 1

    print(f'{n_walks} walks of which {(np.array([r[0] for r in rewards])==-1).mean()*100:.0f}% arrived at destination')

    return th.stack(states), logit_mask == tensor(~adj)

if __name__ == "__main__":
    config = TRLConfig.load_yaml("configs/sentiment_config.yml")

    train_samples, logit_mask = randomwalks(seed=1000)
    eval_prompts = torch.arange(1, data.n_nodes).view(-1, 1)

    def reward_fn(samples):
        out = []

        for s in samples:
            for ix in range(len(s)):
                if s[ix] == 0:
                    out.append(-ix - 1)
                    break

            out.append(-100)

        return out

    gpt_config_or_path = GPT2Config(n_layer=1, n_embd=72, vocab_size=21)

    model : SentimentILQLModel = get_model('SentimentILQLModel')(
        config,
        gpt_config_or_path,
        logit_mask
    )

    train_pipeline : SentimentPipeline = get_pipeline(config.train.pipeline)(train_samples)
    eval_pipeline : SentimentPipeline = get_pipeline(config.train.pipeline)(eval_prompts)

    orch : OfflineSentimentOrchestrator = get_orchestrator(config.train.orchestrator)(
        model,
        train_pipeline,
        eval_pipeline,
        reward_fn
    )

    model.learn()
