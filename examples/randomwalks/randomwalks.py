import networkx as nx
import numpy as np
import torch


def randexclude(rng: np.random.RandomState, n: int, exclude: int) -> int:
    while True:
        x = rng.randint(n)
        if x != exclude:
            return x


def generate_random_walks(  # noqa: max-complexity
    n_nodes=21, max_length=10, n_walks=1000, p_edge=0.1, seed=1002, gpt2_tokenizer=False
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

    char_to_node = {chr(ix + ord("a")): ix for ix in range(n_nodes)}
    node_to_char = {ix: chr(ix + ord("a")) for ix in range(n_nodes)}

    goal = 0
    sample_walks = []
    delimiter = "|" if gpt2_tokenizer else ""

    for _ in range(n_walks):
        node = randexclude(rng, n_nodes, goal)
        walk = [node]

        for istep in range(max_length - 1):
            node = rng.choice(np.nonzero(adj[node])[0])
            walk.append(node)
            if node == goal:
                break

        # code each node by a letter
        # for bpe tokenizer join them over | for a guaranteed split
        walk = [node_to_char[ix] for ix in walk]

        sample_walks.append(delimiter.join(walk))

    # calculate the shortest paths for comparison
    shortest_lengths = []
    g = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    for start in set(range(n_nodes)) - {goal}:
        try:
            shortest_path = nx.shortest_path(g, start, goal)[:max_length]
            shortest_lengths.append(len(shortest_path))
        except Exception:
            shortest_lengths.append(max_length)

    shortest_lengths = torch.tensor(shortest_lengths)

    def metric_fn(samples):
        # a measure for an invalid or a not found path
        infty = 100
        lengths = []
        ref_lengths = []

        for s in samples:
            if gpt2_tokenizer:
                s = s.replace("|", "")

            s = [char_to_node.get(c, 1000) for c in s]
            length = None
            for ix in range(len(s)):
                # a nonexisting path is taken
                if s[ix] >= n_nodes or ix > 0 and not adj[s[ix - 1], s[ix]]:
                    length = infty
                    break
                elif s[ix] == 0:
                    length = ix + 1
                    break

            if length is None:
                length = infty

            lengths.append(length)
            # allows for inorder checking of % optimality
            ref_lengths.append(shortest_lengths[s[0] - 1])

        lengths = torch.tensor(lengths, dtype=torch.float)
        bound_lengths = torch.where(lengths.eq(infty), max_length, lengths).abs()
        ref_lengths = torch.as_tensor(ref_lengths)

        return {
            "lengths": lengths,
            # percentage-optimal \in (0, 1) when compared to the shortest path
            "optimality": (max_length - bound_lengths) / (max_length - ref_lengths),
        }

    logit_mask = torch.tensor(adj)

    eval_prompts = list(sorted(set(w[0] for w in sample_walks)))
    eval_prompts = [prompt + delimiter for prompt in eval_prompts]

    return metric_fn, eval_prompts, sample_walks, logit_mask
