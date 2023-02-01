from typing import Callable, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch


def generate_rand_int_excluding(rng: np.random.RandomState, max: int, exclude: int) -> int:
    """Random integer generator, excluding a specific number

    Args:
        rng: Numpy random number generator
        max: Max number
        exclude: Number to exclude

    Returns:
        Random integer in [0, max], excluding the `exclude` integer.
    """
    while True:
        # Create the random integer
        x = rng.randint(max)

        # Return the random integer if it isn't the exclude value, otherwise try
        # again
        if x != exclude:
            return x


def generate_random_walks(  # noqa: max-complexity
    n_nodes: int = 21,
    max_length: int = 10,
    n_walks: int = 1000,
    p_edge: float = 0.1,
    seed: int = 1002,
    gpt2_tokenizer: bool = False,
) -> Tuple[Callable[[List[str]], Dict[str, List[float]]], List[str], List[str], torch.Tensor,]:
    """Generate random walks

    Args:
        n_nodes: Number of nodes. This should not be more than 26, as we use
        single letters to represent each node.
        max_length: Maximum number of steps in each random walk
        n_walks: Number of random walks (samples) to create
        p_edge: Probability that any source node connects to any other
        destination node
        seed: Random seed
        gpt2_tokenizer: True if GPT2's tokenizer is being used

    Returns:
        Tuple of metric function,
    """
    # Initialise a random state with the seed
    rng = np.random.RandomState(seed)

    # Create the adjacency matrix
    # https://en.wikipedia.org/wiki/Adjacency_matrix
    # This is a 2d matrix, where the rows represent the source nodes and the
    # columns represent the destination nodes. If a cell (i,j) is True, then
    # there is a directional edge from the source node (i) to the destination
    # node (j). If it is false there is no connection.
    while True:
        # Create the adjacency matrix, where each node is connected to each
        # other node, with probability p_edge
        adjacency_matrix: np.ndarray = rng.rand(n_nodes, n_nodes) > (1 - p_edge)

        # Nodes can't be connected to themselves, so the diagonal values must
        # all be False
        np.fill_diagonal(adjacency_matrix, 0)

        # Each destination node (column) must be connected to at least one
        # source node. This checks if this is the case, by checking there is a
        # True value in every column. If it is not the case, we try to generate
        # a new adjacency matrix again from scratch (in the while loop).
        if np.all(adjacency_matrix.sum(1)):
            break

    # Set the goal node as 0
    goal: int = 0

    # The goal node is the terminal state, so we make sure that it doesn't
    # have a directional edge going to any other nodes (i.e. it can only be
    # connected to from previous nodes). We also set the connection to itself as
    # True.
    adjacency_matrix[goal, :] = 0
    adjacency_matrix[goal, goal] = 1

    # Create dicts for converting nodes into characters and vice versa
    # Nodes are converted into characters as these (when split by the delimiter) are
    # guaranteed to be tokenized as individual tokens.
    char_to_node: Dict[str, int] = {chr(ix + ord("a")): ix for ix in range(n_nodes)}
    node_to_char: Dict[int, str] = {ix: chr(ix + ord("a")) for ix in range(n_nodes)}

    # Initialise a list of sample walks
    sample_walks: List[str] = []

    # String delimiter (to force the tokenizer to keep all nodes as separate
    # tokens)
    delimiter: str = "|" if gpt2_tokenizer else ""

    # Create n_walks samples
    for _ in range(n_walks):
        # Create a random starting node (that isn't already at the goal state)
        node: int = generate_rand_int_excluding(rng, n_nodes, goal)

        # Initialise the list of nodes that we visit
        walk_nodes: List[int] = [node]

        # Do a series of steps, until we hit the maximum number of steps or the
        # goal state (whichever comes first)
        for _step in range(max_length - 1):
            # From the starting node, get all the nodes we can move to. Pick one
            # of these at random, and add it to the list of visited nodes
            node = rng.choice(np.nonzero(adjacency_matrix[node])[0])
            walk_nodes.append(node)

            # If we're at the goal state, stop
            if node == goal:
                break

        # Convert the nodes visited to letters (not integers)
        walk: List[str] = [node_to_char[ix] for ix in walk_nodes]

        # Concatenate into a journey, with each node letter separated by the
        # delimiter.
        sample_walks.append(delimiter.join(walk))

    # Initialise list of shortest lengths for each node (to the goal node)
    shortest_lengths: List[int] = []

    # Create a directional graph from the adjacency list
    directional_graph = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)

    # Fore each node (except for the goal node), find the shortest path
    for start in set(range(n_nodes)) - {goal}:
        try:
            # Find the shortest path (up to the max_length)
            shortest_path = nx.shortest_path(directional_graph, start, goal)[:max_length]
            shortest_lengths.append(len(shortest_path))
        except Exception:
            # If there is no path, use the maximum length instead
            shortest_lengths.append(max_length)

    def metric_fn(
        samples: List[str],
    ) -> Dict[str, List[float]]:
        """Metric Function

        Args:
            samples: Batch of samples

        Returns:
            Dict of metrics, each with a key of the metric name and value as a
            list of metric values for each batch item.
        """
        # Length to set if the path is invalid
        invalid_path_length: int = 100

        # Initialise batch lengths & reference lengths (the optimal length
        # starting from each batch items specific start node)
        lengths: List[float] = []
        sample_optimal_lengths: List[int] = []

        for sample_str in samples:
            # Remove GPT2 specific tokenizer delimiter
            if gpt2_tokenizer:
                sample_str = sample_str.replace("|", "")

            # Convert the sample into a list of nodes (default to an unused
            # integer if the node is not found)
            sample: List[int] = [char_to_node.get(c, 1000) for c in sample_str]

            # Initialise the specific sample length
            length: Optional[float] = None

            for node in range(len(sample)):
                # If an invalid path is taken, set the length to the invalid
                # path score
                if sample[node] >= n_nodes or node > 0 and not adjacency_matrix[sample[node - 1], sample[node]]:
                    length = invalid_path_length
                    break

                # Otherwise increment the length for each move (where we don't
                # end up at the goal node)
                elif sample[node] == 0:
                    length = node + 1
                    break

            # Catch the case where there are no moves
            if length is None:
                length = invalid_path_length

            # Store the batch item length & optimal length staring from the
            # start node
            lengths.append(float(length))
            sample_optimal_lengths.append(shortest_lengths[sample[0] - 1])

        # Calculate optimality scores, in [0, 1], as compared to the shortest
        # path
        lengths_tensor = torch.tensor(lengths, dtype=torch.float)
        bound_lengths: torch.Tensor = torch.where(
            lengths_tensor.eq(invalid_path_length), max_length, lengths_tensor
        ).abs()
        optimal_lengths = torch.as_tensor(sample_optimal_lengths)

        # Optimality scores, in [0, 1], as compared to the shortest path
        optimality = (max_length - bound_lengths) / (max_length - optimal_lengths)

        return {
            "lengths": lengths,
            "optimality": optimality.tolist(),
        }

    logit_mask = torch.tensor(adjacency_matrix)

    # Set the evaluation prompts as a list of unique random walk samples, using
    # just the start point (first character) from each samples.
    eval_prompts = list(sorted(set(w[0] for w in sample_walks)))
    eval_prompts = [prompt + delimiter for prompt in eval_prompts]

    return (metric_fn, eval_prompts, sample_walks, logit_mask)
