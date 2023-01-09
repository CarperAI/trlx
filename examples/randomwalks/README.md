Toy problem similar to the one described in [Decision Transformer (Lili Chen et al. 2021)](https://arxiv.org/abs/2106.01345) [1]:
finding graph's shortest paths by learning from a dataset of sampled random
walks.

In this implementation there are not environment dynamics – impossible and
incorrect paths are penalized the same way by a single reward which is given at
the end of the trajectory, measuring how optimal the path is compared to the
shortest possible (bounded in [0, 1]). Paths are represented as strings of
letters, with each letter corresponding to a node in a graph. PPO example uses a
pretrained model for starting transition probabilities, ILQL learns them from
the samples directly.

[1] code for which is not present in the official repo, see issue
https://github.com/kzl/decision-transformer/issues/48
