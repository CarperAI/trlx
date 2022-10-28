import yaml


def load_ray_yaml(path: str):
    with open(path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def get_param_space(config: dict):
    """Get the param space from the config file."""

    def get_strategy(value):
        """Get search space strategy from config.
        A search space defines valid values for your hyperparameters and
        can specify how these values are sampled.

        Refer to the documentation for more info:
        https://docs.ray.io/en/latest/tune/api_docs/search_space.html#tune-sample-docs

        The user will have to define the search space in the config file by providing
        the name of the `strategy` and the `values` to sample from.

        The valid strategies are:
        - `uniform` (List) - Samples uniformly between the given bounds.
        - `quniform` (List) - Samples uniformly between the given bounds, quantized.
        - `loguniform` (List) - Samples uniformly between the given bounds on a log scale.
        - `qloguniform` (List) - Samples uniformly between the given bounds on a log scale, quantized.
        - `randn` (List) - Samples from a normal distribution.
        - `qrandn` (List) - Samples from a normal distribution, quantized.
        - `randint` (List) - Samples uniformly between the given bounds, quantized to integers.
        - `qrandint` (List) - Samples uniformly between the given bounds, quantized to integers.
        - `lograndint` (List) - Samples uniformly between the given bounds on a log scale, quantized to integers.
        - `qlograndint` (List) - Samples uniformly between the given bounds on a log scale, quantized to integers.
        - `choice` (List) - Samples from a discrete set of values.
        - `qrandn` (List) - Samples from a normal distribution, quantized.
        - `grid_search` (List) - Samples from the given list of values.

        """

        strategy = value["strategy"]
        if strategy == "uniform":
            assert isinstance(value["values"], list)
            assert len(value["values"]) == 2
            return tune.uniform(*value["values"])
        elif strategy == "quniform":
            assert isinstance(value["values"], list)
            assert len(value["values"]) == 3
            return tune.quniform(*value["values"])
        elif strategy == "loguniform":
            assert isinstance(value["values"], list)
            assert len(value["values"]) == 3
            return tune.loguniform(*value["values"])
        elif strategy == "qloguniform":
            assert isinstance(value["values"], list)
            assert len(value["values"]) == 4
            return tune.qloguniform(*value["values"])
        elif strategy == "randn":
            assert isinstance(value["values"], list)
            assert len(value["values"]) == 2
            return tune.randn(*value["values"])
        elif strategy == "qrandn":
            assert isinstance(value["values"], list)
            assert len(value["values"]) == 3
            return tune.qrandn(*value["values"])
        elif strategy == "randint":
            assert isinstance(value["values"], list)
            assert len(value["values"]) == 2
            return tune.randint(*value["values"])
        elif strategy == "qrandint":
            assert isinstance(value["values"], list)
            assert len(value["values"]) == 3
            return tune.qrandint(*value["values"])
        elif strategy == "lograndint":
            assert isinstance(value["values"], list)
            assert len(value["values"]) == 3
            return tune.lograndint(*value["values"])
        elif strategy == "qlograndint":
            assert isinstance(value["values"], list)
            assert len(value["values"]) == 4
            return tune.qlograndint(*value["values"])
        elif strategy == "choice":
            assert isinstance(value["values"], list)
            return tune.choice(value["values"])
        elif strategy == "grid":
            assert isinstance(value["values"], list)
            return tune.grid_search(value["values"])

    def parse_param_space(param_space: dict):
        """Parse the param space from the config file by
        replacing the strategies with relevant distribution APIs.

        """
        for k, v in param_space.items():
            if isinstance(v, dict):
                if "strategy" in v.keys():
                    strategy = get_strategy(v)
                    param_space[k] = strategy

        return param_space

    model_dict = parse_param_space(config["model"])
    train_dict = parse_param_space(config["train"])
    method_dict = parse_param_space(config["method"])

    return {"model": model_dict, "train": train_dict, "method": method_dict}
