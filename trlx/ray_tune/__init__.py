from ray import tune


def get_param_space(config: dict):  # noqa: C901
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
            assert 2 <= len(value["values"]) <= 3
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

    for k, v in config.items():
        if k != "tune_config":
            config[k] = get_strategy(v)

    return config


def get_search_alg(tune_config: dict):
    """Initialize the search algorithm and return it.

    Bayesian Optimization is currently supported.
    """
    search_alg = tune_config["search_alg"]

    if search_alg == "bayesopt":
        try:
            from ray.tune.search.bayesopt import BayesOptSearch
        except ImportError:
            raise ImportError(
                "Please pip install bayesian-optimization to use BayesOptSearch."
            )

        assert "metric" in tune_config.keys() and "mode" in tune_config.keys()
        "Please specify metric and mode for BayesOptSearch."

        return BayesOptSearch(metric=tune_config["metric"], mode=tune_config["mode"])
    elif search_alg == "bohb":
        try:
            from ray.tune.search.bohb import TuneBOHB
        except ImportError:
            raise ImportError(
                "Please pip install hpbandster and ConfigSpace to use TuneBOHB."
            )

        assert "metric" in tune_config.keys() and "mode" in tune_config.keys()
        "Please specify metric and mode for TuneBOHB."

        return TuneBOHB()
    elif search_alg == "random":
        return None
    else:
        NotImplementedError("Search algorithm not supported.")


def get_scheduler(tune_config: dict):
    """Initialize the scheduler and return it.

    The schedulers can early terminate bad trials, pause trials,
    clone trials, and alter hyperparameters of a running trial.

    Refer to the documentation for more info:
    https://docs.ray.io/en/latest/tune/api_docs/schedulers.html#tune-schedulers

    Currently available schedulers are:
        - `hyperband` - Implements the HyperBand early stopping algorithm.

    """
    scheduler = tune_config["scheduler"]

    if scheduler == "hyperband":
        return tune.schedulers.HyperBandScheduler()
    elif scheduler == "hyperbandforbohb":
        return tune.schedulers.HyperBandForBOHB()
    elif scheduler == "fifo":
        return None
    else:
        NotImplementedError("Scheduler not supported.")


def get_tune_config(tune_config: dict):
    """Get the tune config to initialized `tune.TuneConfig`
    to be passed `tune.Tuner`.
    """
    if "search_alg" in tune_config.keys() and tune_config["search_alg"] is not None:
        tune_config["search_alg"] = get_search_alg(tune_config)

    if "scheduler" in tune_config.keys() and tune_config["scheduler"] is not None:
        tune_config["scheduler"] = get_scheduler(tune_config)

    # Remove config keys with None values.
    tune_config = {k: v for k, v in tune_config.items() if v is not None}

    return tune_config
