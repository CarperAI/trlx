# python -m trlx.sweep --config configs/sweeps/ppo_sweep.yml examples/ppo_sentiments.py
import argparse
import importlib
import json
from datetime import datetime

import ray
import wandb
import wandb.apis.reports as wb
import yaml
from ray import tune
from ray.air import ScalingConfig
from ray.train.huggingface.accelerate import AccelerateTrainer
from ray.tune.logger import CSVLoggerCallback


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
            raise ImportError("Please pip install bayesian-optimization to use BayesOptSearch.")

        assert "metric" in tune_config.keys() and "mode" in tune_config.keys()
        "Please specify metric and mode for BayesOptSearch."

        return BayesOptSearch(metric=tune_config["metric"], mode=tune_config["mode"])
    elif search_alg == "bohb":
        try:
            from ray.tune.search.bohb import TuneBOHB
        except ImportError:
            raise ImportError("Please pip install hpbandster and ConfigSpace to use TuneBOHB.")

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


def create_report(target_metric, column_names, entity_name, project_name, group_name, best_config):
    report = wb.Report(
        project=project_name,
        title=f"Hyperparameter Optimization Report: {project_name}",
        description=group_name,
    )

    report.blocks = [
        wb.PanelGrid(
            panels=[
                wb.ParallelCoordinatesPlot(
                    columns=[wb.PCColumn(f"c::{column}") for column in column_names] + [wb.PCColumn(target_metric)],
                    layout={"x": 0, "y": 0, "w": 12 * 2, "h": 5 * 2},
                ),
                wb.ParameterImportancePlot(
                    with_respect_to=target_metric,
                    layout={"x": 0, "y": 5, "w": 6 * 2, "h": 4 * 2},
                ),
                wb.ScatterPlot(
                    # Get it from the metric name.
                    title=f"{target_metric} v. Index",
                    x="Index",
                    y=target_metric,
                    running_ymin=True,
                    font_size="small",
                    layout={"x": 6, "y": 5, "w": 6 * 2, "h": 4 * 2},
                ),
            ],
            runsets=[
                wb.Runset(project=project_name).set_filters_with_python_expr(f'group == "{group_name}"'),
            ],
        ),
    ]

    entity_project = f"{entity_name}/{project_name}" if entity_name else project_name
    api = wandb.Api()
    runs = api.runs(entity_project)

    for run in runs:
        if run.group == group_name:
            history = run.history()
            metrics = history.columns
            break

    metrics = [metric for metric in metrics if not metric.startswith("_")]

    line_plot_panels = []
    for metric in metrics:
        line_plot_panels.append(
            wb.LinePlot(
                title=f"{metric}",
                x="Step",
                y=[f"{metric}"],
                title_x="Step",
                smoothing_show_original=True,
                max_runs_to_show=100,
                plot_type="line",
                font_size="auto",
                legend_position="north",
            )
        )

    report.blocks = report.blocks + [
        wb.H1(text="Metrics"),
        wb.PanelGrid(
            panels=line_plot_panels,
            runsets=[wb.Runset(project=project_name).set_filters_with_python_expr(f'group == "{group_name}"')],
        ),
    ]

    if best_config:
        best_config = best_config["train_loop_config"]
        config = {}
        for name, value in best_config.items():
            *layers, var = name.split(".")
            if layers:
                d = config.setdefault(layers[0], {})
                for layer in layers[1:]:
                    d = d.setdefault(layer, {})
                d[var] = value

        report.blocks = report.blocks + [
            wb.H1(text="Best Config"),
            wb.CodeBlock(code=[json.dumps(config, indent=4)], language="json"),
        ]

    report.save()
    print(report.url)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("script", type=str, help="Path to the example script")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="The config file defining the param_space.",
    )

    parser.add_argument(
        "--accelerate_config",
        type=str,
        required=False,
        help="The default config file for the script.",
    )
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs (workers) to use per trial.")
    parser.add_argument("--num_cpus", type=int, default=4, help="Number of CPUs to use per GPU (worker).")
    parser.add_argument("-y", "--assume_yes", action="store_true", help="Don't ask for confirmation")
    parser.add_argument(
        "--server_address",
        type=str,
        default=None,
        required=False,
        help="The address of server to connect to if using Ray Client.",
    )

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    tune_config = get_tune_config(config.pop("tune_config"))
    param_space = get_param_space(config)
    column_names = list(param_space.keys())
    target_metric = tune_config["metric"]

    if args.server_address:
        ray.init(address=f"ray://{args.server_address}")
    else:
        ray.init()

    print(f'WARNING: Importing main from "{args.script}" and everything along with it')

    if not args.assume_yes:
        print("Please confirm y/n: ", end="")
        if input() != "y":
            print("Exiting")
            exit(1)

    # convert a nested path to a module path
    script_path = args.script.replace(".py", "").replace("/", ".")
    script = importlib.import_module(script_path)
    project_name = "sweep_" + script_path.split(".")[-1]

    param_space["train.project_name"] = project_name
    param_space["train.group_name"] = datetime.now().replace(microsecond=0).isoformat()
    param_space_train = {"train_loop_config": param_space}

    tuner = tune.Tuner(
        AccelerateTrainer(
            script.main,
            # Mandatory arg. None means use Accelerate default path
            accelerate_config=args.accelerate_config,
            scaling_config=ScalingConfig(
                trainer_resources={"CPU": 0},
                num_workers=args.num_gpus,
                use_gpu=True,
                resources_per_worker={"CPU": args.num_cpus, "GPU": 1},
            ),
        ),
        param_space=param_space_train,
        tune_config=tune.TuneConfig(**tune_config),
        run_config=ray.air.RunConfig(local_dir="ray_results", callbacks=[CSVLoggerCallback()]),
    )

    results = tuner.fit()
    group_name = param_space["train.group_name"]
    entity_name = param_space.get("train.entity_name", None)

    create_report(target_metric, column_names, entity_name, project_name, group_name, results.get_best_result().config)

    ray.shutdown()
