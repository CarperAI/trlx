import argparse

from trlx.data.configs import TRLConfig

import ray
from ray.air import session
from ray import tune

import wandb
from ray.air.callbacks.wandb import WandbLoggerCallback

from datasets import load_dataset
import trlx

from trlx.ray_tune import load_ray_yaml
from trlx.ray_tune import get_param_space




# def get_tune_config(config: dict):
#     tune_config = config["tune_config"]

#     if "search_alg" in tune_config.keys() and tune_config["search_alg"] is not None:
#         tune_config["search_alg"] = get_search_alg(tune_config)

#     if "scheduler" in tune_config.keys() and tune_config["scheduler"] is not None:
#         tune_config["scheduler"] = get_scheduler(tune_config)

#     tune_config = {k: v for k, v in tune_config.items() if v is not None}

#     return tune_config





# def ppo_sentiments_train(config):
#     from transformers import pipeline

#     sentiment_fn = pipeline("sentiment-analysis", "lvwerra/distilbert-imdb", device=-1)

#     def reward_fn(samples):
#         outputs = sentiment_fn(samples, return_all_scores=True)
#         sentiments = [output[1]["score"] for output in outputs]
#         return sentiments

#     # Take few words off of movies reviews as prompts
#     imdb = load_dataset("imdb", split="train+test")
#     prompts = [" ".join(review.split()[:4]) for review in imdb["text"]]

#     model = trlx.train(
#         "lvwerra/gpt2-imdb",
#         reward_fn=reward_fn,
#         prompts=prompts,
#         eval_prompts=["I don't know much about Hungarian underground"] * 64,
#         config=config,
#     )

#     return model


# def get_search_alg(tune_config: dict):
#     search_alg = tune_config["search_alg"]

#     if search_alg == "bayesopt":
#         try:
#             from ray.tune.search.bayesopt import BayesOptSearch
#         except ImportError:
#             raise ImportError("Please pip install bayesian-optimization to use BayesOptSearch.")

#         assert "metric" in tune_config.keys() and "mode" in tune_config.keys(); "Please specify metric and mode for BayesOptSearch."

#         return BayesOptSearch(metric=tune_config["metric"], mode=tune_config["mode"])

#     return None


# def get_scheduler(tune_config: dict):
#     scheduler = tune_config["scheduler"]

#     if scheduler == "hyperband":
#         return tune.schedulers.HyperBandScheduler()

#     return None


# def train_function(config):
#     config = TRLConfig.from_dict(config)
#     model = ppo_sentiments_train(config)


# def tune_function(train_function, param_space: dict, tune_config: dict):
#     tuner = tune.Tuner(
#         tune.with_resources(train_function, resources={"cpu": 2, "gpu": 1}),
#         param_space=param_space,
#         tune_config=tune.TuneConfig(**tune_config),
#     )

#     results = tuner.fit()
#     print("Best hyperparameters found were: ", results.get_best_result().config)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--config", type=str, default=None, required=True, help="The config file defining the param_space."
#     )
#     parser.add_argument(
#         "--num-cpus", type=int, default=4, help="Number of CPUs to use."
#     )
#     parser.add_argument(
#         "--num-gpus", type=int, default=0, help="Number of GPUs to use."
#     )
#     parser.add_argument(
#         "--smoke-test", action="store_true", help="Finish quickly for testing"
#     )
#     parser.add_argument(
#         "--server-address",
#         type=str,
#         default=None,
#         required=False,
#         help="The address of server to connect to if using Ray Client.",
#     )

#     args, _ = parser.parse_known_args()

#     # Read config and parse it
#     config = load_ray_yaml(args.config)
#     tune_config = get_tune_config(config)
#     param_space = get_param_space(config)

#     # Initialize Ray.
#     if args.smoke_test:
#         ray.init(num_cpus=args.num_cpus, num_gpus=args.num_gpus)
#     elif args.server_address:
#         ray.init(
#             num_cpus=args.num_cpus,
#             num_gpus=args.num_gpus,
#             address=f"ray://{args.server_address}"
#         )

#     tune_function(train_function, param_space, tune_config)

#     # Shut down Ray.
#     ray.shutdown()
