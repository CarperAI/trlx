import json
import logging

import yaml
from lang import Interpreter

import trlx
from trlx.data.configs import TRLConfig

logger = logging.getLogger(__name__)


class DSLDataset:
    def __init__(self):
        self.train_data = json.load(open("dataset/train.json", "r"))
        self.test_data = json.load(open("dataset/test.json", "r"))
        logger.info("Sucessfully loaded the dataset")

    def load_datapoints(self, split="train"):
        if split == "train":
            for datapoint in self.train_data:
                if "ERROR" not in datapoint["input"]:
                    yield datapoint["input"]
        elif split == "test":
            for datapoint in self.test_data:
                yield datapoint["input"]


interpreter = Interpreter()


def reward_fn(samples):
    reward_list = []
    for sample in samples:
        code = sample.split("Function:")[1].strip()
        output = eval(sample.split("Output:")[1].strip().split("Function:")[0].strip())
        interpreted_output = interpreter(code)
        if interpreted_output == "ERROR":
            # If the code is unparsable, we give it a negative reward.
            reward_list.append(-1)
        else:
            # if the code is parseable
            if output == interpreted_output:
                # if the output is correct, we give it a positive reward.
                reward_list.append(1)
            else:
                # if the output is incorrect, we give it a negative reward.
                reward_list.append(-0.5)

    return reward_list


default_config = yaml.safe_load(open("configs/trlx_ppo_config.yml"))


def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)

    # Dataset
    dataset = DSLDataset()
    train_prompts = list(dataset.load_datapoints(split="train"))[:1000]

    trainer = trlx.train(
        reward_fn=reward_fn,
        prompts=train_prompts,
        config=config,
    )
    trainer.save_pretrained("dataset/trained_model")


if __name__ == "__main__":
    # TEST REWARD FUNTION
    assert (
        reward_fn(
            ["Input: 1 Output: [-4,-5,-2] Function: div_n(reverse([-2, -5, -4]),1)"]
        )
    ) == [1]
    assert (
        reward_fn(
            ["Input: 1 Output: [-4,-5,-2] Function: div_n(reverse([-2, -5, -a]),1)"]
        )
    ) == [-1]
    assert (
        reward_fn(
            ["Input: 1 Output: [-4,-5,-2] Function: div_n(reverse([-2, -5, -3]),1)"]
        )
    ) == [-0.5]

    main()
