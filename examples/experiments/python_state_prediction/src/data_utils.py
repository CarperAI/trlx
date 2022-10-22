import datasets
import torch
import random
import numpy as np
import logging



def set_seed(random_seed):
    """
    Random number fixed
    """
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    random.seed(random_seed)
    np.random.seed(random_seed)

TRAIN_TEST_SPLIT = 0.1

def get_python_state_change_dataset(dataset_idt:str="Fraser/python-state-changes"):
    return datasets.load_dataset(dataset_idt,"default")#.train_test_split(test=TRAIN_TEST_SPLIT)

if __name__ == "__main__":
    python_state_dataset = get_python_state_change_dataset()
    print(python_state_dataset)

