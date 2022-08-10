from torch.utils.data import Dataset
from datasets import load_from_disk

from framework.data import GeneralElement

class BasePipeline(Dataset):
    def __init__(self, path : str = "dataset"):
        dataset = load_from_disk(path)
    
        # TODO

    def __getitem__(self, index : int) -> GeneralElement:
        pass

    def __len__(self) -> int:
        pass