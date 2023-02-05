import os
from typing import List

from trlx.data.configs import TRLConfig


def _get_config_dirs(dir: str, config_dir_name: str = "configs") -> List[str]:
    """Returns all sub-directories of `dir` named `configs`."""
    config_dirs = []
    for root, dirs, _ in os.walk(dir):
        for d in dirs:
            if d == config_dir_name:
                config_dirs.append(os.path.join(root, d))
    return config_dirs


def _get_yaml_filepaths(dir: str) -> List[str]:
    """Returns a list of `yml` filepaths in `dir`."""
    filepaths = []
    for file in os.listdir(dir):
        if file.endswith(".yml"):
            filepaths.append(os.path.join(dir, file))
    return filepaths


def test_repo_trl_configs():
    """Tests to ensure all default configs in the repository are valid."""
    config_dirs = ["configs", *_get_config_dirs("examples")]
    config_files = sum(map(_get_yaml_filepaths, config_dirs), [])  # sum for flat-map behavior
    for file in config_files:
        assert os.path.isfile(file), f"Config file {file} does not exist."
        assert file.endswith(".yml"), f"Config file {file} is not a yaml file."
        try:
            config = TRLConfig.load_yaml(file)
            assert (
                config.train.entity_name is None
            ), f"Unexpected entity name in config file `{file}`. Remove before pushing to repo."
        except Exception as e:
            assert False, f"Failed to load config file `{file}` with error `{e}`"
