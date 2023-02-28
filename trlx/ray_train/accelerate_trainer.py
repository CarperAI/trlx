import os
import tempfile
from argparse import Namespace
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple, Type, Union

from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.air.config import DatasetConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchConfig, get_device
from ray.train.trainer import GenDataset

if TYPE_CHECKING:
    from ray.data.preprocessor import Preprocessor
    from ray.tune.trainable import Trainable

from accelerate.commands.config import default_config_file, load_config_from_file
from ray.train.torch import TorchTrainer

from .launch import launch_command, launch_command_parser


class _AccelerateDefaultNamespace(Namespace):
    @property
    def parser(self):
        return launch_command_parser()

    def __getattr__(self, name: str):
        return self.parser.get_default(name)


class _AccelerateConfigWrapper:
    """
    Lets Trainables know to treat this as already loaded file content instead of path.
    """

    def __init__(self, config_raw: str, deepspeed_config_raw: Optional[str] = None) -> None:
        self.config_raw = config_raw
        self.deepspeed_config_raw = deepspeed_config_raw

    def __bool__(self) -> bool:
        return bool(self.config_raw)


class AccelerateTrainer(TorchTrainer):
    def __init__(
        self,
        train_loop_per_worker: Union[Callable[[], None], Callable[[Dict], None]],
        *,
        accelerate_config_path: Union[str, Path, os.PathLike],
        train_loop_config: Optional[Dict] = None,
        torch_config: Optional[TorchConfig] = None,
        scaling_config: Optional[ScalingConfig] = None,
        dataset_config: Optional[Dict[str, DatasetConfig]] = None,
        run_config: Optional[RunConfig] = None,
        datasets: Optional[Dict[str, GenDataset]] = None,
        preprocessor: Optional["Preprocessor"] = None,
        resume_from_checkpoint: Optional[Checkpoint] = None,
    ):
        self.accelerate_config_path = accelerate_config_path or default_config_file
        if isinstance(self.accelerate_config_path, _AccelerateConfigWrapper):
            self._accelerate_config_raw = self.accelerate_config_path.config_raw
            self._deepspeed_config_file_raw = self.accelerate_config_path.deepspeed_config_raw
        else:
            (
                self._accelerate_config_raw,
                self._deepspeed_config_file_raw,
            ) = self._load_accelerate_config()
        super().__init__(
            train_loop_per_worker,
            train_loop_config=train_loop_config,
            torch_config=torch_config,
            scaling_config=scaling_config,
            dataset_config=dataset_config,
            run_config=run_config,
            datasets=datasets,
            preprocessor=preprocessor,
            resume_from_checkpoint=resume_from_checkpoint,
        )

    def training_loop(self) -> None:
        old_train_loop_per_worker = self._train_loop_per_worker
        self._train_loop_per_worker = self._wrap_train_loop_per_worker(
            self._train_loop_per_worker,
            self._accelerate_config_raw,
            self._deepspeed_config_file_raw,
        )
        try:
            ret = super().training_loop()
        finally:
            self._train_loop_per_worker = old_train_loop_per_worker
        return ret

    def as_trainable(self) -> Type["Trainable"]:
        # We want to load the config when the Trainer is first instantiated,
        # and share the contents with the Trainables (which may be on different)
        # nodes
        old_accelerate_config_path = self._param_dict["accelerate_config_path"]
        self._param_dict["accelerate_config_path"] = _AccelerateConfigWrapper(
            self._accelerate_config_raw, self._deepspeed_config_file_raw
        )
        try:
            ret = super().as_trainable()
        finally:
            self._param_dict["accelerate_config_path"] = old_accelerate_config_path
        return ret

    def _load_accelerate_config(self) -> Tuple[str, Optional[str]]:
        # We only load config to dict to obtain the deepspeed_config_file
        config = load_config_from_file(self.accelerate_config_path)
        deepspeed_config_file = getattr(config, "deepspeed_config_file", None)
        deepspeed_config_file_raw = None

        if deepspeed_config_file:
            with open(deepspeed_config_file, "r") as f:
                deepspeed_config_file_raw = f.read()

        # Otherwise, we want to pass raw contents to Trainables for maximum
        # compatibility.
        with open(self.accelerate_config_path, "r") as f:
            return f.read(), deepspeed_config_file_raw

    @classmethod
    def _wrap_train_loop_per_worker(
        cls,
        train_loop_per_worker: Union[Callable[[], None], Callable[[Dict], None]],
        accelerate_config_raw: str,
        deepspeed_config_file_raw: str,
    ):
        """Wrap around train_loop_per_worker to set necessary Accelerate env vars."""

        @wraps(train_loop_per_worker)
        def wrapped_train_loop_per_worker(*args, **kwargs):
            with tempfile.TemporaryDirectory() as tempdir:
                temp_config_file = os.path.join(tempdir, "default_config.yaml")
                with open(temp_config_file, "w") as f:
                    f.write(accelerate_config_raw)

                # Set by TorchBackend
                master_addr = os.environ["MASTER_ADDR"]
                master_port = os.environ["MASTER_PORT"]

                namespace = _AccelerateDefaultNamespace()
                namespace.config_file = temp_config_file
                namespace.num_processes = 1
                namespace.num_machines = session.get_world_size()
                namespace.machine_rank = session.get_world_rank()
                namespace.num_cpu_threads_per_process = session.get_trial_resources().bundles[-1]["CPU"]
                namespace.gpu_ids = None
                namespace.main_process_ip = master_addr
                namespace.main_process_port = master_port

                if deepspeed_config_file_raw:
                    deepspeed_config_file = os.path.join(tempdir, "deepspeed_config.json")
                    with open(deepspeed_config_file, "w") as f:
                        f.write(deepspeed_config_file_raw)
                    namespace.deepspeed_config_file = deepspeed_config_file

                launch_command(namespace)

                os.environ["MASTER_ADDR"] = master_addr
                os.environ["MASTER_PORT"] = master_port
                os.environ["RANK"] = str(session.get_world_rank())
                os.environ["WORLD_RANK"] = str(session.get_world_rank())
                os.environ["CROSS_RANK"] = str(session.get_world_rank())
                os.environ["CROSS_SIZE"] = str(session.get_world_size())
                os.environ["WORLD_SIZE"] = str(session.get_world_size())
                os.environ["LOCAL_RANK"] = str(session.get_local_rank())
                os.environ["LOCAL_WORLD_SIZE"] = str(session.get_local_world_size())
                os.environ["LOCAL_SIZE"] = str(session.get_local_world_size())
                os.environ["ACCELERATE_TORCH_DEVICE"] = str(get_device())

                return train_loop_per_worker(*args, **kwargs)

        return wrapped_train_loop_per_worker
