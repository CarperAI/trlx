# Copyright 2023 Optuna, Hugging Face, CarperAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Logging utilities."""

import logging
import os
import sys
import threading
from logging import CRITICAL  # NOQA
from logging import DEBUG  # NOQA
from logging import ERROR  # NOQA
from logging import FATAL  # NOQA
from logging import INFO  # NOQA
from logging import NOTSET  # NOQA
from logging import WARN  # NOQA
from logging import WARNING  # NOQA
from typing import Optional

import torch
from tqdm import auto as tqdm_lib

_lock = threading.Lock()
_default_handler: Optional[logging.Handler] = None

log_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

_default_log_level = logging.INFO


def _get_default_logging_level():
    """
    If `TRLX_VERBOSITY` env var is set to one of the valid choices, return that as the new default level. If it is
    not - fall back to `_default_log_level`
    """
    env_level_str = os.getenv("TRLX_VERBOSITY", None)
    if env_level_str:
        if env_level_str.lower() in log_levels:
            return log_levels[env_level_str.lower()]
        else:
            logging.getLogger().warning(
                f"Unknown option TRLX_VERBOSITY={env_level_str}, " f"has to be one of: { ', '.join(log_levels.keys()) }"
            )
    return _default_log_level


def _get_library_name() -> str:
    return __name__.split(".")[0]


def _get_library_root_logger() -> logging.Logger:
    return logging.getLogger(_get_library_name())


def _configure_library_root_logger() -> None:
    global _default_handler

    with _lock:
        if _default_handler:
            # This library has already configured the library root logger.
            return
        _default_handler = logging.StreamHandler()  # Set sys.stderr as stream.
        _default_handler.flush = sys.stderr.flush

        # Apply our default configuration to the library root logger.
        library_root_logger = _get_library_root_logger()
        library_root_logger.addHandler(_default_handler)
        library_root_logger.setLevel(_get_default_logging_level())
        library_root_logger.propagate = False


def _reset_library_root_logger() -> None:
    global _default_handler

    with _lock:
        if not _default_handler:
            return

        library_root_logger = _get_library_root_logger()
        library_root_logger.removeHandler(_default_handler)
        library_root_logger.setLevel(logging.NOTSET)
        _default_handler = None


def get_log_levels_dict():
    return log_levels


class MultiProcessAdapter(logging.LoggerAdapter):
    """A logger adapter for handling multi-process logging"""

    def log(self, level, msg, *args, **kwargs):
        """
        Consumes an additional kwarg called `ranks` to determine which processes should log.
        NOTE: To specify all processes, pass in an empty list `ranks=[]`

        Default: ["0"], i.e. only the main process logs
        """
        # By default, silence all non-main processes
        ranks = kwargs.pop("ranks", ["0"])
        should_log = os.environ.get("RANK", "0") in ranks or len(ranks) == 0
        if self.isEnabledFor(level) and should_log:
            msg, kwargs = self.process(msg, kwargs)
            self.logger._log(level, msg, args, **kwargs)

    def process(self, msg, kwargs):
        this_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        return f"[RANK {this_rank}] {msg}", kwargs


def get_logger(name: Optional[str] = None) -> MultiProcessAdapter:
    """
    Returns a `logging.Logger` for `name` that can handle multiple processes

    Args:
        name: Name of the logger

    Usage:
        >> logger = get_logger(__name__)
        >> logger.debug("Check the...", ranks=["0", "1"])  # Only main and rank 1 log
    """
    if name is None:
        name = _get_library_name()
    _configure_library_root_logger()
    logger = logging.getLogger(name)
    return MultiProcessAdapter(logger, {})


def get_verbosity() -> int:
    """
    Return the current level for trlx's root logger as an int.
    Returns:
        `int`: The logging level.
    <Tip>
    trlx has following logging levels:
    - 50: `trlx.logging.CRITICAL` or `trlx.logging.FATAL`
    - 40: `trlx.logging.ERROR`
    - 30: `trlx.logging.WARNING` or `trlx.logging.WARN`
    - 20: `trlx.logging.INFO`
    - 10: `trlx.logging.DEBUG`
    </Tip>
    """

    _configure_library_root_logger()
    return _get_library_root_logger().getEffectiveLevel()


def set_verbosity(verbosity: int) -> None:
    """
    Set the verbosity level for trlX's root logger.
    Args:
        verbosity (`int`):
            Logging level, e.g., one of:
            - `trlx.logging.CRITICAL` or `trlx.logging.FATAL`
            - `trlx.logging.ERROR`
            - `trlx.logging.WARNING` or `trlx.logging.WARN`
            - `trlx.logging.INFO`
            - `trlx.logging.DEBUG`
    """

    _configure_library_root_logger()
    _get_library_root_logger().setLevel(verbosity)


def disable_default_handler() -> None:
    """Disable the default handler of trlx's root logger."""

    _configure_library_root_logger()

    assert _default_handler is not None
    _get_library_root_logger().removeHandler(_default_handler)


def enable_default_handler() -> None:
    """Enable the default handler of trlx's root logger."""

    _configure_library_root_logger()

    assert _default_handler is not None
    _get_library_root_logger().addHandler(_default_handler)


def add_handler(handler: logging.Handler) -> None:
    """Adds a handler to trlx's root logger."""

    _configure_library_root_logger()

    assert handler is not None
    _get_library_root_logger().addHandler(handler)


def remove_handler(handler: logging.Handler) -> None:
    """Removes given handler from the trlx's root logger."""

    _configure_library_root_logger()

    assert handler is not None and handler not in _get_library_root_logger().handlers
    _get_library_root_logger().removeHandler(handler)


def disable_propagation() -> None:
    """
    Disable propagation of the library log outputs. Note that log propagation is disabled by default.
    """

    _configure_library_root_logger()
    _get_library_root_logger().propagate = False


def enable_propagation() -> None:
    """
    Enable propagation of the library log outputs. Please disable the trlx's default handler to prevent
    double logging if the root logger has been configured.
    """

    _configure_library_root_logger()
    _get_library_root_logger().propagate = True


def enable_explicit_format() -> None:
    """
    Enable explicit formatting for every trlx's logger. The explicit formatter is as follows:
    ```
        [ASCTIME] [LEVELNAME] [FILENAME:LINE NUMBER:FUNCNAME] MESSAGE
    ```
    All handlers currently bound to the root logger are affected by this method.
    """
    handlers = _get_library_root_logger().handlers

    for handler in handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s] %(message)s"
        )
        handler.setFormatter(formatter)


def reset_format() -> None:
    """
    Resets the formatting for trlx's loggers.
    All handlers currently bound to the root logger are affected by this method.
    """
    handlers = _get_library_root_logger().handlers

    for handler in handlers:
        handler.setFormatter(None)


def warning_advice(self, *args, **kwargs):
    """
    This method is identical to `logger.warning()`, but if env var TRLX_NO_ADVISORY_WARNINGS=1 is set, this
    warning will not be printed
    """
    no_advisory_warnings = os.getenv("TRLX_NO_ADVISORY_WARNINGS", False)
    if no_advisory_warnings:
        return
    self.warning(*args, **kwargs)


logging.Logger.warning_advice = warning_advice


class EmptyTqdm:
    """Dummy tqdm which doesn't do anything."""

    def __init__(self, *args, **kwargs):  # pylint: disable=unused-argument
        self._iterator = args[0] if args else None

    def __iter__(self):
        return iter(self._iterator)

    def __getattr__(self, _):
        """Return empty function."""

        def empty_fn(*args, **kwargs):  # pylint: disable=unused-argument
            return

        return empty_fn

    def __enter__(self):
        return self

    def __exit__(self, type_, value, traceback):
        return


_tqdm_active = True


class _tqdm_cls:
    def __call__(self, *args, **kwargs):
        if _tqdm_active:
            return tqdm_lib.tqdm(*args, **kwargs)
        else:
            return EmptyTqdm(*args, **kwargs)

    def set_lock(self, *args, **kwargs):
        self._lock = None
        if _tqdm_active:
            return tqdm_lib.tqdm.set_lock(*args, **kwargs)

    def get_lock(self):
        if _tqdm_active:
            return tqdm_lib.tqdm.get_lock()


tqdm = _tqdm_cls()


def is_progress_bar_enabled() -> bool:
    """Return a boolean indicating whether tqdm progress bars are enabled."""
    global _tqdm_active
    return bool(_tqdm_active)


def enable_progress_bar():
    """Enable tqdm progress bar."""
    global _tqdm_active
    _tqdm_active = True


def disable_progress_bar():
    """Disable tqdm progress bar."""
    global _tqdm_active
    _tqdm_active = False
