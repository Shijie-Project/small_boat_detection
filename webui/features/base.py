"""What a feature is, plus the argument handling train and test have in common.

A feature turns the form values the page posts into a :class:`JobSpec`; the app
hands that to the shared job runner. Anything a second feature would also need
(config lookup, checkpoint lookup, the torchrun prefix) lives here rather than
in the feature modules.
"""

import sys
import time

from ..core.paths import TRAIN_SCRIPT, resolve


class JobSpec:
    """A command line, the environment tweaks it needs, and what to show about it."""

    def __init__(self, cmd, env=None, meta=None):
        self.cmd = cmd
        self.env = dict(env or {})
        self.meta = dict(meta or {})


class Feature:
    """One tab in the dashboard.

    Subclasses set ``name``/``label`` and implement :meth:`build`. Override
    :meth:`register` to add endpoints of your own beyond ``/api/<name>/start``.
    """

    name = ""
    label = ""
    description = ""

    def build(self, params) -> JobSpec:
        raise NotImplementedError

    def register(self, router):
        """Hook for extra routes; the start endpoint is wired up by the app."""

    def info(self):
        return {"name": self.name, "label": self.label, "description": self.description}


# --------------------------------------------------------------------------- #
# Form value helpers. They raise ValueError, which the app turns into a 400.
# --------------------------------------------------------------------------- #
def timestamp():
    return time.strftime("%Y%m%d-%H%M%S")


def python_executable(params):
    return params.get("python") or sys.executable


def positive_int(params, key, default):
    raw = str(params.get(key, "")).strip() or str(default)
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{key} must be a number, got {raw!r}") from None
    if value < 1:
        raise ValueError(f"{key} must be >= 1, got {value}")
    return value


def existing_file(params, key, kind, required=True):
    """Validate one of the paths we listed in ``/api/options``."""
    value = str(params.get(key, "")).strip()
    if not value:
        if required:
            raise ValueError(f"{kind} is required")
        return ""
    path = resolve(value)
    if path is None or not path.is_file():
        raise ValueError(f"{kind} not found: {value}")
    return value


def config_path(params):
    return existing_file(params, "config", "config (-c)")


def gpu_env(params):
    gpus = str(params.get("gpus", "")).strip()
    return {"CUDA_VISIBLE_DEVICES": gpus} if gpus else {}


def launcher(python, nproc, port):
    """``train.py`` under torchrun when several GPUs are asked for, else plain python."""
    if nproc > 1:
        return [
            python,
            "-m",
            "torch.distributed.run",
            f"--master_port={port}",
            f"--nproc_per_node={nproc}",
            TRAIN_SCRIPT,
        ]
    return [python, TRAIN_SCRIPT]
