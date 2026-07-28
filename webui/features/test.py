"""Test: ``train.py -c <config> -r <ckpt> --test-only``, mirroring scripts/dist_test.sh.

``train.py`` writes the results next to the checkpoint it evaluated, so there is
no ``--output-dir`` to pass here.
"""

from pathlib import Path

from .base import (
    Feature,
    JobSpec,
    config_path,
    existing_file,
    gpu_env,
    launcher,
    positive_int,
    python_executable,
)


class TestFeature(Feature):
    name = "test"
    label = "Test"
    description = "Evaluate a checkpoint; results land beside the checkpoint."

    def build(self, params):
        config = config_path(params)
        checkpoint = existing_file(params, "checkpoint", "checkpoint (-r)")
        seed = positive_int(params, "seed", 42)
        nproc = positive_int(params, "nproc", 1)
        port = positive_int(params, "port", 7778)

        cmd = launcher(python_executable(params), nproc, port)
        cmd += ["-c", config, "-r", checkpoint, "--test-only", "--seed", str(seed)]

        outdir = str(Path(checkpoint).parent).replace("\\", "/")
        meta = {"feature": self.name, "config": config, "outdir": outdir, "cmd": " ".join(cmd)}
        return JobSpec(cmd, env=gpu_env(params), meta=meta)
