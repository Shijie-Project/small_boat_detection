"""Train: ``train.py -c <config> [-t <ckpt>]``, mirroring scripts/dist_train.sh."""

from pathlib import Path

from .base import (
    Feature,
    Field,
    JobSpec,
    config_path,
    existing_file,
    gpu_env,
    launcher,
    positive_int,
    python_executable,
    runtime_rows,
    timestamp,
)


class TrainFeature(Feature):
    name = "train"
    label = "Train"
    description = "Start a training run from scratch or from a tuning checkpoint."
    fields = [
        Field(
            "config",
            "Config (-c)",
            kind="choice",
            source="configs",
            value="configs/dome/Dome-M-AEA.yml",
            prefer="AEA",
        ),
        Field(
            "checkpoint",
            "Tuning checkpoint (-t, optional)",
            kind="choice",
            source="checkpoints",
            optional=True,
            empty_label="(none / from scratch)",
        ),
        *runtime_rows(port=7789),
    ]

    def build(self, params):
        config = config_path(params)
        tuning = existing_file(params, "checkpoint", "tuning checkpoint (-t)", required=False)
        seed = positive_int(params, "seed", 42)
        nproc = positive_int(params, "nproc", 1)
        port = positive_int(params, "port", 7789)

        outdir = f"output/{Path(config).stem}/{timestamp()}"
        cmd = launcher(python_executable(params), nproc, port)
        cmd += ["-c", config, "--seed", str(seed), "--output-dir", outdir]
        if tuning:
            cmd += ["-t", tuning]

        meta = {"feature": self.name, "config": config, "outdir": outdir, "cmd": " ".join(cmd)}
        return JobSpec(cmd, env=gpu_env(params), meta=meta)
