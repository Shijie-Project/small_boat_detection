"""Label Studio: ``label-studio start --data-dir <dir> -p <port>``, mirroring tools/starter.sh.

Unlike train and test this is a server, not a run that finishes: it stays up
until Stop. It wants no GPU, so it runs in its own slot (``core/jobs.py``) and
training can start, finish and start again underneath it.
"""

import os
import shutil
import sys
from pathlib import Path

from ..core.jobs import SERVICE_SLOT
from ..core.paths import DATA_DIRS, rel, resolve
from .base import Feature, Field, JobSpec, positive_int, text


def executable(params):
    """The ``label-studio`` command: the form's, else the one next to this python.

    ``label_studio`` ships no ``__main__``, so ``-m`` is not an option -- it is
    the console script or nothing. The interpreter's own Scripts/bin directory
    comes first, so a venv wins over whatever PATH happens to say.
    """
    given = text(params, "executable")
    if given:
        path = Path(given)
        if not path.is_file():
            found = shutil.which(given)
            if found is None:
                raise ValueError(f"label-studio executable not found: {given}")
            return found
        return str(path)

    names = ("label-studio.exe", "label-studio.cmd") if os.name == "nt" else ("label-studio",)
    for name in names:
        candidate = Path(sys.executable).parent / name
        if candidate.is_file():
            return str(candidate)
    found = shutil.which("label-studio")
    if found is None:
        raise ValueError(
            "label-studio not found next to the running python or on PATH -- "
            "`pip install label-studio`, or fill in the executable field."
        )
    return found


def directory(params, key, kind, must_exist=True):
    """One of the two directory fields, kept inside the trees the UI may touch."""
    value = text(params, key)
    if not value:
        raise ValueError(f"{kind} is required")
    path = resolve(value)
    if path is None:
        raise ValueError(f"{kind} is outside the project: {value}")
    if must_exist and not path.is_dir():
        raise ValueError(f"{kind} not found: {value}")
    if not must_exist and not path.parent.is_dir():
        raise ValueError(f"{kind}: parent directory does not exist: {value}")
    return path


def domain(params):
    """The ngrok domain, with the scheme the user probably pasted stripped off."""
    value = text(params, "ngrok")
    for prefix in ("https://", "http://"):
        if value.startswith(prefix):
            value = value[len(prefix) :]
    return value.rstrip("/")


class LabelStudioFeature(Feature):
    name = "label-studio"
    label = "Label Studio"
    slot = SERVICE_SLOT
    description = (
        "Start the annotation server. It keeps running until you press Stop "
        "here, and does not hold up train / test — its log is the *Services* "
        "tab of the console."
    )
    fields = [
        Field(
            "data_dir",
            "Data dir (--data-dir)",
            value=rel(DATA_DIRS),
            info="Where Label Studio keeps its own database and media.",
        ),
        Field(
            "local_files_root",
            "Local files document root",
            value=rel(DATA_DIRS),
            info="LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT — the images it may serve.",
        ),
        [
            Field("port", "Port (-p)", value="80"),
            Field(
                "ngrok",
                "ngrok domain (optional)",
                info="e.g. obliging-maggot-frank.ngrok-free.app — enables public access.",
            ),
        ],
        Field(
            "executable",
            "label-studio executable (optional)",
            info="Blank: the one next to the running python, else PATH.",
        ),
    ]

    def build(self, params):
        command = executable(params)
        data_dir = directory(params, "data_dir", "data dir", must_exist=False)
        files_root = directory(params, "local_files_root", "local files document root")
        port = positive_int(params, "port", 80)
        if port > 65535:
            raise ValueError(f"port must be <= 65535, got {port}")
        ngrok = domain(params)

        env = {
            "LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED": "true",
            "LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT": str(files_root),
        }
        notes = []
        url = f"http://localhost:{port}"
        if ngrok:
            url = f"https://{ngrok}"
            env["CSRF_TRUSTED_ORIGINS"] = url
            env["LABEL_STUDIO_HOST"] = url
            notes = [
                f"[webui] Label Studio will trust origin: {url}",
                "[webui] start ngrok in another terminal:",
                f"[webui]     ngrok http --url={ngrok} {port}",
            ]

        cmd = [command, "start", "--data-dir", str(data_dir), "-p", str(port)]
        meta = {
            "feature": self.name,
            "url": url,
            "outdir": rel(data_dir),
            "cmd": " ".join(cmd),
        }
        return JobSpec(cmd, env=env, meta=meta, notes=notes)
