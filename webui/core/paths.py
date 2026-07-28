"""Filesystem layout shared by every feature.

The package lives in ``<project>/webui``, so the project root -- the directory
every job has to run from and every path in the UI is relative to -- is the
parent of the package.
"""

import os
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent.parent
ROOT = PACKAGE_DIR.parent
STATIC_DIR = PACKAGE_DIR / "static"

CONFIG_DIR = ROOT / "configs" / "dome"
CKPT_DIRS = ROOT.parent / "ckpts"
DATA_DIRS = ROOT.parent / "data"
TRAIN_SCRIPT = "train.py"

HOST = os.environ.get("WEBUI_HOST", "127.0.0.1")
PORT = int(os.environ.get("WEBUI_PORT", "8000"))


def rel(path) -> str:
    """Path relative to the project root, with forward slashes."""
    path = Path(path)
    try:
        path = path.resolve().relative_to(ROOT)
    except ValueError:
        pass
    return str(path).replace("\\", "/")


def resolve(rel_path):
    """Resolve a browser-supplied relative path inside the project root.

    Returns ``None`` when the path escapes the root; the page only ever sends
    back paths we listed ourselves, so anything else is a bug or a probe.
    """
    if not rel_path:
        return None
    candidate = Path(rel_path)
    if candidate.is_absolute():
        return None
    resolved = (ROOT / candidate).resolve()
    if resolved != ROOT and ROOT not in resolved.parents:
        return None
    return resolved
