"""Discovery of the configs / checkpoints that fill the dropdowns.

Every feature picks from the same two lists, so they are collected once here
and served through a single ``/api/options`` endpoint.
"""

import sys

from .paths import CKPT_DIRS, CONFIG_DIR, ROOT, rel


def list_configs():
    if not CONFIG_DIR.is_dir():
        return []
    return [rel(p) for p in sorted(CONFIG_DIR.glob("*.yml"))]


def list_checkpoints():
    out = []
    if CKPT_DIRS.is_dir():
        out.extend(rel(p) for p in sorted(CKPT_DIRS.rglob("*.pth")) if p.stem.startswith("best_"))
    return out


def options():
    """Everything the page needs to build its forms."""
    return {
        "configs": list_configs(),
        "checkpoints": list_checkpoints(),
        "python": sys.executable,
        "root": str(ROOT),
    }
