"""Discovery of the configs / checkpoints that fill the dropdowns.

Every feature picks from the same two lists, so they are collected once here
and served through a single ``/api/options`` endpoint.
"""

import os
import sys

from .paths import CKPT_DIRS, CONFIG_DIR, ROOT, SATELLITE_DIR, rel


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
# Output of the tiler and of tiled inference -- thousands of files, never inputs.
SKIP_DIRS = ("split_images", "crops")


def generated(name):
    """A folder we wrote ourselves: our own output is never an input."""
    return name in SKIP_DIRS or "_det" in name


def list_configs():
    if not CONFIG_DIR.is_dir():
        return []
    return [rel(p) for p in sorted(CONFIG_DIR.glob("*.yml"))]


def list_checkpoints():
    out = []
    if CKPT_DIRS.is_dir():
        out.extend(rel(p) for p in sorted(CKPT_DIRS.rglob("*.pth")) if p.stem.startswith("best_"))
    return out


def list_satellite_images(limit=200):
    """What the tiler can be pointed at: the whole folder first, then each image.

    The purchased imagery arrives nested (``zip files/<order>/<image>.tif``), so
    this walks the tree -- minus the folders we generate ourselves, which hold
    hundreds of crops and are never an input.
    """
    if not SATELLITE_DIR.is_dir():
        return []
    found = [rel(SATELLITE_DIR)]
    for root, dirs, files in os.walk(SATELLITE_DIR):
        dirs[:] = sorted(d for d in dirs if not generated(d))
        found += [rel(os.path.join(root, f)) for f in sorted(files) if f.lower().endswith(IMAGE_SUFFIXES)]
        if len(found) >= limit:
            break
    return found


def options():
    """Everything the page needs to build its forms."""
    return {
        "configs": list_configs(),
        "checkpoints": list_checkpoints(),
        "satellite": list_satellite_images(),
        "python": sys.executable,
        "root": str(ROOT),
    }
