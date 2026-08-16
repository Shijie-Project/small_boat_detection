"""Discovery of the configs / checkpoints that fill the dropdowns.

Every feature picks from the same two lists, so they are collected once here
and served through a single ``/api/options`` endpoint.
"""

import os
import sys

from .paths import CKPT_DIRS, CONFIG_DIR, DATA_DIRS, ROOT, SATELLITE_DIR, rel


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
SPLIT_DIRNAME = "split_images"  # must match tile_satellite.py
OUT_DIRNAME = "inf_det"  # must match torch_inf_dir.py: holds the predictions we import
# Output of the tiler and of tiled inference -- thousands of files, never inputs.
SKIP_DIRS = (SPLIT_DIRNAME, "crops")


def generated(name):
    """A folder we wrote ourselves: our own output is never an input."""
    return name in SKIP_DIRS or "_det" in name


def list_configs():
    if not CONFIG_DIR.is_dir():
        return []
    return [rel(p) for p in sorted(CONFIG_DIR.glob("*.yml"))]


def list_checkpoints():
    """The weights worth picking: a run's ``best_stg*.pth`` and the kept ``*-best.pth``.

    The second half is what inference actually runs -- the checkpoints promoted
    out of a run folder and given a name (``Dome-M-AEA-best.pth``).
    """
    out = []
    if CKPT_DIRS.is_dir():
        out.extend(
            rel(p) for p in sorted(CKPT_DIRS.rglob("*.pth")) if p.stem.startswith("best_") or p.stem.endswith("-best")
        )
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


def holds_images(path):
    """True if this folder has image files of its own."""
    try:
        return any(name.lower().endswith(IMAGE_SUFFIXES) for name in os.listdir(path))
    except OSError:
        return False


def list_tile_dirs(limit=200):
    """What inference can be pointed at: each ``split_images/<scene>/``, root first.

    The root stands for "every scene under it" -- the inference script's
    ``--all``. Our own ``inf_det/`` output is not an input, so it never shows up.
    """
    if not SATELLITE_DIR.is_dir():
        return []
    found = []
    for root, dirs, _ in os.walk(SATELLITE_DIR):
        if os.path.basename(root) == SPLIT_DIRNAME:
            scenes = [d for d in sorted(dirs) if not generated(d) and holds_images(os.path.join(root, d))]
            if scenes:
                found.append(rel(root))
                found += [rel(os.path.join(root, d)) for d in scenes]
            dirs[:] = []  # the scenes themselves hold nothing but tiles
        else:  # descend, but only into split_images once we reach it
            dirs[:] = sorted(d for d in dirs if not generated(d) or d == SPLIT_DIRNAME)
        if len(found) >= limit:
            break
    return found


def list_prediction_files(limit=200):
    """COCO results files worth importing: every inference run's, then the split ones.

    The inference runs come first because they are the ones written from the
    dashboard; ``<split>_preds.json`` is what ``train.py --test-only`` leaves in
    the annotations folder.
    """
    found = []
    if SATELLITE_DIR.is_dir():
        for root, dirs, files in os.walk(SATELLITE_DIR):
            dirs[:] = sorted(d for d in dirs if not generated(d) or d in (SPLIT_DIRNAME, OUT_DIRNAME))
            if "predictions.json" in files:
                found.append(rel(os.path.join(root, "predictions.json")))
            if len(found) >= limit:
                return found
    annotations = DATA_DIRS / "annotations"
    if annotations.is_dir():
        found += [rel(p) for p in sorted(annotations.glob("*_preds.json"))]
    return found[:limit]


def options():
    """Everything the page needs to build its forms."""
    return {
        "configs": list_configs(),
        "checkpoints": list_checkpoints(),
        "satellite": list_satellite_images(),
        "tiles": list_tile_dirs(),
        "predictions": list_prediction_files(),
        "python": sys.executable,
        "root": str(ROOT),
    }
