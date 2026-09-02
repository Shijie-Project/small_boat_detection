"""Turn a Label Studio JSON export into the COCO file training reads.

The other direction of ``predictions_to_annotations.py``: once the boxes in
Label Studio have been drawn and corrected, *Export -> JSON* gives a list of
tasks, and this turns that list into ``<split>_coco.json`` -- the same shape as
``../data/annotations/val_coco.json``::

    {"info": {...},
     "categories": [{"id": 3, "name": "ship"}],
     "images":      [{"width": 1024, "height": 1024, "id": 0, "file_name": "tile.png"}],
     "annotations": [{"id": 0, "image_id": 0, "category_id": 3, "segmentation": [],
                      "bbox": [x, y, w, h], "ignore": 0, "iscrowd": 0, "area": w * h}]}

What it does to get there:

  * ``data.image`` -> ``file_name``. The export holds a URL
    (``/data/local-files/?d=commercial_data%5Ctile.png``); training wants the
    bare file name, which is what the images folder is keyed by.
  * percent -> pixels. Label Studio stores every box as a percentage of the
    image; ``original_width``/``original_height`` on the region give the size to
    multiply by, and boxes are clamped to the image and rounded to whole pixels.
  * one class. Every box becomes ``--category-id`` (3, AI-TOD's *ship*, which is
    what the existing train/val files use) unless ``--per-label`` is given, in
    which case each ``rectanglelabels`` name gets an id of its own.
  * cancelled annotations are dropped, and a task annotated more than once
    contributes its most recently updated annotation only.

An image with no boxes is kept, as a background image -- ``--skip-empty`` drops
those instead.

Both export flavours are accepted: *JSON* (tasks with an ``annotations`` list)
and *JSON-MIN* (the flattened one), and ``-i`` takes as many of them as you
like -- several projects, or several rounds of the same one, become one COCO
file. They are read in the order given, and an image two of them both hold is
taken from the later one.

Usage
-----
    python tools/annotation/ls_to_coco.py -i ../data/annotations/new_ls.json

    # several exports at once, oldest first
    python tools/annotation/ls_to_coco.py -o ../data/annotations/all_coco.json \\
        -i ../data/export/project-10-at-2026-01-01.json \\
           ../data/export/project-11-at-2026-02-01.json

    # straight into the dataset: ids continue after the existing ones, and a
    # re-export of a task already in there replaces it
    python tools/annotation/ls_to_coco.py -i ../data/annotations/new_ls.json \\
        --merge ../data/annotations/all_coco.json -o ../data/annotations/all_coco.json

    # look first, write nothing
    python tools/annotation/ls_to_coco.py -i ../data/annotations/new_ls.json --dry-run

Splitting the result into train / val is ``tools/annotation/random_split_coco.py``.
"""

import argparse
import datetime
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


# AI-TOD's ship class: what train_coco.json / val_coco.json label every box.
SHIP_CATEGORY_ID = 3
SHIP_CATEGORY_NAME = "ship"
DEFAULT_SIZE = 1024  # fallback when nothing records a task's image size
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
# Label Studio prefixes uploaded files with an 8-hex-char hash + "__".
_LS_HASH_PREFIX = re.compile(r"^[0-9a-fA-F]{8}__")


# --------------------------------------------------------------------------- #
# Reading the export
# --------------------------------------------------------------------------- #
def basename_of(value):
    """The image file name behind a Label Studio data value."""
    parsed = urlparse(str(value))
    query = parse_qs(parsed.query)
    raw = unquote(query["d"][0]) if query.get("d") else unquote(parsed.path)
    name = raw.replace("\\", "/").rsplit("/", 1)[-1]
    return _LS_HASH_PREFIX.sub("", name)


def load_export(path):
    """The task list of an export, whichever way it was written."""
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        for key in ("tasks", "data", "items"):
            if isinstance(data.get(key), list):
                return data[key]
        raise ValueError(f"{path}: not a Label Studio export -- expected a list of tasks")
    if not isinstance(data, list):
        raise ValueError(f"{path}: not a Label Studio export -- expected a list of tasks")
    return data


def load_exports(paths):
    """Every export's tasks, in the order the files were given."""
    tasks = []
    for path in paths:
        found = load_export(path)
        print(f"{os.path.basename(path)}: {len(found)} task(s)")
        tasks += found
    return tasks


def task_image(task, key):
    """The task's image value: ``data.<key>``, else whatever looks like an image."""
    data = task.get("data") if isinstance(task.get("data"), dict) else task
    if isinstance(data.get(key), str):
        return data[key]
    for value in data.values():
        if isinstance(value, str) and basename_of(value).lower().endswith(IMAGE_SUFFIXES):
            return value
    return ""


def regions_of(task, annotator, use_predictions, counts):
    """The regions of one task -- its newest real annotation, or nothing.

    A JSON-MIN task has no ``annotations``: the regions sit directly under the
    labelling control's name, so the list is found by shape instead.
    """
    if "annotations" not in task:
        for value in task.values():
            if isinstance(value, list) and any(isinstance(item, dict) and "x" in item for item in value):
                return value
        return []

    annotations = [a for a in task.get("annotations") or [] if not a.get("was_cancelled")]
    if annotator:
        annotations = [a for a in annotations if str(a.get("completed_by")) == annotator]
    if not annotations:
        if use_predictions and task.get("predictions"):
            counts["tasks taken from predictions"] += 1
            return list(task["predictions"][-1].get("result") or [])
        return []
    if len(annotations) > 1:
        counts["tasks annotated more than once"] += 1
    newest = max(annotations, key=lambda a: str(a.get("updated_at") or a.get("created_at") or ""))
    return list(newest.get("result") or [])


def rectangle(region):
    """One region as a plain dict, or ``None`` when it is not a box.

    Handles both the nested (``{"value": {...}}``) and the flat JSON-MIN form.
    """
    if not isinstance(region, dict):
        return None
    value = region.get("value") if isinstance(region.get("value"), dict) else region
    if not all(key in value for key in ("x", "y", "width", "height")):
        return None
    labels = value.get("rectanglelabels") or value.get("labels") or []
    return {
        "label": str(labels[0]) if labels else "",
        "x": float(value["x"]),
        "y": float(value["y"]),
        "w": float(value["width"]),
        "h": float(value["height"]),
        "rotation": float(value.get("rotation") or 0.0),
        "image_width": int(region.get("original_width") or value.get("original_width") or 0),
        "image_height": int(region.get("original_height") or value.get("original_height") or 0),
    }


# --------------------------------------------------------------------------- #
# Label Studio region -> COCO box
# --------------------------------------------------------------------------- #
def corners(rect, width, height):
    """The box's four corners in pixels; Label Studio rotates about ``(x, y)``."""
    x = rect["x"] / 100.0 * width
    y = rect["y"] / 100.0 * height
    w = rect["w"] / 100.0 * width
    h = rect["h"] / 100.0 * height
    angle = math.radians(rect["rotation"])
    cos, sin = math.cos(angle), math.sin(angle)
    return [(x + dx * cos - dy * sin, y + dx * sin + dy * cos) for dx, dy in ((0, 0), (w, 0), (w, h), (0, h))]


def to_bbox(rect, width, height, counts):
    """A region as a whole-pixel COCO ``[x, y, w, h]``, clamped to the image.

    A box that rounds away to nothing is kept at 1 px rather than dropped: it
    was drawn on purpose, and these are 5 m boats on a 0.5 m image.
    """
    points = corners(rect, width, height)
    x0 = max(0.0, min(p[0] for p in points))
    y0 = max(0.0, min(p[1] for p in points))
    x1 = min(float(width), max(p[0] for p in points))
    y1 = min(float(height), max(p[1] for p in points))
    if x1 <= 0 or y1 <= 0 or x0 >= width or y0 >= height or x1 <= x0 or y1 <= y0:
        counts["boxes outside the image"] += 1
        return None

    x, y = round(x0), round(y0)
    w, h = round(x1 - x0), round(y1 - y0)
    if w < 1 or h < 1:
        counts["boxes rounded up to 1 px"] += 1
        w, h = max(w, 1), max(h, 1)
    x = min(x, max(width - w, 0))
    y = min(y, max(height - h, 0))
    return [float(x), float(y), float(w), float(h)]


# --------------------------------------------------------------------------- #
# Image sizes
# --------------------------------------------------------------------------- #
def size_of_file(path):
    """``(width, height)`` of an image on disk, without decoding it if we can."""
    try:
        from PIL import Image

        with Image.open(path) as image:
            return image.size
    except Exception:  # noqa: BLE001 -- no PIL, or a format it will not open
        pass
    try:
        import cv2

        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is not None:
            return image.shape[1], image.shape[0]
    except Exception:  # noqa: BLE001 -- no cv2, or an unreadable file
        pass
    return None


class Sizes:
    """Image sizes, from the regions themselves or from the images folder."""

    def __init__(self, folder, default):
        self.default = default
        self.cache = {}
        self.files = {}
        if folder:
            for path in Path(folder).rglob("*"):
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                    self.files.setdefault(path.name, path)

    def of(self, name, rects, counts):
        """The task's image size: the regions know it, else the file, else the default."""
        for rect in rects:
            if rect["image_width"] > 0 and rect["image_height"] > 0:
                return rect["image_width"], rect["image_height"]
        if name in self.cache:
            return self.cache[name]
        size = size_of_file(self.files[name]) if name in self.files else None
        if size is None:
            counts[f"images sized by the {self.default} px default"] += 1
            size = (self.default, self.default)
        self.cache[name] = size
        return size


# --------------------------------------------------------------------------- #
# Conversion
# --------------------------------------------------------------------------- #
def category_table(labels, args):
    """``{label: category id}`` plus the ``categories`` list that goes with it."""
    if not args.per_label:
        table = dict.fromkeys(labels, args.category_id)
        return table, [{"id": args.category_id, "name": args.category_name}]
    table = {label: args.first_category_id + i for i, label in enumerate(sorted(labels))}
    return table, [{"id": index, "name": label} for label, index in sorted(table.items(), key=lambda kv: kv[1])]


def convert(tasks, args, sizes, counts):
    """The exports as ``(images, label counts)``, ids not yet given.

    Keyed by file name, so a task that several exports (or one export twice
    over) hold is converted once: the last one read is the one kept.
    """
    keep = {label.strip() for label in args.labels.split(",") if label.strip()} if args.labels else None
    images, labels = {}, Counter()

    for task in tasks:
        name = basename_of(task_image(task, args.image_key))
        if not name:
            counts["tasks with no image"] += 1
            continue

        regions = regions_of(task, args.annotator, args.use_predictions, counts)
        rects = [rect for rect in (rectangle(region) for region in regions) if rect]
        if keep is not None:
            dropped = [r for r in rects if r["label"] not in keep]
            counts["boxes with another label"] += len(dropped)
            rects = [r for r in rects if r["label"] in keep]

        width, height = sizes.of(name, rects, counts)
        boxes = []
        for rect in rects:
            bbox = to_bbox(rect, width, height, counts)
            if bbox is None:
                continue
            if args.min_size and (bbox[2] < args.min_size or bbox[3] < args.min_size):
                counts[f"boxes smaller than {args.min_size} px"] += 1
                continue
            label = rect["label"] or args.category_name
            labels[label] += 1
            boxes.append((label, bbox))

        previous = images.pop(name, None)
        if previous is not None:
            counts["images the later export replaced"] += 1
            for label, _ in previous["boxes"]:
                labels[label] -= 1

        if not boxes and args.skip_empty:
            counts["empty images skipped"] += 1
            continue
        images[name] = {"name": name, "width": width, "height": height, "boxes": boxes}

    return list(images.values()), +labels  # +Counter: labels left at zero by a replacement


def assemble(images, table, categories, args, base, counts):
    """The COCO dict: ``base`` (a merged file, or empty) with ``images`` added."""
    coco_images = list(base.get("images") or [])
    coco_annotations = list(base.get("annotations") or [])

    by_name = {}
    for image in coco_images:
        by_name.setdefault(Path(str(image.get("file_name", ""))).name, image)

    next_image_id = max((int(i.get("id", -1)) for i in coco_images), default=-1) + 1
    next_ann_id = max((int(a.get("id", -1)) for a in coco_annotations), default=-1) + 1
    dropped_ids = set()

    for image in images:
        existing = by_name.get(image["name"])
        if existing is not None:
            if args.on_duplicate == "error":
                raise ValueError(f"{image['name']} is already in the merged file (--on-duplicate)")
            if args.on_duplicate == "skip":
                counts["images already in the merged file"] += 1
                continue
            counts["images replaced in the merged file"] += 1
            dropped_ids.add(existing["id"])
            coco_images.remove(existing)

        image_id = next_image_id
        next_image_id += 1
        coco_images.append(
            {"width": image["width"], "height": image["height"], "id": image_id, "file_name": image["name"]}
        )
        for label, bbox in image["boxes"]:
            coco_annotations.append(
                {
                    "id": next_ann_id,
                    "image_id": image_id,
                    "category_id": table.get(label, args.category_id),
                    "segmentation": [],
                    "bbox": bbox,
                    "ignore": 0,
                    "iscrowd": 0,
                    "area": bbox[2] * bbox[3],
                }
            )
            next_ann_id += 1

    if dropped_ids:
        coco_annotations = [a for a in coco_annotations if a.get("image_id") not in dropped_ids]

    merged_categories = list(base.get("categories") or [])
    known = {c.get("id") for c in merged_categories}
    merged_categories += [c for c in categories if c["id"] not in known]

    info = dict(base.get("info") or {})
    info.setdefault("year", datetime.date.today().year)
    info.setdefault("version", "1.0")
    info.setdefault("contributor", "Label Studio")
    info.setdefault("url", "")
    info["description"] = "converted from " + ", ".join(os.path.basename(p) for p in args.input)
    info["date_created"] = str(datetime.datetime.now())

    return {
        "info": info,
        "categories": merged_categories,
        "images": coco_images,
        "annotations": coco_annotations,
    }


def write_json(path, data):
    """Write via a temporary file: ``--merge`` may be writing over its own input."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Label Studio JSON export -> COCO annotations for training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        nargs="+",
        metavar="EXPORT",
        help="the exported json(s) (JSON or JSON-MIN); later files win on a repeated image",
    )
    parser.add_argument("-o", "--output", default="", help="where to write; default <input>_coco.json")
    parser.add_argument("--images", default="", help="images folder, for the size of tasks with no boxes")
    parser.add_argument("--image-key", default="image", help="the task data field holding the image")
    parser.add_argument("--category-id", type=int, default=SHIP_CATEGORY_ID, help="category every box gets")
    parser.add_argument("--category-name", default=SHIP_CATEGORY_NAME, help="what to call that category")
    parser.add_argument(
        "--per-label",
        action="store_true",
        help="one category per rectanglelabel instead of one for everything",
    )
    parser.add_argument("--first-category-id", type=int, default=0, help="first id given by --per-label")
    parser.add_argument("--labels", default="", help="keep only these labels, comma separated")
    parser.add_argument("--min-size", type=int, default=0, help="drop boxes narrower/shorter than this (px)")
    parser.add_argument("--skip-empty", action="store_true", help="drop images that ended up with no boxes")
    parser.add_argument("--annotator", default="", help="keep only this annotator's work (completed_by id)")
    parser.add_argument(
        "--use-predictions",
        action="store_true",
        help="fall back to a task's predictions when it has no annotation",
    )
    parser.add_argument("--merge", default="", help="add to this COCO file instead of starting empty")
    parser.add_argument(
        "--on-duplicate",
        choices=("replace", "skip", "error"),
        default="replace",
        help="an image the merged file already has",
    )
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE, help="image size to assume as a last resort")
    parser.add_argument("--dry-run", action="store_true", help="report what would be written, write nothing")
    args = parser.parse_args(argv)

    for path in args.input:
        if not Path(path).is_file():
            parser.error(f"export not found: {path}")
    if args.merge and not Path(args.merge).is_file():
        parser.error(f"file to merge into not found: {args.merge}")
    if args.images and not Path(args.images).is_dir():
        parser.error(f"images folder not found: {args.images}")
    if not args.output:
        first = Path(args.input[0])
        args.output = str(first.with_name(f"{first.stem}_coco.json"))
    return args


def main(argv=None):
    args = parse_args(argv)
    counts = Counter()

    tasks = load_exports(args.input)
    if len(args.input) > 1:
        print(f"{len(args.input)} exports: {len(tasks)} task(s)")

    sizes = Sizes(args.images, args.size)
    images, labels = convert(tasks, args, sizes, counts)
    table, categories = category_table(labels, args)

    base = {}
    if args.merge:
        with open(args.merge, encoding="utf-8") as handle:
            base = json.load(handle)
        print(
            f"merging into {os.path.basename(args.merge)}: "
            f"{len(base.get('images') or [])} image(s), {len(base.get('annotations') or [])} box(es)"
        )

    coco = assemble(images, table, categories, args, base, counts)

    boxes = sum(len(image["boxes"]) for image in images)
    empty = sum(1 for image in images if not image["boxes"])
    print(f"converted {len(images)} image(s), {boxes} box(es) ({empty} image(s) with none)")
    for label, count in sorted(labels.items(), key=lambda kv: -kv[1]):
        print(f"  {label or '(unlabelled)'}: {count} -> category {table.get(label, args.category_id)}")
    for note, count in sorted(counts.items()):
        print(f"  {note}: {count}")

    total = f"{len(coco['images'])} image(s), {len(coco['annotations'])} box(es)"
    if args.dry_run:
        print(f"[dry run] would write {total} to {args.output}")
        return 0
    write_json(args.output, coco)
    print(f"wrote {total} to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
