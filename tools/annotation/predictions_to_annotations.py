"""Turn a COCO ``predictions.json`` into Label Studio annotations, task by task.

The detector's boxes become real annotations -- the ones you open and correct --
rather than review overlays. Every task in the project is looked at; a task whose
image appears in the predictions file gets its boxes, and a task that does not
appear is left exactly as it was. That is the whole rule: the file decides which
tasks are touched, so a per-scene ``inf_det/predictions.json`` only ever affects
the tiles of that scene.

  * ``--existing skip`` (the default) never overwrites work that is already
    there: a task that already has an annotation is left alone.
  * every imported box carries an id starting with ``pred``, which is what
    ``--undo`` finds when you want them gone again.
  * ``--dry-run`` reports exactly what would happen and writes nothing.

Where the file comes from
-------------------------
``tools/inference/torch_inf_dir.py`` writes ``predictions.json`` in COCO results
format, keyed by ``image_id``. Label Studio knows images by file name, so the
ids have to be resolved back to names, from either:

  * ``detections.json`` beside the predictions file -- what the inference run
    wrote, holding the tile name and image_id of every tile, and pointing at the
    tile folder whose ``tiles.json`` gives each tile's pixel size; or
  * ``--coco <annotations.json>`` -- any COCO file whose ``images`` cover the
    ids, e.g. ``../data/annotations/val_coco.json`` for a ``val_preds.json``.

Settings come from the same environment as ``import_preannotations.py``
(``LABEL_STUDIO_URL`` / ``_API_KEY`` / ``_PROJECT_ID``, and the control names
``LABEL_STUDIO_FROM_NAME`` / ``_TO_NAME`` / ``_DATA_IMAGE_KEY``), read from the
first ``.env`` found: ``--env``, ``./.env``, then ``../data/.env``.

Usage
-----
    python tools/annotation/predictions_to_annotations.py \\
        -p ../data/satellite_images/split_images/<scene>/inf_det/predictions.json --dry-run

    # for real, labelling every box `ship`, keeping only confident ones
    python tools/annotation/predictions_to_annotations.py -p <file> --min-score 0.5

    # remove what a previous run imported
    python tools/annotation/predictions_to_annotations.py --undo
"""

import argparse
import json
import os
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import dotenv
from label_studio_sdk import LabelStudio


# Region ids start with this, so --undo can tell our boxes from hand-drawn ones.
IMPORT_TAG = "pred"
DEFAULT_IMAGE_SIZE = 1024  # fallback when nothing records a tile's real size
ENV_CANDIDATES = (".env", "../data/.env", "data/.env")


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
def load_env(explicit):
    """Read the first ``.env`` we can find; returns the path used, or ``None``."""
    for candidate in ([explicit] if explicit else []) + list(ENV_CANDIDATES):
        path = Path(candidate)
        if path.is_file():
            dotenv.load_dotenv(dotenv_path=path, override=False)
            return str(path)
    if explicit:
        raise SystemExit(f"--env: no such file: {explicit}")
    return None


def setting(name, default=""):
    return os.getenv(name, default)


# --------------------------------------------------------------------------- #
# Reading the predictions and working out which image each id is
# --------------------------------------------------------------------------- #
def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def group_predictions(results, min_score, classes):
    """``image_id -> [prediction]``, score- and class-filtered."""
    grouped = defaultdict(list)
    for pred in results:
        if "image_id" not in pred or "bbox" not in pred:
            continue
        if float(pred.get("score", 1.0)) < min_score:
            continue
        if classes and int(pred.get("category_id", -1)) not in classes:
            continue
        grouped[int(pred["image_id"])].append(pred)
    return grouped


def sizes_from_manifest(tile_dir):
    """``{tile name: (width, height)}`` from the tiler's ``tiles.json``, if it is there.

    The tile *file* is the padded canvas, which is what Label Studio displays --
    not the ``content_w/h`` of real pixels inside it, so an edge tile would be
    placed wrong if the content size were used here.
    """
    if not tile_dir:
        return {}
    path = Path(tile_dir) / "tiles.json"
    if not path.is_file():
        return {}
    manifest = load_json(path)
    return {entry["name"]: (int(entry["width"]), int(entry["height"])) for entry in manifest.get("tiles", [])}


def images_from_detections(path):
    """``image_id -> (file_name, width, height)`` from an inference run's detections.json."""
    if not path.is_file():
        return {}
    data = load_json(path)
    sizes = sizes_from_manifest(data.get("tile_dir"))
    images = {}
    for tile in data.get("tiles", []):
        if "image_id" not in tile:
            continue
        width, height = sizes.get(tile["name"], (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE))
        images[int(tile["image_id"])] = (tile["name"], width, height)
    return images


def images_from_coco(path):
    """``image_id -> (file_name, width, height)`` from a COCO annotations file."""
    data = load_json(path)
    return {
        int(image["id"]): (
            image["file_name"],
            int(image.get("width") or DEFAULT_IMAGE_SIZE),
            int(image.get("height") or DEFAULT_IMAGE_SIZE),
        )
        for image in data.get("images", [])
    }


def resolve_images(predictions_path, coco_path):
    """Where every image_id's name and size comes from; the two sources merge.

    ``--coco`` wins where both know an id: it is the file the ids were made to
    match in the first place.
    """
    images = images_from_detections(Path(predictions_path).with_name("detections.json"))
    if coco_path:
        images.update(images_from_coco(coco_path))
    if not images:
        raise SystemExit(
            f"cannot tell which image each id is: no detections.json beside {predictions_path} and no --coco given"
        )
    return images


# --------------------------------------------------------------------------- #
# Label Studio
# --------------------------------------------------------------------------- #
def connect():
    url = setting("LABEL_STUDIO_URL", "http://localhost:80")
    key = setting("LABEL_STUDIO_API_KEY")
    if not key:
        raise SystemExit("LABEL_STUDIO_API_KEY is not set (looked in the .env files listed in --help)")
    return LabelStudio(base_url=url, api_key=key)


def field(obj, key, default=None):
    """Read a field from either a dict or an SDK object."""
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def basename_of(path_or_url):
    """The image file name behind a Label Studio data value."""
    parsed = urlparse(str(path_or_url))
    query = parse_qs(parsed.query)
    if query.get("d"):
        return Path(unquote(query["d"][0])).name
    return Path(unquote(parsed.path)).name


def fetch_tasks(client, project_id):
    project = client.projects.get(id=project_id)
    tasks = list(client.tasks.list(project=project_id, fields="all"))
    print(f"project '{project.title}' (id={project_id}): {len(tasks)} task(s)")
    return project, tasks


def project_labels(project, from_name):
    """The labels the project's config offers for our control, or ``None``."""
    config = getattr(project, "parsed_label_config", None) or {}
    control = config.get(from_name) if isinstance(config, dict) else None
    return list(control.get("labels", [])) if isinstance(control, dict) else None


def task_basename(task, image_key):
    data = field(task, "data", {}) or {}
    value = data.get(image_key)
    return basename_of(value) if value else ""


# --------------------------------------------------------------------------- #
# COCO box -> Label Studio region
# --------------------------------------------------------------------------- #
def to_region(bbox, width, height, label, from_name, to_name):
    """A COCO ``[x, y, w, h]`` pixel box as an LS rectangle, in percent, clamped."""
    x, y, w, h = (float(v) for v in bbox)
    x0, y0 = min(max(x, 0.0), width), min(max(y, 0.0), height)
    x1, y1 = min(max(x + w, 0.0), width), min(max(y + h, 0.0), height)
    if x1 - x0 <= 0 or y1 - y0 <= 0:
        return None
    return {
        "id": IMPORT_TAG + uuid.uuid4().hex[:8],
        "from_name": from_name,
        "to_name": to_name,
        "type": "rectanglelabels",
        "original_width": width,
        "original_height": height,
        "image_rotation": 0,
        "value": {
            "x": 100.0 * x0 / width,
            "y": 100.0 * y0 / height,
            "width": 100.0 * (x1 - x0) / width,
            "height": 100.0 * (y1 - y0) / height,
            "rotation": 0,
            "rectanglelabels": [label],
        },
    }


def is_imported(region):
    return str(field(region, "id", "")).startswith(IMPORT_TAG)


# --------------------------------------------------------------------------- #
# Import
# --------------------------------------------------------------------------- #
def run_import(client, tasks, images, grouped, args):
    """Write one annotation per matching task. Returns the counts to report."""
    from_name = setting("LABEL_STUDIO_FROM_NAME", "label")
    to_name = setting("LABEL_STUDIO_TO_NAME", "image")
    image_key = setting("LABEL_STUDIO_DATA_IMAGE_KEY", "image")

    # the predictions are keyed by image_id; tasks are known by file name
    by_basename = {}
    for image_id, (file_name, width, height) in images.items():
        if grouped.get(image_id):
            by_basename[Path(file_name).name] = (image_id, width, height)

    counts = defaultdict(int)
    boxes = 0
    for task in tasks:
        name = task_basename(task, image_key)
        match = by_basename.pop(name, None)
        if match is None:
            counts["skipped (not in file)"] += 1
            continue

        image_id, width, height = match
        regions = [
            region
            for region in (
                to_region(pred["bbox"], width, height, args.label, from_name, to_name) for pred in grouped[image_id]
            )
            if region is not None
        ]
        if not regions:
            counts["skipped (no usable box)"] += 1
            continue

        existing = field(task, "annotations", []) or []
        task_id = int(field(task, "id"))
        if existing and args.existing == "skip":
            counts["skipped (already annotated)"] += 1
            continue

        boxes += len(regions)
        if args.dry_run:
            counts["would import"] += 1
            print(f"  {name}: {len(regions)} box(es){' (replacing)' if existing else ''}")
            continue

        if existing and args.existing == "replace":
            for annotation in existing:
                client.annotations.delete(id=int(field(annotation, "id")))
            client.annotations.create(id=task_id, result=regions)
            counts["replaced"] += 1
        elif existing:  # append: keep what is there, add ours to the first one
            target = existing[0]
            kept = list(field(target, "result", []) or [])
            client.annotations.update(id=int(field(target, "id")), result=kept + regions)
            counts["appended"] += 1
        else:
            client.annotations.create(id=task_id, result=regions)
            counts["created"] += 1

    counts["images with no task"] = len(by_basename)
    return counts, boxes, sorted(by_basename)


def run_undo(client, tasks):
    """Remove what an earlier import left: our regions, and the annotations that held only them."""
    updated = deleted = 0
    for task in tasks:
        for annotation in field(task, "annotations", []) or []:
            regions = list(field(annotation, "result", []) or [])
            kept = [region for region in regions if not is_imported(region)]
            if len(kept) == len(regions):
                continue
            annotation_id = int(field(annotation, "id"))
            if kept:
                client.annotations.update(id=annotation_id, result=kept)
                updated += 1
            else:
                client.annotations.delete(id=annotation_id)
                deleted += 1
    print(f"undo: stripped imported boxes from {updated} annotation(s), deleted {deleted} import-only annotation(s)")


# --------------------------------------------------------------------------- #
def main(args):
    env_file = load_env(args.env)
    print(f"settings from {env_file or 'the environment'}")

    project_id = args.project or int(setting("LABEL_STUDIO_PROJECT_ID", "0"))
    if project_id <= 0:
        raise SystemExit("no project: pass --project or set LABEL_STUDIO_PROJECT_ID")

    client = connect()
    project, tasks = fetch_tasks(client, project_id)

    if args.undo:
        run_undo(client, tasks)
        return

    if not args.predictions:
        raise SystemExit("-p/--predictions is required (or use --undo)")

    labels = project_labels(project, setting("LABEL_STUDIO_FROM_NAME", "label"))
    if labels and args.label not in labels:
        raise SystemExit(f"label {args.label!r} is not in the project's config; it offers: {', '.join(labels)}")

    results = load_json(args.predictions)
    grouped = group_predictions(results, args.min_score, set(args.classes or []))
    images = resolve_images(args.predictions, args.coco)
    print(
        f"{len(results)} box(es) in {os.path.basename(args.predictions)}"
        f" -> {sum(len(v) for v in grouped.values())} kept over {len(grouped)} image(s)"
        f" (min score {args.min_score}{', classes ' + str(args.classes) if args.classes else ''})"
    )

    counts, boxes, orphans = run_import(client, tasks, images, grouped, args)
    verb = "would import" if args.dry_run else "imported"
    print(f"{verb} {boxes} box(es) as `{args.label}`")
    for key in sorted(counts):
        print(f"  {key}: {counts[key]}")
    if orphans:
        print(f"  ({len(orphans)} predicted image(s) have no task in the project, e.g. {orphans[0]})")
    if args.dry_run:
        print("dry run: nothing was written")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-p", "--predictions", help="COCO results json, e.g. <scene>/inf_det/predictions.json")
    parser.add_argument(
        "--coco",
        default=None,
        help="COCO annotations whose `images` resolve the image_ids "
        "(default: the detections.json beside the predictions file)",
    )
    parser.add_argument("--label", default="ship", help="rectanglelabel to give every imported box (default: ship)")
    parser.add_argument("--min-score", type=float, default=0.0, help="drop predictions below this score")
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=None,
        help="keep only these category ids (default: every class in the file)",
    )
    parser.add_argument(
        "--existing",
        choices=["skip", "replace", "append"],
        default="skip",
        help="what to do with a task that already has an annotation (default: skip, touch nothing)",
    )
    parser.add_argument("--dry-run", action="store_true", help="report what would happen; write nothing")
    parser.add_argument("--undo", action="store_true", help="remove the boxes a previous import created")
    parser.add_argument("--project", type=int, default=0, help="project id (default: LABEL_STUDIO_PROJECT_ID)")
    parser.add_argument(
        "--env", default=None, help=f"the .env to read (default: first of {', '.join(ENV_CANDIDATES)})"
    )
    args = parser.parse_args()
    if not args.predictions and not args.undo:
        sys.exit("nothing to do: pass -p/--predictions, or --undo")
    main(args)
