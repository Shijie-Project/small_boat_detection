"""
Sync COCO detector predictions into a Label Studio project for hand-correction.

Predictions are imported as green ``ship-pred`` boxes that sit in the SAME
annotation as the manual red ``ship`` ground-truth (GT) boxes, so you can see
where the model and the annotator disagree and nudge the GT accordingly.

Three modes (see ``--help``):

  (default)    import every prediction above ``SCORE_THRESHOLD``.
  --add-pred   import ONLY predictions that disagree with GT, i.e. whose best
               IoU with any GT box in the same image is below ``--iou-threshold``.
  --clean      remove every imported ``ship-pred`` box and prediction object,
               leaving the GT ``ship`` boxes untouched.

Run it from a directory that contains a ``.env`` with the ``LABEL_STUDIO_*``
settings. COCO files are read from ``<ANNOTATION_ROOT_DIR>/{split}_coco.json``
(GT) and ``<ANNOTATION_ROOT_DIR>/{split}_preds.json`` (predictions).
"""

import argparse
import json
import os
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qs, unquote, urlparse

import dotenv
from label_studio_sdk import LabelStudio


# --------------------------------------------------------------------------- #
# Configuration (read once from the environment / .env)
# --------------------------------------------------------------------------- #
dotenv.load_dotenv(dotenv_path=Path.cwd().joinpath(".env"))

LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL", "http://localhost:80")
LABEL_STUDIO_API_KEY = os.getenv("LABEL_STUDIO_API_KEY")
LABEL_STUDIO_PROJECT_ID = int(os.getenv("LABEL_STUDIO_PROJECT_ID", "9"))

# Names of the controls in the Label Studio labeling config.
FROM_NAME = os.getenv("LABEL_STUDIO_FROM_NAME", "label")
TO_NAME = os.getenv("LABEL_STUDIO_TO_NAME", "image")
DATA_IMAGE_KEY = os.getenv("LABEL_STUDIO_DATA_IMAGE_KEY", "image")

# Only import predictions with score >= this.
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.25"))
# --add-pred: a prediction "disagrees" with GT when its best IoU is below this.
IOU_MATCH_THRESHOLD = float(os.getenv("IOU_MATCH_THRESHOLD", "0.5"))

ANNOTATION_ROOT_DIR = Path("../data/annotations")
DEFAULT_IMAGE_SIZE = 1024  # fallback when a COCO image lacks width/height

# RectangleLabels used in the Label Studio project. GT boxes are `ship`,
# imported predictions are `ship-pred`. Must match the labeling config XML.
GT_LABEL = "ship"
PRED_LABEL = "ship-pred"


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #
def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def coco_path(split: str, kind: Literal["coco", "preds"]) -> Path:
    """Path to the GT (`coco`) or prediction (`preds`) file for a split."""
    return Path(ANNOTATION_ROOT_DIR, f"{split}_{kind}.json").resolve()


def get_field(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field from either a dict or an SDK object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def basename_from_ls_path(path_or_url: str) -> str:
    """Extract the image file name from a Label Studio data path or URL."""
    parsed = urlparse(path_or_url)
    query = parse_qs(parsed.query)
    if query.get("d"):
        return Path(unquote(query["d"][0])).name
    return Path(unquote(parsed.path)).name


# --------------------------------------------------------------------------- #
# Geometry: IoU and COCO -> Label Studio box conversion
# --------------------------------------------------------------------------- #
def bbox_iou(a_xywh: list[float], b_xywh: list[float]) -> float:
    """IoU of two COCO ``[x, y, w, h]`` pixel boxes."""
    ax, ay, aw, ah = a_xywh
    bx, by, bw, bh = b_xywh
    inter_w = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    inter_h = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def best_iou_with_gt(pred_bbox: list[float], gt_bboxes: list[list[float]]) -> float:
    """Highest IoU between a prediction box and any GT box (0 if there are none)."""
    return max((bbox_iou(pred_bbox, g) for g in gt_bboxes), default=0.0)


def coco_box_to_region(
    bbox_xywh: list[float],
    image_width: int,
    image_height: int,
) -> dict[str, Any] | None:
    """
    Convert a COCO ``[x, y, w, h]`` pixel box into a Label Studio ``ship-pred``
    rectangle region, clamped to the image. Returns None for degenerate boxes.
    """
    x, y, w, h = map(float, bbox_xywh)
    if w <= 0 or h <= 0:
        return None

    x1 = min(max(x, 0.0), float(image_width))
    y1 = min(max(y, 0.0), float(image_height))
    x2 = min(max(x + w, 0.0), float(image_width))
    y2 = min(max(y + h, 0.0), float(image_height))
    new_w, new_h = x2 - x1, y2 - y1
    if new_w <= 0 or new_h <= 0:
        return None

    return {
        "id": uuid.uuid4().hex[:10],
        "from_name": FROM_NAME,
        "to_name": TO_NAME,
        "type": "rectanglelabels",
        "original_width": image_width,
        "original_height": image_height,
        "image_rotation": 0,
        "value": {
            "x": 100.0 * x1 / image_width,
            "y": 100.0 * y1 / image_height,
            "width": 100.0 * new_w / image_width,
            "height": 100.0 * new_h / image_height,
            "rotation": 0,
            "rectanglelabels": [PRED_LABEL],
        },
    }


# --------------------------------------------------------------------------- #
# `ship-pred` region bookkeeping inside an annotation's result list
# --------------------------------------------------------------------------- #
def is_pred_region(region: Any) -> bool:
    """True if a single annotation result region is a ``ship-pred`` rectangle."""
    if get_field(region, "type") != "rectanglelabels":
        return False
    value = get_field(region, "value", {}) or {}
    return PRED_LABEL in (get_field(value, "rectanglelabels", []) or [])


def without_pred_regions(regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop imported ``ship-pred`` regions, keep everything else (e.g. GT ``ship``)."""
    return [r for r in regions if not is_pred_region(r)]


# --------------------------------------------------------------------------- #
# COCO loaders (build the per-image lookups used during the sync)
# --------------------------------------------------------------------------- #
def index_image_info(coco_gt: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """image_id -> {file_name, width, height} from the GT file."""
    info: dict[int, dict[str, Any]] = {}
    for img in coco_gt.get("images", []):
        info[int(img["id"])] = {
            "file_name": img["file_name"],
            "width": int(img.get("width") or DEFAULT_IMAGE_SIZE),
            "height": int(img.get("height") or DEFAULT_IMAGE_SIZE),
        }
    return info


def index_gt_boxes(coco_gt: dict[str, Any]) -> dict[int, list[list[float]]]:
    """image_id -> list of GT ``[x, y, w, h]`` boxes (for IoU matching)."""
    boxes: dict[int, list[list[float]]] = defaultdict(list)
    for ann in coco_gt.get("annotations", []):
        if "image_id" in ann and "bbox" in ann:
            boxes[int(ann["image_id"])].append([float(v) for v in ann["bbox"]])
    return boxes


def index_predictions(
    coco_preds: list[dict[str, Any]],
    score_threshold: float,
) -> dict[int, list[dict[str, Any]]]:
    """image_id -> list of predictions scoring at or above ``score_threshold``."""
    preds: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for pred in coco_preds:
        if float(pred.get("score", 0.0)) < score_threshold:
            continue
        if "image_id" in pred and "bbox" in pred:
            preds[int(pred["image_id"])].append(pred)
    return preds


def select_pred_regions(
    preds: list[dict[str, Any]],
    gt_boxes: list[list[float]],
    image_width: int,
    image_height: int,
    iou_threshold: float | None,
) -> list[dict[str, Any]]:
    """
    Turn one image's predictions into ``ship-pred`` regions.

    When ``iou_threshold`` is set (--add-pred), keep only predictions that
    disagree with GT (best IoU below the threshold); otherwise keep all.
    """
    regions: list[dict[str, Any]] = []
    for pred in preds:
        if iou_threshold is not None and best_iou_with_gt(pred["bbox"], gt_boxes) >= iou_threshold:
            continue
        region = coco_box_to_region(pred["bbox"], image_width, image_height)
        if region is not None:
            regions.append(region)
    return regions


# --------------------------------------------------------------------------- #
# Label Studio access
# --------------------------------------------------------------------------- #
def connect_label_studio() -> LabelStudio:
    """Create a Label Studio client, validating the required settings."""
    if not LABEL_STUDIO_API_KEY:
        raise ValueError("LABEL_STUDIO_API_KEY is not set.")
    if LABEL_STUDIO_PROJECT_ID <= 0:
        raise ValueError("LABEL_STUDIO_PROJECT_ID must be set to a valid project id.")
    return LabelStudio(base_url=LABEL_STUDIO_URL, api_key=LABEL_STUDIO_API_KEY)


def fetch_tasks(ls: LabelStudio) -> list[Any]:
    """Load every task (with annotations/predictions) for the configured project."""
    project = ls.projects.get(id=LABEL_STUDIO_PROJECT_ID)
    tasks = list(ls.tasks.list(project=project.id, fields="all"))
    print(f"Loaded {len(tasks)} tasks from project '{project.title}' (id={project.id})")
    return tasks


def index_tasks_by_basename(tasks: list[Any]) -> dict[str, Any]:
    """image file name -> task, so COCO images can be matched to LS tasks."""
    by_basename: dict[str, Any] = {}
    for task in tasks:
        data = getattr(task, "data", {}) or {}
        image_ref = data.get(DATA_IMAGE_KEY)
        if image_ref:
            by_basename[basename_from_ls_path(str(image_ref))] = task
    return by_basename


def sync_pred_regions(ls: LabelStudio, task: Any, regions: list[dict[str, Any]]) -> str:
    """
    Make the task's first annotation hold exactly ``regions`` as its ``ship-pred``
    boxes: strip any previously imported ``ship-pred``, then add the new ones,
    always keeping the GT ``ship`` boxes.

    Passing ``regions=[]`` is valid and the key reason this iterates tasks rather
    than COCO images: a task NOT covered by the current split (or whose
    predictions now all agree with GT) gets its stale ``ship-pred`` cleared. An
    annotation left with nothing is deleted; tasks that need no change cost no
    API call.

    Returns one of "created", "updated", "cleared", "unchanged".
    """
    annotations = get_field(task, "annotations", []) or []
    target = annotations[0] if annotations else None

    if target is None:
        if not regions:
            return "unchanged"  # no annotation and nothing to add
        ls.annotations.create(id=int(get_field(task, "id")), result=regions)
        return "created"

    existing = list(get_field(target, "result", []) or [])
    if not regions and not any(is_pred_region(r) for r in existing):
        return "unchanged"  # nothing to add and no stale ship-pred to remove

    ann_id = int(get_field(target, "id"))
    merged = without_pred_regions(existing) + regions
    if merged:
        ls.annotations.update(id=ann_id, result=merged)
    else:
        ls.annotations.delete(id=ann_id)  # annotation only ever held predictions
    return "updated" if regions else "cleared"


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def import_predictions(
    split: Literal["train", "val", "test", "all"],
    iou_threshold: float | None = None,
) -> None:
    """
    Import predictions as ``ship-pred`` boxes alongside the GT.

    ``iou_threshold=None`` imports every prediction above ``SCORE_THRESHOLD``.
    A value (the --add-pred mode) keeps only predictions that disagree with GT.
    """
    coco_gt = load_json(coco_path(split, "coco"))
    coco_preds = load_json(coco_path(split, "preds"))

    image_info = index_image_info(coco_gt)
    gt_boxes_by_image = index_gt_boxes(coco_gt)
    preds_by_image = index_predictions(coco_preds, SCORE_THRESHOLD)
    image_id_by_basename = {Path(info["file_name"]).name: image_id for image_id, info in image_info.items()}

    ls = connect_label_studio()
    task_by_basename = index_tasks_by_basename(fetch_tasks(ls))

    # Iterate over LS tasks (not COCO images): tasks the current split doesn't
    # cover still need their stale `ship-pred` from a previous import cleared.
    counts = {"created": 0, "updated": 0, "cleared": 0, "unchanged": 0}
    kept_regions = 0

    for basename, task in task_by_basename.items():
        image_id = image_id_by_basename.get(basename)
        if image_id is None:
            regions: list[dict[str, Any]] = []  # task not in this split -> clear stale only
        else:
            info = image_info[image_id]
            regions = select_pred_regions(
                preds=preds_by_image.get(image_id, []),
                gt_boxes=gt_boxes_by_image.get(image_id, []),
                image_width=info["width"],
                image_height=info["height"],
                iou_threshold=iou_threshold,
            )

        counts[sync_pred_regions(ls, task, regions)] += 1
        kept_regions += len(regions)

    images_without_task = sum(
        1 for info in image_info.values() if Path(info["file_name"]).name not in task_by_basename
    )

    print(
        f"Done. created={counts['created']} updated={counts['updated']} "
        f"cleared={counts['cleared']} unchanged={counts['unchanged']}."
    )
    print(f"Imported {kept_regions} '{PRED_LABEL}' box(es) total.")
    if iou_threshold is not None:
        print(f"(--add-pred: kept only predictions with best IoU < {iou_threshold} vs GT.)")
    print(f"Split images with no matching LS task: {images_without_task}")


def clean() -> None:
    """
    Reset the project to GT only.

    Removes every imported ``ship-pred`` region and all Label Studio prediction
    objects, preserving the manual ``ship`` boxes and their per-region
    attributes. An annotation left empty (it only held predictions) is deleted.
    """
    ls = connect_label_studio()
    tasks = fetch_tasks(ls)

    annotations_updated = annotations_deleted = predictions_deleted = 0

    for task in tasks:
        for ann in get_field(task, "annotations", []) or []:
            existing = list(get_field(ann, "result", []) or [])
            kept = without_pred_regions(existing)
            if len(kept) == len(existing):
                continue  # nothing to strip

            ann_id = int(get_field(ann, "id"))
            if kept:
                ls.annotations.update(id=ann_id, result=kept)
                annotations_updated += 1
            else:
                ls.annotations.delete(id=ann_id)
                annotations_deleted += 1

        for pred in get_field(task, "predictions", []) or []:
            pred_id = get_field(pred, "id")
            if pred_id is not None:
                ls.predictions.delete(id=int(pred_id))
                predictions_deleted += 1

    print(
        f"Clean done. Stripped '{PRED_LABEL}' from {annotations_updated} annotation(s), "
        f"deleted {annotations_deleted} prediction-only annotation(s) and "
        f"{predictions_deleted} prediction object(s). GT '{GT_LABEL}' boxes preserved."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="all",
        help="Which split's COCO files to use (ignored with --clean). Default: all.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help=f"Delete every '{PRED_LABEL}' region and prediction object, keeping only GT '{GT_LABEL}' boxes.",
    )
    parser.add_argument(
        "--add-pred",
        action="store_true",
        help="Import only predictions that disagree with GT (best IoU < --iou-threshold).",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=IOU_MATCH_THRESHOLD,
        help=f"IoU below which a prediction counts as disagreeing with GT (--add-pred only). "
        f"Default: {IOU_MATCH_THRESHOLD}.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.clean:
        clean()
    elif args.add_pred:
        import_predictions(split=args.split, iou_threshold=args.iou_threshold)
    else:
        import_predictions(split=args.split)
