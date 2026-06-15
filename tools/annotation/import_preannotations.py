import json
import os
import uuid
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qs, unquote, urlparse

import dotenv
from label_studio_sdk import LabelStudio


dotenv.load_dotenv(dotenv_path=Path.cwd().joinpath(".env"))

# Label Studio config.
"""
<View>
  <style>
    .row {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 3px;
    }
  </style>

  <!-- Attribute 1 -->
  <View className="row">
    <Header value="Wake Intensity" size="5"/>
    <Choices name="wake_intensity" toName="image" perRegion="true" required="false" choice="single" showInline="true">
      <Choice value="none" hint="no visible wake"/>
      <Choice value="weak" hint="visible but not distinct"/>
      <Choice value="moderate" hint="clear but not long"/>
      <Choice value="strong" hint="very distinct and/or long"/>
    </Choices>
  </View>

  <!-- Attribute 2 -->
  <View className="row">
    <Header value="Boat Length" size="5"/>
    <Choices name="length_range" toName="image" perRegion="true" required="false" choice="single" showInline="true">
      <Choice value="&lt;5m"/>
      <Choice value="5m-8m"/>
      <Choice value="8m-10m"/>
      <Choice value="10m-12m"/>
      <Choice value="12m-15m"/>
      <Choice value="15m-20m"/>
      <Choice value="&gt;20m"/>
    </Choices>
  </View>

  <RectangleLabels name="label" toName="image" snap="pixel">
    <Label value="ship" background="red" selected="false"/>
    <Label value="ship-pred" background="green" selected="false"/>
  </RectangleLabels>

  <Image name="image" value="$image" smoothing="false" zoom="true" zoomControl="true"/>

</View>
"""

LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL", "http://localhost:80")
LABEL_STUDIO_API_KEY = os.getenv("LABEL_STUDIO_API_KEY")
LABEL_STUDIO_PROJECT_ID = int(os.getenv("LABEL_STUDIO_PROJECT_ID", "9"))


FROM_NAME = os.getenv("LABEL_STUDIO_FROM_NAME", "label")
TO_NAME = os.getenv("LABEL_STUDIO_TO_NAME", "image")
DATA_IMAGE_KEY = os.getenv("LABEL_STUDIO_DATA_IMAGE_KEY", "image")

# Prediction controls
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.25"))

ANNOTATION_ROOT_DIR = Path("../data/annotations")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def basename_from_ls_path(path_or_url: str) -> str:
    parsed = urlparse(path_or_url)
    query = parse_qs(parsed.query)

    if "d" in query and query["d"]:
        return Path(unquote(query["d"][0])).name

    return Path(unquote(parsed.path)).name


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def coco_bbox_to_ls_percent(
    bbox_xywh: list[float],
    image_width: int | float,
    image_height: int | float,
) -> dict[str, float] | None:
    """
    Convert COCO bbox [x, y, w, h] in pixels
    to Label Studio percentage bbox.
    """
    x, y, w, h = map(float, bbox_xywh)

    if w <= 0 or h <= 0:
        return None

    x1 = clamp(x, 0.0, float(image_width))
    y1 = clamp(y, 0.0, float(image_height))
    x2 = clamp(x + w, 0.0, float(image_width))
    y2 = clamp(y + h, 0.0, float(image_height))

    new_w = x2 - x1
    new_h = y2 - y1
    if new_w <= 0 or new_h <= 0:
        return None

    return {
        "x": 100.0 * x1 / float(image_width),
        "y": 100.0 * y1 / float(image_height),
        "width": 100.0 * new_w / float(image_width),
        "height": 100.0 * new_h / float(image_height),
        "rotation": 0,
    }


def build_result_item(
    label_name: str,
    image_width: int,
    image_height: int,
    bbox_xywh: list[float],
) -> dict[str, Any] | None:
    value = coco_bbox_to_ls_percent(
        bbox_xywh=bbox_xywh,
        image_width=image_width,
        image_height=image_height,
    )
    if value is None:
        return None

    value["rectanglelabels"] = [label_name]

    return {
        "id": uuid.uuid4().hex[:10],
        "from_name": FROM_NAME,
        "to_name": TO_NAME,
        "type": "rectanglelabels",
        "original_width": image_width,
        "original_height": image_height,
        "image_rotation": 0,
        "value": value,
    }


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field from either a dict or an object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def region_is_ship_pred(item: Any) -> bool:
    """True if a single annotation result region is a `ship-pred` rectangle."""
    if _get(item, "type") != "rectanglelabels":
        return False
    value = _get(item, "value", {}) or {}
    return "ship-pred" in (_get(value, "rectanglelabels", []) or [])


def strip_ship_pred_regions(result: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop previously imported `ship-pred` regions, keep everything else (e.g. manual `ship`)."""
    return [r for r in result if not region_is_ship_pred(r)]


def main(split: Literal["train", "val", "test", "all"]) -> None:
    # Input files
    coco_gt_json = Path(ANNOTATION_ROOT_DIR, f"{split}_coco.json").resolve()
    coco_pred_json = Path(ANNOTATION_ROOT_DIR, f"{split}_preds.json").resolve()

    if not LABEL_STUDIO_API_KEY:
        raise ValueError("LABEL_STUDIO_API_KEY is not set.")
    if LABEL_STUDIO_PROJECT_ID <= 0:
        raise ValueError("LABEL_STUDIO_PROJECT_ID must be set to a valid project id.")

    coco_gt = load_json(coco_gt_json)
    coco_preds = load_json(coco_pred_json)

    images = coco_gt.get("images", [])
    categories = coco_gt.get("categories", [])

    image_info_by_id: dict[int, dict[str, Any]] = {}
    for img in images:
        try:
            image_info_by_id[int(img["id"])] = {
                "file_name": img["file_name"],
                "width": int(img["width"]),
                "height": int(img["height"]),
            }
        except TypeError:
            warnings.warn(
                f"Image ID {img.get('id')} ('{img.get('file_name')}') has missing or invalid "
                f"width/height values. Defaulting to 1024x1024.",
                UserWarning,
                stacklevel=2,
            )
            image_info_by_id[int(img["id"])] = {
                "file_name": img["file_name"],
                "width": 1024,
                "height": 1024,
            }

    category_name_by_id = {int(cat["id"]): str(cat["name"]) for cat in categories}

    preds_by_image_id: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for pred in coco_preds:
        score = float(pred.get("score", 0.0))
        if score < SCORE_THRESHOLD:
            continue
        if "image_id" not in pred or "bbox" not in pred:
            continue
        preds_by_image_id[int(pred["image_id"])].append(pred)

    ls = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=LABEL_STUDIO_API_KEY)
    project = ls.projects.get(id=LABEL_STUDIO_PROJECT_ID)

    tasks = list(ls.tasks.list(project=project.id, fields="all"))
    print(f"Loaded {len(tasks)} tasks from project '{project.title}' (id={project.id})")

    # Build mapping: LS task image basename -> task
    task_by_image_basename: dict[str, Any] = {}
    for task in tasks:
        task_data = getattr(task, "data", {}) or {}
        image_ref = task_data.get(DATA_IMAGE_KEY)
        if not image_ref:
            continue
        task_by_image_basename[basename_from_ls_path(str(image_ref))] = task

    updated = 0
    created = 0
    skipped_no_task = 0
    skipped_no_preds = 0

    for image_id, image_info in image_info_by_id.items():
        file_name = image_info["file_name"]
        image_width = image_info["width"]
        image_height = image_info["height"]

        task = task_by_image_basename.get(Path(file_name).name)
        if task is None:
            skipped_no_task += 1
            continue

        preds = preds_by_image_id.get(image_id, [])
        if not preds:
            skipped_no_preds += 1
            continue

        result: list[dict[str, Any]] = []

        for pred in preds:
            category_id = int(pred["category_id"])
            label_name = category_name_by_id.get(category_id)
            if label_name is None:
                continue

            label_name = "ship-pred"  # we manually set the label name to `ship-pred` here
            item = build_result_item(
                label_name=label_name,
                image_width=image_width,
                image_height=image_height,
                bbox_xywh=pred["bbox"],
            )
            if item is None:
                continue

            result.append(item)

        if not result:
            continue

        # Merge the `ship-pred` boxes into the image's existing annotation so a
        # single annotation carries BOTH the manual `ship` boxes and the model's
        # `ship-pred` boxes (overlapping), making prediction-vs-annotation
        # differences visible in one view. On re-run, stale `ship-pred` regions
        # are stripped first and replaced with the fresh predictions, while the
        # manual `ship` boxes are preserved.
        annotations = _get(task, "annotations", []) or []
        target = annotations[0] if annotations else None

        if target is not None:
            existing = list(_get(target, "result", []) or [])
            merged = strip_ship_pred_regions(existing) + result
            ls.annotations.update(id=int(_get(target, "id")), result=merged)
            updated += 1
        else:
            # No manual annotation exists yet -> create one holding just the predictions.
            ls.annotations.create(id=task.id, result=result)
            created += 1

    print(f"Done. Updated {updated} existing annotation(s) and created {created} new one(s) with 'ship-pred' boxes.")
    print(f"Skipped (no matching LS task): {skipped_no_task}")
    print(f"Skipped (task exists but no predictions): {skipped_no_preds}")


if __name__ == "__main__":
    main(split="all")
