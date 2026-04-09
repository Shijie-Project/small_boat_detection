import json
import os
import shutil
import warnings
from pathlib import Path


def resplit_dataset(
    ndjson_file: str | Path,
    coco_json_file: str | Path,
    image_folder: str | Path,
    output_dir: str | Path,
    output_splits: list[str],
):
    """
    Resplits a COCO dataset and its images based on an Ultralytics ndjson file.

    Args:
        ndjson_file:    Path to the .ndjson file containing new split assignments
        coco_json_file: Path to the original COCO JSON annotation file
        image_folder:   Directory containing the original images
        output_dir:     Root directory for output
        output_splits:  List of allowed split names, e.g. ["train", "val"]
    """
    ndjson_file = Path(ndjson_file)
    coco_json_file = Path(coco_json_file)
    image_folder = Path(image_folder)
    output_dir = Path(output_dir)

    # ── 1. Load COCO JSON ────────────────────────────────────────────────────
    print(f"Loading COCO annotation file: {coco_json_file}")
    with open(coco_json_file, encoding="utf-8") as f:
        data = json.load(f)

    categories = data.get("categories", [])
    info = data.get("info", {})

    # Index images by filename stem to handle .png / .jpg differences
    all_images: dict[str, dict] = {Path(img["file_name"]).stem: img for img in data.get("images", [])}

    # Index annotations by image_id for fast lookup
    ann_by_image: dict[int, list] = {}
    for ann in data.get("annotations", []):
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    print(f"Loaded images: {len(all_images)}, annotations: {sum(len(v) for v in ann_by_image.values())}")

    # ── 2. Parse ndjson and assign images to splits ──────────────────────────
    valid_splits = set(output_splits)
    splits: dict[str, list[str]] = {s: [] for s in output_splits}

    with open(ndjson_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") != "image":
                continue

            stem = Path(obj["file"]).stem
            split = obj.get("split", "train")

            if split not in valid_splits:
                raise ValueError(
                    f"Undefined split '{split}' found in ndjson file. Allowed values: {sorted(valid_splits)}"
                )
            if stem not in all_images:
                warnings.warn(f"Image '{stem}' from ndjson not found in COCO JSON, skipping")
                continue

            splits[split].append(stem)

    # ── 3. Create output directories ─────────────────────────────────────────
    for split_name, stems in splits.items():
        if stems:
            os.makedirs(output_dir / split_name, exist_ok=True)

    # ── 4. Write new JSON files and copy images ──────────────────────────────
    for split_name, stems in splits.items():
        if not stems:
            print(f"\n[{split_name}] No images found, skipping.")
            continue

        print(f"\n[{split_name}] Processing {len(stems)} images...")

        split_images = []
        split_annotations = []
        missing_images = []

        for stem in stems:
            img = all_images[stem]
            split_images.append(img)
            split_annotations.extend(ann_by_image.get(img["id"], []))

            src = image_folder / img["file_name"]
            dst = output_dir / split_name / img["file_name"]
            if src.exists():
                shutil.move(src, dst)
            else:
                missing_images.append(str(src))

        if missing_images:
            print(f"  ⚠ {len(missing_images)} image file(s) missing (first: {missing_images[0]})")

        # Write the new COCO JSON for this split
        new_coco = {
            "info": info,
            "categories": categories,
            "images": split_images,
            "annotations": split_annotations,
        }
        out_json = output_dir / f"{split_name}_coco.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(new_coco, f, indent=4, ensure_ascii=False)

        print(f"  ✓ Written {out_json} — images: {len(split_images)}, annotations: {len(split_annotations)}")


if __name__ == "__main__":
    resplit_dataset(
        ndjson_file="./annotations/ultralytics_split.ndjson",
        coco_json_file="./annotations/all_coco.json",
        image_folder="./images",
        output_dir="./split_images",
        output_splits=["train", "val"],
    )
