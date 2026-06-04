#!/usr/bin/env python3
"""
Convert a COCO-format detection file into Label Studio JSON import format.

Matches the project schema seen in the example export:
  - from_name="label", type="rectanglelabels"  -> fixed "ship"
  - from_name="length_range", type="choices"   -> from the COCO category name
  - from_name="wake_intensity", type="choices" -> NOT present in COCO, skipped

Label Studio bbox values (x, y, width, height) are PERCENTAGES of image size.
COCO bbox is [x, y, w, h] in absolute pixels (top-left origin).

Usage:
    python coco_to_labelstudio.py all_coco.json output_ls.json \
        --image-prefix "/data/local-files/?d=/images/all/"
"""

import argparse
import glob
import json
import os
import random
import string
from collections import defaultdict


# Image file extensions recognised when scanning --images-dir
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# Category names that represent the length_range attribute (choices)
LENGTH_RANGE_LABELS = {
    "<5m",
    "5m-8m",
    "8m-10m",
    "10m-12m",
    "12m-15m",
    "15m-20m",
    ">20m",
}
# Category names that represent wake_intensity (choices) - kept for reference
WAKE_LABELS = {"none", "weak", "moderate", "strong"}
# The fixed rectangle label
RECT_LABEL = "ship"


def gen_id(n=10):
    chars = string.ascii_letters + string.digits + "_-"
    return "".join(random.choice(chars) for _ in range(n))


def convert(coco, image_prefix):
    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    img_by_id = {im["id"]: im for im in coco["images"]}  # noqa

    # group annotations by image
    anns_by_img = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    tasks = []
    for task_idx, im in enumerate(coco["images"], start=1):
        W = im["width"]
        H = im["height"]
        results = []

        for a in anns_by_img.get(im["id"], []):
            x, y, w, h = a["bbox"]
            px = x / W * 100.0
            py = y / H * 100.0
            pw = w / W * 100.0
            ph = h / H * 100.0
            region_id = gen_id()
            cat_name = cat_id_to_name.get(a["category_id"], "")

            base_value = {
                "x": px,
                "y": py,
                "width": pw,
                "height": ph,
                "rotation": 0,
            }

            # rectangle (ship)
            rect_value = dict(base_value)
            rect_value["rectanglelabels"] = [RECT_LABEL]
            results.append(
                {
                    "original_width": W,
                    "original_height": H,
                    "image_rotation": 0,
                    "value": rect_value,
                    "id": region_id,
                    "from_name": "label",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "origin": "manual",
                }
            )

            # length_range choice (from COCO category, if it is a length label)
            if cat_name in LENGTH_RANGE_LABELS:
                ch_value = dict(base_value)
                ch_value["choices"] = [cat_name]
                results.append(
                    {
                        "original_width": W,
                        "original_height": H,
                        "image_rotation": 0,
                        "value": ch_value,
                        "id": region_id,
                        "from_name": "length_range",
                        "to_name": "image",
                        "type": "choices",
                        "origin": "manual",
                    }
                )
            elif cat_name in WAKE_LABELS:
                ch_value = dict(base_value)
                ch_value["choices"] = [cat_name]
                results.append(
                    {
                        "original_width": W,
                        "original_height": H,
                        "image_rotation": 0,
                        "value": ch_value,
                        "id": region_id,
                        "from_name": "wake_intensity",
                        "to_name": "image",
                        "type": "choices",
                        "origin": "manual",
                    }
                )

        task = {
            "data": {"image": f"{image_prefix}{im['file_name']}"},
            "annotations": [{"result": results}],
            "predictions": [],
        }
        tasks.append(task)

    return tasks


def merge_coco(json_files):
    """Merge several COCO json files into one, reassigning ids to avoid clashes.

    Categories are de-duplicated by name. Images are de-duplicated by file_name,
    so when different files carry annotations for the same image they are all
    attached to a single merged image entry. Image and annotation ids are
    renumbered so that files with overlapping ids can be combined safely.
    """
    merged = {"images": [], "annotations": [], "categories": []}
    cat_name_to_id = {}
    fname_to_img_id = {}  # file_name -> merged image id (dedupe across files)
    next_img_id = 1
    next_ann_id = 1

    for jf in json_files:
        with open(jf, encoding="utf-8") as f:
            coco = json.load(f)

        # categories: share a single id space keyed by name
        local_cat_map = {}
        for c in coco.get("categories", []):
            name = c["name"]
            if name not in cat_name_to_id:
                new_id = len(cat_name_to_id) + 1
                cat_name_to_id[name] = new_id
                merged["categories"].append({"id": new_id, "name": name})
            local_cat_map[c["id"]] = cat_name_to_id[name]

        # images: dedupe by file_name, renumber ids
        local_img_map = {}
        for im in coco.get("images", []):
            fname = im["file_name"].split("-")[-1]
            if fname in fname_to_img_id:
                local_img_map[im["id"]] = fname_to_img_id[fname]
                continue
            new_im = dict(im)
            new_im["id"] = next_img_id
            new_im["file_name"] = fname

            fname_to_img_id[fname] = next_img_id
            local_img_map[im["id"]] = next_img_id
            next_img_id += 1
            merged["images"].append(new_im)

        # annotations: renumber ids and remap image/category references
        for a in coco.get("annotations", []):
            new_a = dict(a)
            new_a["id"] = next_ann_id
            next_ann_id += 1
            new_a["image_id"] = local_img_map[a["image_id"]]
            new_a["category_id"] = local_cat_map[a["category_id"]]
            merged["annotations"].append(new_a)

    return merged


def load_coco(path):
    """Load a COCO json file, or merge every *.json found under a directory."""
    if os.path.isdir(path):
        json_files = sorted(glob.glob(os.path.join(path, "*.json")))
        if not json_files:
            raise FileNotFoundError(f"No .json files found under {path}")
        print(f"Loading {len(json_files)} json file(s) from {path}")
        return merge_coco(json_files)

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def report_missing_images(coco, images_dir):
    """Compare images on disk against those referenced by the COCO data.

    Prints images present in images_dir but missing from the COCO json (i.e. not
    yet annotated/included), as well as the reverse case for visibility.
    """
    referenced = {os.path.basename(im["file_name"]) for im in coco["images"]}

    on_disk = set()
    for root, _, files in os.walk(images_dir):
        for name in files:
            if os.path.splitext(name)[1].lower() in IMG_EXTS:
                on_disk.add(name)

    not_in_coco = sorted(on_disk - referenced)
    not_on_disk = sorted(referenced - on_disk)

    print(f"Images on disk: {len(on_disk)} | referenced by COCO: {len(referenced)}")
    print(f"Images on disk but NOT in COCO json: {len(not_in_coco)}")
    for name in not_in_coco:
        print(f"  [missing-from-coco] {name}")
    if not_on_disk:
        print(f"Images in COCO json but NOT on disk: {len(not_on_disk)}")
        for name in not_on_disk:
            print(f"  [missing-from-disk] {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("coco_json", help="input COCO json file, or a directory of COCO json files (all *.json merged)")
    ap.add_argument("output_json", help="output Label Studio json")
    ap.add_argument(
        "--image-prefix",
        default="/data/local-files/?d=/images/all/",
        help="prefix prepended to file_name to build the image URL",
    )
    ap.add_argument("--images-dir", default=None, help="directory containing all images")
    args = ap.parse_args()

    coco = load_coco(args.coco_json)

    if args.images_dir is not None:
        report_missing_images(coco, args.images_dir)

    tasks = convert(coco, args.image_prefix)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=None)

    print(f"Wrote {len(tasks)} tasks to {args.output_json}")


if __name__ == "__main__":
    main()
