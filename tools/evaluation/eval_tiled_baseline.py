"""Evaluate a detector on a COCO test set using TILED inference, single-class AP.

Motivation
----------
The AITOD-pretrained baseline is an 800-input model, but our boat test images
are 1024. Instead of resizing 1024 -> 800 (which shrinks tiny boats), we slide
an 800x800 window over each 1024 image at ORIGINAL resolution (tile=pad=input=800,
overlapping, global NMS) -- exactly torch_inf_tiled.py. This feeds the model its
native 800 with no down-scaling, so small boats keep their pixels.

Class handling (class-agnostic / single-class)
----------------------------------------------
The pretrained model's label space differs from ours: it only ever fires the
"ship" class (default index 3). We keep just those predictions (via --classes),
relabel them to a single category id (--map-to, default 1), and collapse EVERY
ground-truth annotation to that same id. COCOeval then scores pure localization
("did we find the boats?") regardless of the original class taxonomy.

Outputs (in --out-dir)
----------------------
  predictions.json   COCO results format [{image_id, category_id, bbox xywh, score}]
  gt_singleclass.json  the test GT with all categories collapsed to one
  + prints AITOD-style AP/AR (per size bin) to stdout

Then, for per-image precision/recall/F1 + worst-case CSV:
  python tools/evaluation/per_image_metrics.py \
      --gt <out-dir>/gt_singleclass.json --pred <out-dir>/predictions.json \
      --class-agnostic --score-threshold 0.3

Usage
-----
python tools/evaluation/eval_tiled_baseline.py \
    -c configs/dome/Dome-M-AITOD.yml -r <aitod_ckpt.pth> \
    --gt   C:/.../data/annotations/val_coco.json \
    --img-dir C:/.../data/images/val \
    -d cuda --tile 800 --pad 800 --input_size 800 \
    --classes 3 --map-to 1 --out-dir output/aitod_baseline_tiled
"""

import argparse
import copy
import json
import os
import sys
from collections import defaultdict
from types import SimpleNamespace

from PIL import Image, ImageDraw


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
Image.MAX_IMAGE_PIXELS = None

from tools.inference.torch_inf_tiled import build_model, run_tiles


def collapse_gt(gt, cat_id, cat_name="boat"):
    """Return a copy of a COCO GT dict with every annotation forced to one class."""
    out = copy.deepcopy(gt)
    out["categories"] = [{"id": cat_id, "name": cat_name, "supercategory": cat_name}]
    for ann in out["annotations"]:
        ann["category_id"] = cat_id
    return out


def save_vis(img, boxes, scores, gt_boxes, vis_thrh, out_path):
    """Side-by-side panel: left = GT (green), right = predictions (red)."""
    W, H = img.size

    left = img.copy()
    dl = ImageDraw.Draw(left)
    for gx, gy, gw, gh in gt_boxes:  # GT in COCO xywh
        dl.rectangle([gx, gy, gx + gw, gy + gh], outline="lime", width=3)
    dl.text((5, 5), f"GT ({len(gt_boxes)})", fill="lime")

    right = img.copy()
    dr = ImageDraw.Draw(right)
    n_pred = 0
    for (x1, y1, x2, y2), sc in zip(boxes.tolist(), scores.tolist()):
        if sc < vis_thrh:
            continue
        dr.rectangle([x1, y1, x2, y2], outline="red", width=3)
        dr.text((x1, y1), f"{sc:.2f}", fill="yellow")
        n_pred += 1
    dr.text((5, 5), f"pred ({n_pred})", fill="red")

    # stitch left | right with a thin white separator
    sep = 4
    canvas = Image.new("RGB", (W * 2 + sep, H), (255, 255, 255))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (W + sep, 0))
    canvas.save(out_path, quality=90)


def main(args):  # run_tiles is already wrapped in @torch.no_grad()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.gt) as f:
        gt = json.load(f)

    model = build_model(args)

    # None = keep every predicted class (for a model whose classes are all boats);
    # otherwise keep only the requested label(s)
    keep_classes = None if args.all_classes else args.classes

    # per-tile args expected by run_tiles
    tile_args = SimpleNamespace(
        tile=args.tile,
        pad=args.pad,
        input_size=args.input_size,
        overlap=args.overlap,
        thrh=args.thrh,
        nms_iou=args.nms_iou,
        batch=args.batch,
        classes=keep_classes,
    )

    # GT boxes per image, for the left (GT) panel of the side-by-side vis
    gt_by_img = defaultdict(list)
    if args.save_vis:
        for ann in gt["annotations"]:
            if not ann.get("iscrowd", 0):
                gt_by_img[ann["image_id"]].append(ann["bbox"])

    vis_dir = os.path.join(args.out_dir, "vis")
    if args.save_vis:
        os.makedirs(vis_dir, exist_ok=True)

    predictions = []
    images = gt["images"]
    print(
        f"{len(images)} test images | tile={args.tile} pad={args.pad} "
        f"input_size={args.input_size} classes={keep_classes or 'ALL'} -> cat {args.map_to}"
    )
    for i, img_info in enumerate(images):
        path = os.path.join(args.img_dir, img_info["file_name"])
        img = Image.open(path).convert("RGB")
        boxes, scores, labels = run_tiles(model, args.device, img, tile_args, crops_dir=None)
        if args.save_vis:
            vis_name = os.path.splitext(os.path.basename(img_info["file_name"]))[0] + "_det.jpg"
            save_vis(
                img, boxes, scores, gt_by_img.get(img_info["id"], []), args.vis_thrh, os.path.join(vis_dir, vis_name)
            )
        for (x1, y1, x2, y2), sc in zip(boxes.tolist(), scores.tolist()):
            predictions.append(
                {
                    "image_id": img_info["id"],
                    "category_id": args.map_to,  # collapse to one class
                    "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],  # xyxy -> xywh
                    "score": round(sc, 4),
                }
            )
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(images)} images done, {len(predictions)} dets so far")

    pred_path = os.path.join(args.out_dir, "predictions.json")
    gt_sc_path = os.path.join(args.out_dir, "gt_singleclass.json")
    with open(pred_path, "w") as f:
        json.dump(predictions, f)
    gt_sc = collapse_gt(gt, args.map_to)
    with open(gt_sc_path, "w") as f:
        json.dump(gt_sc, f)
    print(f"Saved {pred_path} ({len(predictions)} dets)\nSaved {gt_sc_path}")

    # ---- class-agnostic COCO AP (same package the repo's evaluator uses) ----
    from faster_coco_eval_aitod import COCO, COCOeval_faster

    coco_gt = COCO(gt_sc_path)
    coco_dt = coco_gt.loadRes(pred_path) if predictions else COCO()
    E = COCOeval_faster(coco_gt, iouType="bbox", print_function=print, separate_eval=True)
    E.cocoDt = coco_dt
    E.params.imgIds = [im["id"] for im in images]
    E.evaluate()
    E.accumulate()
    E.summarize()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-c", "--config", required=True)
    p.add_argument("-r", "--resume", required=True)
    p.add_argument("--gt", required=True, help="COCO ground-truth json of the test set")
    p.add_argument("--img-dir", required=True, help="folder holding the test images")
    p.add_argument("-d", "--device", default="cuda")
    p.add_argument("--tile", type=int, default=800)
    p.add_argument("--pad", type=int, default=800)
    p.add_argument(
        "--input_size", type=int, default=800, help="must match the model's eval_spatial_size (800 for AITOD)"
    )
    p.add_argument("--overlap", type=int, default=24)
    p.add_argument(
        "--thrh", type=float, default=0.0, help="keep-all default 0 so AP integrates over all scores; raise for speed"
    )
    p.add_argument("--nms_iou", type=float, default=0.5)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--classes", type=int, nargs="+", default=[3], help="model label(s) to keep (AITOD ship = 3)")
    p.add_argument(
        "--all-classes",
        action="store_true",
        help="keep EVERY predicted class (use for your own boat model whose "
        "classes are all boat sub-types); overrides --classes",
    )
    p.add_argument("--map-to", type=int, default=1, help="single category id all kept dets + all GT are collapsed to")
    p.add_argument(
        "--save-vis",
        action="store_true",
        help="save a side-by-side image per test image: left = GT (green), right = pred (red)",
    )
    p.add_argument(
        "--vis-thrh",
        type=float,
        default=0.3,
        help="only draw predictions above this score in the visualization (does NOT affect AP)",
    )
    p.add_argument("--out-dir", default="output/tiled_baseline")
    args = p.parse_args()
    main(args)
