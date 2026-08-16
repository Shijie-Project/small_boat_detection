"""Per-image detection metrics from a COCO GT json and a COCO-format predictions json.

For each image, predictions are greedily matched to ground truths (highest score
first, best IoU first) at a given IoU threshold, producing per-image TP / FP / FN,
precision / recall / F1 and IoU statistics. Results are written to a CSV sorted so
the worst images come first, ready for failure-case analysis.

Box-level true negatives do not exist in detection; instead each image is assigned
an image-level status:
    TP-img : has GT and at least one matched prediction
    FN-img : has GT but no matched prediction
    FP-img : empty image but model predicted something
    TN-img : empty image and model stayed silent

The predictions json is the standard COCO results format
[{"image_id", "category_id", "bbox" [x,y,w,h], "score"}, ...],
e.g. the predictions.json dumped by `det_solver.val()`.

Usage:
    python tools/evaluation/per_image_metrics.py \
        --gt path/to/instances_val.json \
        --pred output/predictions.json \
        --iou-threshold 0.5 --score-threshold 0.3 \
        --output per_image_metrics.csv
"""

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np


def box_iou_xywh(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of boxes in COCO [x, y, w, h] format.

    Returns an (N, M) matrix.
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)))

    b1 = boxes1.astype(np.float64).copy()
    b2 = boxes2.astype(np.float64).copy()
    # convert to x1, y1, x2, y2
    b1[:, 2:] += b1[:, :2]
    b2[:, 2:] += b2[:, :2]

    lt = np.maximum(b1[:, None, :2], b2[None, :, :2])
    rb = np.minimum(b1[:, None, 2:], b2[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]

    area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def match_image(
    gt_boxes: np.ndarray,
    gt_cats: np.ndarray,
    pred_boxes: np.ndarray,
    pred_cats: np.ndarray,
    pred_scores: np.ndarray,
    iou_threshold: float,
    class_agnostic: bool = False,
):
    """Greedy one-to-one matching of predictions to GTs for a single image.

    Predictions are visited in descending score order; each takes the unmatched
    GT with the highest IoU >= iou_threshold (same category unless class_agnostic).

    Returns (matched_ious, n_tp, n_fp, n_fn, best_iou_per_gt).
    """
    iou = box_iou_xywh(pred_boxes, gt_boxes)
    if not class_agnostic and len(pred_boxes) and len(gt_boxes):
        iou = np.where(pred_cats[:, None] == gt_cats[None, :], iou, 0.0)

    gt_taken = np.zeros(len(gt_boxes), dtype=bool)
    matched_ious = []
    for p in np.argsort(-pred_scores):
        if len(gt_boxes) == 0:
            break
        cand = np.where(~gt_taken, iou[p], -1.0)
        g = int(np.argmax(cand))
        if cand[g] >= iou_threshold:
            gt_taken[g] = True
            matched_ious.append(float(cand[g]))

    n_tp = len(matched_ious)
    n_fp = len(pred_boxes) - n_tp
    n_fn = len(gt_boxes) - n_tp
    # localization quality regardless of threshold: best IoU any prediction
    # achieves on each GT (0 when the GT is completely missed)
    best_iou_per_gt = iou.max(axis=0) if len(pred_boxes) and len(gt_boxes) else np.zeros(len(gt_boxes))
    return matched_ious, n_tp, n_fp, n_fn, best_iou_per_gt


def main(args):
    with open(args.gt) as f:
        gt = json.load(f)
    with open(args.pred) as f:
        preds = json.load(f)

    # default: write the CSV next to the predictions json
    if args.output is None:
        args.output = os.path.join(os.path.dirname(os.path.abspath(args.pred)), "per_image_metrics.csv")

    images = {img["id"]: img for img in gt["images"]}

    gt_by_img = defaultdict(list)
    for ann in gt["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        gt_by_img[ann["image_id"]].append(ann)

    pred_by_img = defaultdict(list)
    for p in preds:
        if p["score"] >= args.score_threshold:
            pred_by_img[p["image_id"]].append(p)

    rows = []
    totals = defaultdict(float)
    for img_id, img in images.items():
        gts = gt_by_img.get(img_id, [])
        ps = pred_by_img.get(img_id, [])

        gt_boxes = np.array([g["bbox"] for g in gts]) if gts else np.zeros((0, 4))
        gt_cats = np.array([g["category_id"] for g in gts]) if gts else np.zeros(0)
        pred_boxes = np.array([p["bbox"] for p in ps]) if ps else np.zeros((0, 4))
        pred_cats = np.array([p["category_id"] for p in ps]) if ps else np.zeros(0)
        pred_scores = np.array([p["score"] for p in ps]) if ps else np.zeros(0)

        matched_ious, tp, fp, fn, best_iou_per_gt = match_image(
            gt_boxes,
            gt_cats,
            pred_boxes,
            pred_cats,
            pred_scores,
            args.iou_threshold,
            args.class_agnostic,
        )

        precision = tp / (tp + fp) if tp + fp > 0 else float("nan")
        recall = tp / (tp + fn) if tp + fn > 0 else float("nan")
        f1 = (
            2 * precision * recall / (precision + recall)
            if tp + fp > 0 and tp + fn > 0 and precision + recall > 0
            else float("nan")
        )

        if len(gts):
            image_status = "TP-img" if tp > 0 else "FN-img"
        else:
            image_status = "FP-img" if len(ps) else "TN-img"

        rows.append(
            {
                "image_id": img_id,
                "file_name": img.get("file_name", ""),
                "n_gt": len(gts),
                "n_pred": len(ps),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(precision, 4) if precision == precision else "",
                "recall": round(recall, 4) if recall == recall else "",
                "f1": round(f1, 4) if f1 == f1 else "",
                "mean_matched_iou": round(float(np.mean(matched_ious)), 4) if matched_ious else "",
                "min_matched_iou": round(float(np.min(matched_ious)), 4) if matched_ious else "",
                "mean_best_iou_per_gt": round(float(best_iou_per_gt.mean()), 4) if len(gts) else "",
                "image_status": image_status,
            }
        )

        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        totals[image_status] += 1

    # worst images first: most FN, then most FP, then lowest mean matched IoU
    rows.sort(key=lambda r: (-r["fn"], -r["fp"], r["mean_matched_iou"] or 1.0))

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print(f"Images: {len(rows)}  (score_thr={args.score_threshold}, iou_thr={args.iou_threshold})")
    print(f"  dataset TP={tp:.0f} FP={fp:.0f} FN={fn:.0f}")
    print(f"  dataset precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")
    print(
        f"  image-level: TP-img={totals['TP-img']:.0f} FN-img={totals['FN-img']:.0f} "
        f"FP-img={totals['FP-img']:.0f} TN-img={totals['TN-img']:.0f}"
    )
    print(f"Per-image metrics written to {args.output}")

    n = min(args.show_worst, len(rows))
    if n:
        print(f"\nWorst {n} images:")
        for r in rows[:n]:
            print(
                f"  {r['file_name']:<50} gt={r['n_gt']:<3} tp={r['tp']:<3} "
                f"fp={r['fp']:<3} fn={r['fn']:<3} mean_iou={r['mean_matched_iou'] or '-'}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gt", required=True, help="COCO ground-truth annotation json")
    parser.add_argument("--pred", required=True, help="COCO results-format predictions json")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for a TP match")
    parser.add_argument("--score-threshold", type=float, default=0.3, help="discard predictions below this score")
    parser.add_argument("--class-agnostic", action="store_true", help="ignore category when matching")
    parser.add_argument(
        "--output", default=None, help="output CSV path (default: per_image_metrics.csv next to --pred)"
    )
    parser.add_argument("--show-worst", type=int, default=20, help="print the N worst images")
    args = parser.parse_args()
    main(args)
