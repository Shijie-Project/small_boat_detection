"""Inference over a folder of tiles that ``tile_satellite.py`` already cut.

The other two scripts sit at the ends of the range: ``torch_inf.py`` takes one
image (or a folder, one forward pass at a time, no stitching), and
``torch_inf_tiled.py`` takes a whole satellite scene and does the cutting
itself. This one is for the workflow the project actually uses -- cut the scene
once with ``tools/dataset/tile_satellite.py``, look at the tiles, annotate some
of them, and then run the detector over the folder that came out.

What it does with a tile folder
-------------------------------
  * every tile goes through the net at its ORIGINAL resolution: a 1024 tile
    with the config's 1024 eval size is fed as-is, and a tile smaller than that
    (a hand-drawn region, say) is pasted top-left into a black canvas rather
    than upscaled, exactly like the tiler pads its own edge tiles
  * tiles are batched (``--batch``), so a 42-tile scene is a handful of forward
    passes instead of 42
  * boxes whose centre falls in the black padding are dropped, the rest are
    clamped to the real content
  * ``tiles.json`` -- the manifest the tiler writes next to the tiles -- gives
    every tile's origin in the source image, so the per-tile boxes are shifted
    back into full-image coordinates, boats cut in half at a tile border are
    stitched back together, and duplicates from overlapping tiles are removed
    by a class-wise NMS. Without a manifest the tiles are still processed, just
    not merged.

Outputs, in ``<tile folder>/inf_det/`` unless ``-o`` says otherwise:

  predictions.json  the per-tile boxes in COCO results format, the same shape
                    ``det_solver.val()`` dumps, so `per_image_metrics.py` and
                    `import_preannotations.py` take it as they are. Pass
                    ``--coco <annotations>`` and a tile that is in there keeps
                    its ``image_id``, which is what makes the two comparable.
  detections.json   per-tile boxes AND (with a manifest) the merged
                    full-image boxes, with the source path and size
  detections.csv    one row per detection, full-image coords when merged
  tiles/*_det.jpg   the tiles that had a detection, boxes drawn (--save-tiles)
  <stem>_overlay.jpg  the whole scene with the merged boxes on it (--overlay)

Usage
-----
    # one scene folder
    python tools/inference/torch_inf_dir.py \
        -c configs/dome/Dome-M-AEA.yml -r ../ckpts/Dome-M-AEA-best.pth \
        -i ../data/satellite_images/split_images/<scene> -d cuda

    # every scene under split_images/
    python tools/inference/torch_inf_dir.py -c ... -r ... \
        -i ../data/satellite_images/split_images --all

    # no -i at all: pick the folder from a numbered list
    python tools/inference/torch_inf_dir.py -c ... -r ...
"""

import argparse
import csv
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw
from torchvision.ops import batched_nms


# a satellite scene redrawn for the overlay exceeds PIL's decompression guard
Image.MAX_IMAGE_PIXELS = None

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.core import YAMLConfig


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
MANIFEST = "tiles.json"
OUT_DIRNAME = "inf_det"  # holds "_det", so a re-run of the tiler skips it
DEFAULT_SPLIT_ROOT = "../data/satellite_images/split_images"


# --------------------------------------------------------------------------- #
# Model -- loaded the same way torch_inf.py does it.
# --------------------------------------------------------------------------- #
def build_model(config, resume, device):
    """``(model, input_size)``; the size comes from the config's eval_spatial_size."""
    cfg = YAMLConfig(config, resume=resume)

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    checkpoint = torch.load(resume, map_location="cpu")
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            return self.postprocessor(self.model(images), orig_target_sizes)

    eval_size = cfg.yaml_cfg.get("eval_spatial_size") or [800, 800]
    return Model().to(device).eval(), int(max(eval_size))


# --------------------------------------------------------------------------- #
# Finding the work: which folders hold tiles, and what the manifest says.
# --------------------------------------------------------------------------- #
def tiles_in(folder):
    """The tile file names in a folder, sorted; empty if it is not a tile folder."""
    if not os.path.isdir(folder):
        return []
    return sorted(name for name in os.listdir(folder) if name.lower().endswith(IMAGE_SUFFIXES) and "_det" not in name)


def find_scenes(root):
    """``[(name, folder)]`` -- the folder itself if it holds tiles, else its sub-folders.

    ``split_images/`` is a folder of scene folders, so pointing at either level
    does the obvious thing.
    """
    root = os.path.normpath(os.path.abspath(root))
    if tiles_in(root):
        return [(os.path.basename(root), root)]
    scenes = []
    for name in sorted(os.listdir(root)):
        sub = os.path.join(root, name)
        if name != OUT_DIRNAME and tiles_in(sub):
            scenes.append((name, sub))
    return scenes


def choose_scene(root):
    """Ask which scene to run, when the command line did not say."""
    root = os.path.normpath(os.path.abspath(root))
    scenes = find_scenes(root)
    if not scenes:
        raise SystemExit(f"no tile folders under {root} -- cut a scene first with tools/dataset/tile_satellite.py")
    print(f"tile folders under {root}:\n")
    for index, (name, folder) in enumerate(scenes, 1):
        print(f"  [{index:2d}] {name}  ({len(tiles_in(folder))} tiles)")
    print("  [ 0] all of them\n")
    answer = input("which one? ").strip()
    if not answer.isdigit() or int(answer) > len(scenes):
        raise SystemExit("nothing selected")
    index = int(answer)
    return scenes if index == 0 else [scenes[index - 1]]


def load_manifest(folder):
    """``tiles.json`` as ``(manifest, {tile name: entry})``, or ``(None, {})``."""
    path = os.path.join(folder, MANIFEST)
    if not os.path.isfile(path):
        return None, {}
    with open(path) as handle:
        manifest = json.load(handle)
    return manifest, {entry["name"]: entry for entry in manifest.get("tiles", [])}


# --------------------------------------------------------------------------- #
# Inference.
# --------------------------------------------------------------------------- #
def prepare(im, input_size):
    """``(canvas, canvas size, content w, content h)`` for one tile.

    A tile at or below the net's input size keeps every pixel where it is and
    the remainder is black padding -- the same canvas the tiler builds for its
    own edge tiles. A larger tile (``--whole-regions``) has to be resized down,
    which shrinks the boats; the caller warns about it.
    """
    w, h = im.size
    if w <= input_size and h <= input_size:
        if (w, h) == (input_size, input_size):
            return im, input_size, w, h
        canvas = Image.new("RGB", (input_size, input_size), (0, 0, 0))
        canvas.paste(im, (0, 0))
        return canvas, input_size, w, h
    return im, max(w, h), w, h


def filter_dets(boxes, scores, labels, thrh, classes, content_w, content_h):
    """Score / class / padding filter, boxes clamped to the real content."""
    keep = scores > thrh
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    if classes is not None and boxes.numel():
        wanted = torch.zeros_like(labels, dtype=torch.bool)
        for c in classes:
            wanted |= labels == c
        boxes, scores, labels = boxes[wanted], scores[wanted], labels[wanted]

    if boxes.numel():  # a box centred in the padding is not a boat
        cx = (boxes[:, 0] + boxes[:, 2]) / 2
        cy = (boxes[:, 1] + boxes[:, 3]) / 2
        inside = (cx < content_w) & (cy < content_h)
        boxes, scores, labels = boxes[inside], scores[inside], labels[inside]

    if boxes.numel():
        boxes = boxes.clone()
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, content_w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, content_h)

    return boxes.cpu(), scores.cpu(), labels.cpu()


def draw_boxes(im, dets, names, width=2):
    """Boxes and ``label score`` text on a copy of the tile."""
    canvas = im.copy()
    pen = ImageDraw.Draw(canvas)
    for det in dets:
        box = det["bbox_xyxy"]
        pen.rectangle(box, outline="red", width=width)
        pen.text(
            (box[0], max(0, box[1] - 10)), f"{names.get(det['label'], det['label'])} {det['score']:.2f}", fill="yellow"
        )
    return canvas


@torch.no_grad()
def run_scene(model, input_size, folder, out_dir, args, names):
    """Detect on every tile in ``folder``. Returns the per-tile records."""
    tile_names = tiles_in(folder)
    _, entries = load_manifest(folder)
    to_tensor = T.Compose([T.Resize((input_size, input_size)), T.ToTensor()])
    tiles_dir = os.path.join(out_dir, "tiles")
    if args.save_tiles != "none":
        os.makedirs(tiles_dir, exist_ok=True)

    if args.max_tiles:
        tile_names = tile_names[: args.max_tiles]

    records = []
    batch, meta = [], []
    warned_big = False
    started = time.time()

    def flush():
        if not batch:
            return
        images = torch.cat(batch, 0)
        sizes = torch.tensor([[s, s] for _, _, s, _, _ in meta], device=args.device, dtype=torch.float32)
        labels, boxes, scores = model(images, sizes)
        for i, (name, im, _, content_w, content_h) in enumerate(meta):
            b, s, l = filter_dets(boxes[i], scores[i], labels[i], args.thrh, args.classes, content_w, content_h)
            dets = [
                {"bbox_xyxy": [round(v, 1) for v in box], "score": round(score, 4), "label": int(label)}
                for box, score, label in zip(b.tolist(), s.tolist(), l.tolist())
            ]
            entry = entries.get(name, {})
            records.append(
                {
                    "name": name,
                    "x": entry.get("x"),
                    "y": entry.get("y"),
                    "content_w": content_w,
                    "content_h": content_h,
                    "detections": dets,
                }
            )
            if args.save_tiles == "all" or (args.save_tiles == "hits" and dets):
                stem = os.path.splitext(name)[0]
                draw_boxes(im, dets, names).save(os.path.join(tiles_dir, f"{stem}_det.jpg"), quality=90)
        batch.clear()
        meta.clear()

    print(f"  {len(tile_names)} tiles, input {input_size}, batch {args.batch}")
    for index, name in enumerate(tile_names, 1):
        im = Image.open(os.path.join(folder, name)).convert("RGB")
        canvas, canvas_size, content_w, content_h = prepare(im, input_size)
        if canvas_size > input_size and not warned_big:
            print(f"    note: tiles larger than {input_size} px are resized down (first: {name}, {im.size})")
            warned_big = True
        # the manifest knows how much of a padded edge tile is real content
        entry = entries.get(name)
        if entry:
            content_w = min(content_w, entry.get("content_w", content_w))
            content_h = min(content_h, entry.get("content_h", content_h))

        batch.append(to_tensor(canvas).unsqueeze(0).to(args.device))
        meta.append((name, canvas, canvas_size, content_w, content_h))
        if len(batch) >= args.batch:
            flush()
        if index % 25 == 0 or index == len(tile_names):
            print(f"    {index}/{len(tile_names)} tiles")
    flush()

    total = sum(len(record["detections"]) for record in records)
    print(f"  {total} raw detections in {time.time() - started:.1f}s")
    return records


# --------------------------------------------------------------------------- #
# Merging the tiles back into the full image.
# --------------------------------------------------------------------------- #
def to_global(records):
    """Per-tile boxes shifted into full-image coordinates.

    Tiles without an origin (no manifest entry) are skipped -- there is nowhere
    to put them.
    """
    boxes, scores, labels, sources = [], [], [], []
    for record in records:
        if record["x"] is None:
            continue
        for det in record["detections"]:
            x0, y0, x1, y1 = det["bbox_xyxy"]
            boxes.append([x0 + record["x"], y0 + record["y"], x1 + record["x"], y1 + record["y"]])
            scores.append(det["score"])
            labels.append(det["label"])
            sources.append(record["name"])
    if not boxes:
        return torch.zeros(0, 4), torch.zeros(0), torch.zeros(0, dtype=torch.int64), []
    return (
        torch.tensor(boxes, dtype=torch.float32),
        torch.tensor(scores, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.int64),
        sources,
    )


def border_lines(records):
    """The x and y coordinates where two tiles meet, in full-image coords."""
    xs, ys = set(), set()
    for record in records:
        if record["x"] is None:
            continue
        xs.update((record["x"], record["x"] + record["content_w"]))
        ys.update((record["y"], record["y"] + record["content_h"]))
    return xs, ys


def stitch_borders(boxes, scores, labels, records, tol):
    """Union the two halves of a boat that a tile border cut in two.

    Only boxes that actually run into a shared tile edge are considered -- a
    box ending on the line and another of the same class starting on it,
    overlapping along the line. Tiles cut with an overlap do not need this (the
    whole boat is in the neighbour and NMS keeps the better copy), but the
    default cut has ``--overlap 0``, where nothing else recovers the boat.
    """
    if len(boxes) < 2 or tol < 0:
        return boxes, scores, labels

    xs, ys = border_lines(records)

    def near(value, lines):
        return any(abs(value - line) <= tol for line in lines)

    # only boxes touching a border can have been cut by one
    touching = [
        i
        for i in range(len(boxes))
        if near(boxes[i, 0].item(), xs)
        or near(boxes[i, 2].item(), xs)
        or near(boxes[i, 1].item(), ys)
        or near(boxes[i, 3].item(), ys)
    ]

    parent = list(range(len(boxes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    for a_pos, i in enumerate(touching):
        for j in touching[a_pos + 1 :]:
            if labels[i] != labels[j]:
                continue
            ax0, ay0, ax1, ay1 = boxes[i].tolist()
            bx0, by0, bx1, by1 = boxes[j].tolist()
            # left|right across a vertical border, overlapping in y
            side_by_side = (abs(ax1 - bx0) <= tol or abs(bx1 - ax0) <= tol) and min(ay1, by1) - max(ay0, by0) > 0
            # top|bottom across a horizontal border, overlapping in x
            stacked = (abs(ay1 - by0) <= tol or abs(by1 - ay0) <= tol) and min(ax1, bx1) - max(ax0, bx0) > 0
            if side_by_side or stacked:
                union(i, j)

    groups = {}
    for i in range(len(boxes)):
        groups.setdefault(find(i), []).append(i)
    if len(groups) == len(boxes):
        return boxes, scores, labels

    merged_boxes, merged_scores, merged_labels = [], [], []
    for members in groups.values():
        part = boxes[members]
        merged_boxes.append(
            [
                part[:, 0].min().item(),
                part[:, 1].min().item(),
                part[:, 2].max().item(),
                part[:, 3].max().item(),
            ]
        )
        best = int(scores[members].argmax())
        merged_scores.append(scores[members][best].item())
        merged_labels.append(int(labels[members][best]))
    print(f"  stitched {len(boxes)} -> {len(merged_boxes)} across tile borders")
    return (
        torch.tensor(merged_boxes, dtype=torch.float32),
        torch.tensor(merged_scores, dtype=torch.float32),
        torch.tensor(merged_labels, dtype=torch.int64),
    )


def merge(records, args):
    """Full-image detections: shifted, stitched at the borders, de-duplicated."""
    boxes, scores, labels, _ = to_global(records)
    if not len(boxes):
        return []

    if args.edge_tol >= 0:
        boxes, scores, labels = stitch_borders(boxes, scores, labels, records, args.edge_tol)

    keep = batched_nms(boxes, scores, labels, args.nms_iou)
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    order = scores.argsort(descending=True)
    print(f"  {len(keep)} detections after NMS (IoU {args.nms_iou})")
    return [
        {
            "bbox_xyxy": [round(v, 1) for v in boxes[i].tolist()],
            "score": round(scores[i].item(), 4),
            "label": int(labels[i]),
        }
        for i in order.tolist()
    ]


# --------------------------------------------------------------------------- #
# Writing the results out.
# --------------------------------------------------------------------------- #
def write_json(path, folder, manifest, records, merged, args, names):
    payload = {
        "tile_dir": folder.replace("\\", "/"),
        "config": args.config,
        "checkpoint": args.resume,
        "score_threshold": args.thrh,
        "classes": args.classes,
        "class_names": {str(k): v for k, v in names.items()},
        "nms_iou": args.nms_iou,
        "n_tiles": len(records),
        "n_detections": len(merged) if merged else sum(len(r["detections"]) for r in records),
        "tiles": records,
    }
    if manifest:
        payload["source"] = manifest.get("source")
        payload["source_width"] = manifest.get("width")
        payload["source_height"] = manifest.get("height")
        payload["detections"] = merged
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=1)


def write_csv(path, records, merged, names):
    """Merged detections when there are any, otherwise the per-tile ones."""
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        if merged:
            writer.writerow(["x0", "y0", "x1", "y1", "score", "label", "class"])
            for det in merged:
                writer.writerow(det["bbox_xyxy"] + [det["score"], det["label"], names.get(det["label"], "")])
        else:
            writer.writerow(["tile", "x0", "y0", "x1", "y1", "score", "label", "class"])
            for record in records:
                for det in record["detections"]:
                    writer.writerow(
                        [record["name"]] + det["bbox_xyxy"] + [det["score"], det["label"], names.get(det["label"], "")]
                    )


def write_overlay(path, manifest, merged, mode, max_side, names):
    """The whole scene with the merged boxes drawn on it."""
    source = manifest.get("source") if manifest else None
    if not source or not os.path.isfile(source):
        print(f"  no source image to draw on ({source}) -- overlay skipped")
        return

    image = Image.open(source).convert("RGB")
    scale = 1.0
    if mode == "preview" and max(image.size) > max_side:
        scale = max_side / max(image.size)
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.BILINEAR)

    pen = ImageDraw.Draw(image)
    for det in merged:
        x0, y0, x1, y1 = (v * scale for v in det["bbox_xyxy"])
        # a 6 px boat in a /4 preview is 1.5 px: draw a box big enough to see
        pad = max(0.0, 6 - (x1 - x0)) / 2
        pen.rectangle([x0 - pad, y0 - pad, x1 + pad, y1 + pad], outline="red", width=2)
    image.save(path, quality=90)
    print(f"  overlay -> {path} ({'full res' if scale == 1.0 else f'{scale:.3f}x'}, {len(merged)} boxes)")


def load_coco(path):
    """``(names, image ids, next free id)`` from a COCO json; empty when unused.

    The ids matter: a ``predictions.json`` is only comparable with the ground
    truth if a tile carries the same ``image_id`` there as in the annotations,
    which is what ``per_image_metrics.py`` and ``import_preannotations.py``
    join on. Tiles the file does not mention are numbered after its last one.
    """
    if not path:
        return {}, {}, 1
    with open(path) as handle:
        data = json.load(handle)
    names = {int(c["id"]): c["name"] for c in data.get("categories", [])}
    ids = {image["file_name"]: int(image["id"]) for image in data.get("images", [])}
    return names, ids, (max(ids.values()) + 1 if ids else 1)


def assign_image_ids(records, known, next_id):
    """Give every tile an ``image_id``, reusing the ground truth's where it has one."""
    for record in records:
        if record["name"] in known:
            record["image_id"] = known[record["name"]]
        else:
            record["image_id"] = next_id
            next_id += 1
    return next_id


def coco_results(records):
    """The per-tile detections as COCO results, the shape ``predictions.json`` has.

    Same list ``det_solver.val()`` dumps: ``bbox`` is xywh, and the extra
    ``segmentation`` / ``area`` / ``id`` / ``iscrowd`` fields are the ones
    pycocotools' ``loadRes`` fills in, so the file can be loaded either way.
    """
    results = []
    for record in records:
        for det in record["detections"]:
            x0, y0, x1, y1 = det["bbox_xyxy"]
            # the corners are already rounded to 0.1 px; round the derived
            # width and height too, rather than carry their float noise
            w, h = round(x1 - x0, 1), round(y1 - y0, 1)
            results.append(
                {
                    "image_id": record["image_id"],
                    "category_id": det["label"],
                    "bbox": [x0, y0, w, h],
                    "score": det["score"],
                    "segmentation": [[x0, y0, x0, y1, x1, y1, x1, y0]],
                    "area": round(w * h, 2),
                    "id": len(results) + 1,
                    "iscrowd": 0,
                }
            )
    return results


def write_predictions(path, records):
    """``predictions.json`` -- COCO results for the tiles, ready for the eval tools."""
    results = coco_results(records)
    with open(path, "w") as handle:
        json.dump(results, handle)
    print(f"  {len(results)} boxes -> {os.path.basename(path)} (COCO results, tile coordinates)")


# --------------------------------------------------------------------------- #
def main(args):
    names, known_ids, next_id = load_coco(args.coco)

    if args.input:
        scenes = find_scenes(args.input)
        if not scenes:
            raise SystemExit(f"no tiles in {args.input}")
        if len(scenes) > 1 and not args.all:
            scenes = choose_scene(args.input)
    else:
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..", DEFAULT_SPLIT_ROOT)
        scenes = find_scenes(root) if args.all else choose_scene(root)

    print(f"loading {args.resume}")
    model, input_size = build_model(args.config, args.resume, args.device)
    if args.input_size:
        input_size = args.input_size
    print(f"model ready on {args.device}, input size {input_size}")

    grand_total = 0
    for scene_index, (name, folder) in enumerate(scenes, 1):
        print(f"[{scene_index}/{len(scenes)}] {name}")
        out_dir = os.path.join(args.output, name) if args.output else os.path.join(folder, OUT_DIRNAME)
        os.makedirs(out_dir, exist_ok=True)

        manifest, _ = load_manifest(folder)
        if manifest is None:
            print(f"  no {MANIFEST} here -- tiles are processed but not merged into the full image")

        records = run_scene(model, input_size, folder, out_dir, args, names)
        next_id = assign_image_ids(records, known_ids, next_id)
        merged = merge(records, args) if manifest else []

        write_predictions(os.path.join(out_dir, "predictions.json"), records)
        write_json(os.path.join(out_dir, "detections.json"), folder, manifest, records, merged, args, names)
        write_csv(os.path.join(out_dir, "detections.csv"), records, merged, names)
        if args.overlay != "none" and merged:
            write_overlay(
                os.path.join(out_dir, f"{name}_overlay.jpg"),
                manifest,
                merged,
                args.overlay,
                args.max_side,
                names,
            )
        count = len(merged) if merged else sum(len(r["detections"]) for r in records)
        grand_total += count
        print(f"  {count} detections -> {out_dir}")

    print(f"done: {grand_total} detections in {len(scenes)} folder(s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--config", required=True, help="model config yml")
    parser.add_argument("-r", "--resume", required=True, help="checkpoint .pth")
    parser.add_argument(
        "-i",
        "--input",
        default=None,
        help="folder of tiles, or a folder of such folders (split_images/); omit to pick one from a list",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=f"output root; results go in <root>/<folder name>/ (default: <tile folder>/{OUT_DIRNAME})",
    )
    parser.add_argument("--all", action="store_true", help="run every tile folder found, no prompt")
    parser.add_argument("-d", "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch", type=int, default=8, help="tiles per forward pass")
    parser.add_argument("--thrh", type=float, default=0.4, help="score threshold")
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=[3],
        help="keep only these class ids (default 3, the boat class in the AEA annotations)",
    )
    parser.add_argument("--all-classes", action="store_true", help="keep every class")
    parser.add_argument("--nms_iou", type=float, default=0.5, help="IoU for the global NMS")
    parser.add_argument(
        "--edge-tol",
        type=int,
        default=2,
        help="px of slack when stitching a box cut by a tile border; -1 disables stitching",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=None,
        help="size fed to the net; default is the config's eval_spatial_size (1024 for AEA, 800 for AITOD)",
    )
    parser.add_argument(
        "--save-tiles",
        choices=["hits", "all", "none"],
        default="hits",
        help="which tiles to save with boxes drawn (default: only those with a detection)",
    )
    parser.add_argument(
        "--overlay",
        choices=["preview", "full", "none"],
        default="preview",
        help="draw the merged boxes on the source image (default: downscaled preview)",
    )
    parser.add_argument("--max-side", type=int, default=4096, help="--overlay preview: longest side in px")
    parser.add_argument(
        "--coco",
        "--categories",
        dest="coco",
        default=None,
        help="COCO json (e.g. ../data/annotations/val_coco.json): class names for the "
        "drawings, and the image_id a tile listed in it keeps in predictions.json",
    )
    parser.add_argument("--max-tiles", type=int, default=0, help="only the first N tiles, for a quick check")
    args = parser.parse_args()
    if args.all_classes:
        args.classes = None
    main(args)
