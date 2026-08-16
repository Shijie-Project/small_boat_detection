#!/usr/bin/env python
"""Render augmented training tiles so the augmentation can be eyeballed.

    python tools/visualization/preview_transforms.py -c configs/dome/Dome-M-AEA.yml -n 12

Writes to ``output/transform_preview/`` by default:

    tile_XX.png    the augmented tile with its boxes drawn
    crops.png      every box in those tiles as a 64 px crop, side by side

The contact sheet is the useful one for this dataset -- at 1024 px a 6 px boat
is a speck, and it is where you can tell a pasted boat from a real one.
"""

import argparse
import os
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torchvision.ops import box_convert
from torchvision.utils import draw_bounding_boxes, save_image

from src.core import YAMLConfig


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--config", required=True, help="config to take the train transforms from")
    parser.add_argument("-n", "--num", type=int, default=12, help="how many tiles to render")
    parser.add_argument("-o", "--output-dir", default="output/transform_preview")
    parser.add_argument("--start", type=int, default=0, help="first dataset index")
    parser.add_argument("--epoch", type=int, default=0, help="epoch to report to the aug policy")
    parser.add_argument("--crop-size", type=int, default=64, help="side of each contact-sheet crop")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def box_layout(cfg):
    """What the pipeline's last ConvertBoxes leaves the boxes as.

    It hands back a plain tensor, so the format cannot be read off the result --
    but the config that produced it says exactly what it is.
    """
    fmt, normalized = "xyxy", False
    ops = cfg.yaml_cfg["train_dataloader"]["dataset"]["transforms"].get("ops") or []
    for op in ops:
        if isinstance(op, dict) and op.get("type") == "ConvertBoxes":
            fmt = str(op.get("fmt", "xyxy")).lower()
            normalized = bool(op.get("normalize", False))
    return fmt, normalized


def to_xyxy(boxes, fmt, normalized, height, width):
    """Boxes as absolute xyxy so they can be drawn."""
    boxes = boxes.detach().float().as_subclass(torch.Tensor)
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)
    if normalized:
        boxes = boxes * torch.tensor([width, height, width, height], dtype=boxes.dtype)
    if fmt != "xyxy":
        boxes = box_convert(boxes, in_fmt=fmt, out_fmt="xyxy")
    return boxes


def crop_around(image, box, size):
    """A fixed-size window centred on the box, padded at the tile edge."""
    _, height, width = image.shape
    cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
    half = size // 2
    patch = torch.zeros(3, size, size, dtype=image.dtype)
    x1, y1 = max(0, cx - half), max(0, cy - half)
    x2, y2 = min(width, cx + half), min(height, cy + half)
    region = image[:, y1:y2, x1:x2]
    patch[:, : region.shape[1], : region.shape[2]] = region
    return patch


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    cfg = YAMLConfig(args.config)
    dataset = cfg.train_dataloader.dataset
    dataset.set_epoch(args.epoch)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fmt, normalized = box_layout(cfg)
    crops, total_boxes = [], 0
    for offset in range(args.num):
        index = (args.start + offset) % len(dataset)
        image, target = dataset[index]
        image = image.detach().float().clamp(0, 1)
        boxes = to_xyxy(target["boxes"], fmt, normalized, image.shape[1], image.shape[2])
        total_boxes += len(boxes)

        drawn = draw_bounding_boxes((image * 255).to(torch.uint8), boxes, colors="red", width=2)
        save_image(drawn.float() / 255.0, out_dir / f"tile_{offset:02d}.png")
        crops.extend(crop_around(image, box, args.crop_size) for box in boxes)

    if crops:
        grid = torch.stack(crops)
        columns = min(16, len(crops))
        save_image(grid, out_dir / "crops.png", nrow=columns, padding=2, pad_value=1.0)

    print(f"{args.num} tiles, {total_boxes} boxes -> {out_dir}{os.sep}")
    print(f"  tile_00..{args.num - 1:02d}.png   full tiles with boxes")
    print(f"  crops.png              {len(crops)} crops of {args.crop_size} px")


if __name__ == "__main__":
    main()
