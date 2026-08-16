"""
Concatenate tiled-inference visualizations side by side for easy comparison.

Each tiled *_det.jpg is itself a GT | pred pair (left = green "GT", right =
red "pred"). We keep only the *pred* half of each model so the final panel is:

    +-----------+-----------+-----------+
    |    src    | AEA_pred  |base_pred  |
    +-----------+-----------+-----------+

  * src      : the original satellite image (no boxes)
  * AEA_pred : pred half of output/aea_tiled/vis/<name>_det.jpg
  * base_pred: pred half of output/aitod_baseline_tiled/vis/<name>_det.jpg

All three are resized to a common height, a text label is drawn on top of
each, and the three are laid out horizontally into a single JPG.

Usage
-----
python tools/visualization/concat_tiled_compare.py            # defaults below
python tools/visualization/concat_tiled_compare.py \
    --src-dir ../data/images/all \
    --aea-dir output/aea_tiled/vis \
    --baseline-dir output/aitod_baseline_tiled/vis \
    --out-dir output/compare_src_aea_baseline \
    --height 900
"""

import argparse
import os

from PIL import Image, ImageDraw, ImageFont


# huge satellite images exceed PIL's default decompression-bomb guard
Image.MAX_IMAGE_PIXELS = None

DET_SUFFIX = "_det.jpg"
LABEL_H = 40  # height of the text banner above each image
GAP = 8  # white gap between panels
BG = (255, 255, 255)


def load_font(size=28):
    for name in ("arial.ttf", "DejaVuSans.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def find_source(src_dir, base):
    """Locate the original image for a given basename (extension unknown)."""
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
        p = os.path.join(src_dir, base + ext)
        if os.path.isfile(p):
            return p
    return None


def take_pred_half(img, side="right"):
    """A tiled *_det.jpg is [GT | pred] side by side. Return the pred half.

    The two panels are square (each == image height), so we crop a
    height-wide column from the requested side. This drops the small gap /
    the other panel and the 'GT'/'pred' corner labels stay with their panel.
    """
    h = img.height
    if img.width <= h:  # not a paired image, nothing to split
        return img
    if side == "left":
        return img.crop((0, 0, h, h))
    return img.crop((img.width - h, 0, img.width, h))


def resize_to_height(img, height):
    if img.height == height:
        return img
    w = max(1, round(img.width * height / img.height))
    return img.resize((w, height), Image.LANCZOS)


def label_panel(img, text, font):
    """Return a copy of img with a white banner + centered label on top."""
    panel = Image.new("RGB", (img.width, img.height + LABEL_H), BG)
    panel.paste(img, (0, LABEL_H))
    draw = ImageDraw.Draw(panel)
    tb = draw.textbbox((0, 0), text, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    draw.text(((img.width - tw) / 2, (LABEL_H - th) / 2 - tb[1]), text, fill=(0, 0, 0), font=font)
    return panel


def hconcat(panels):
    total_w = sum(p.width for p in panels) + GAP * (len(panels) - 1)
    total_h = max(p.height for p in panels)
    canvas = Image.new("RGB", (total_w, total_h), BG)
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0))
        x += p.width + GAP
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", default="../data/images/all")
    ap.add_argument("--aea-dir", default="output/aea_tiled/vis")
    ap.add_argument("--baseline-dir", default="output/aitod_baseline_tiled/vis")
    ap.add_argument("--out-dir", default="output/compare_src_aea_baseline")
    ap.add_argument("--height", type=int, default=900, help="common panel height in px")
    ap.add_argument(
        "--pred-side",
        choices=["right", "left"],
        default="right",
        help="which half of the *_det.jpg holds the prediction (tiled output = GT left, pred right)",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    font = load_font(28)

    names = sorted(f for f in os.listdir(args.aea_dir) if f.endswith(DET_SUFFIX))
    print(f"{len(names)} AEA visualizations found")

    ok, skipped = 0, 0
    for fname in names:
        base = fname[: -len(DET_SUFFIX)]
        aea_path = os.path.join(args.aea_dir, fname)
        base_path = os.path.join(args.baseline_dir, fname)
        src_path = find_source(args.src_dir, base)

        missing = [n for n, p in (("baseline", base_path), ("src", src_path)) if not p or not os.path.isfile(p)]
        if missing:
            print(f"  skip {base}: missing {', '.join(missing)}")
            skipped += 1
            continue

        src = Image.open(src_path).convert("RGB")
        aea = take_pred_half(Image.open(aea_path).convert("RGB"), args.pred_side)
        bas = take_pred_half(Image.open(base_path).convert("RGB"), args.pred_side)

        src = resize_to_height(src, args.height)
        aea = resize_to_height(aea, args.height)
        bas = resize_to_height(bas, args.height)

        panels = [
            label_panel(src, "src", font),
            label_panel(aea, "AEA_pred", font),
            label_panel(bas, "baseline_pred", font),
        ]
        out = hconcat(panels)
        out_path = os.path.join(args.out_dir, base + "_compare.jpg")
        out.save(out_path, quality=92)
        ok += 1

    print(f"done: {ok} panels written to {args.out_dir}, {skipped} skipped")


if __name__ == "__main__":
    main()
