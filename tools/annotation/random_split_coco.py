"""Split one COCO dataset into the train / val folders training reads.

The step after ``ls_to_coco.py``: annotating leaves a single file covering
every image (``all_coco.json``) beside a single folder of images
(``images/all``), and training wants them cut in two -- a config names
``images/train`` + ``train_coco.json`` and ``images/val`` + ``val_coco.json``,
which is the layout this writes::

    ../data/
        images/all/                  <- --images, every image
        images/train/  images/val/   <- --images-out, written here
        annotations/all_coco.json    <- --coco
        annotations/train_coco.json  annotations/val_coco.json

``--val 0.2`` is the default: one image in five, so train : val is 4 : 1. A
third split follows from ``--test``; left at 0 no test files are written at all.

The draw is stratified on whether an image has boxes, so the background images
-- 128 of the 914 at the time of writing -- keep the same proportion in each
split instead of piling into one of them. ``--seed`` fixes the draw: the same
seed on the same dataset gives the same split back.

Images are copied by default, which costs a second copy of the folder (1.4 GB
today). ``--mode link`` hard-links them instead -- no extra disk, and the tree
still looks like real files to everything downstream -- as long as source and
destination are on one volume. ``--mode none`` writes the two json files and
leaves the images alone, for when they are already where they need to be.

Usage
-----
    # the whole thing, with every default
    python tools/annotation/random_split_coco.py

    # explicit, and hard-linked rather than copied
    python tools/annotation/random_split_coco.py \\
        --coco ../data/annotations/all_coco.json \\
        --images ../data/images/all --val 0.2 --mode link

    # look first, write nothing
    python tools/annotation/random_split_coco.py --dry-run

A split written twice leaves the previous one's files behind in the folders;
``--clean`` empties them first.
"""

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path


SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")


# --------------------------------------------------------------------------- #
# The draw
# --------------------------------------------------------------------------- #
def counts_for(total, val, test):
    """How many of ``total`` go to each split; train takes the remainder.

    Rounding lands on val and test, so ``--test 0`` writes no test split at
    all rather than the one or two images a leftover would leave in it.
    """
    n_val = round(total * val)
    n_test = round(total * test)
    return {"train": total - n_val - n_test, "val": n_val, "test": n_test}


def draw(images, boxes_of, val, test, seed):
    """``{split: [image, ...]}``, drawn separately for annotated and empty images."""
    rng = random.Random(seed)
    splits = {name: [] for name in SPLITS}
    annotated = [image for image in images if boxes_of.get(image["id"])]
    empty = [image for image in images if not boxes_of.get(image["id"])]

    for group in (annotated, empty):
        group = list(group)
        rng.shuffle(group)
        taken = 0
        for name, count in counts_for(len(group), val, test).items():
            splits[name] += group[taken : taken + count]
            taken += count
    return splits, len(annotated), len(empty)


# --------------------------------------------------------------------------- #
# Moving the image files
# --------------------------------------------------------------------------- #
def hardlink(src, dst):
    """Hard-link, or copy when the filesystem will not (a different volume)."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


TRANSFER = {"copy": shutil.copy2, "link": hardlink, "move": shutil.move}


def empty_folder(folder):
    """Delete the image files a previous split left in ``folder``."""
    removed = 0
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            path.unlink()
            removed += 1
    return removed


def transfer_images(names, source, folder, mode, clean, dry_run):
    """Put one split's images in ``folder``; returns what happened, as a line."""
    if mode == "none":
        return "images left where they are"
    if dry_run:
        return f"would {mode} {len(names)} image(s) to {folder}"

    folder.mkdir(parents=True, exist_ok=True)
    note = ""
    if clean:
        removed = empty_folder(folder)
        note = f", {removed} old file(s) removed" if removed else ""

    move = TRANSFER[mode]
    done = skipped = 0
    for name in names:
        source_path = source / name
        target = folder / name
        # A re-run with --mode link finds its own links already in place.
        if mode != "move" and target.exists() and source_path.exists() and target.samefile(source_path):
            skipped += 1
            continue
        move(str(source_path), str(target))
        done += 1
    already = f", {skipped} already there" if skipped else ""
    return f"{mode}: {done} image(s) -> {folder}{already}{note}"


# --------------------------------------------------------------------------- #
def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def split_coco(
    coco,
    images,
    images_out,
    annotations_out,
    val=0.2,
    test=0.0,
    seed=42,
    mode="copy",
    clean=False,
    keep_missing=False,
    dry_run=False,
):
    """Do the split; returns ``{split: (image count, box count)}``."""
    coco, images = Path(coco), Path(images)
    images_out, annotations_out = Path(images_out), Path(annotations_out)

    with coco.open(encoding="utf-8") as handle:
        data = json.load(handle)

    boxes_of = {}
    for annotation in data.get("annotations") or []:
        boxes_of.setdefault(annotation["image_id"], []).append(annotation)
    every = list(data.get("images") or [])
    print(f"{coco.name}: {len(every)} image(s), {sum(len(b) for b in boxes_of.values())} box(es)")

    # An image the folder does not hold cannot be trained on -- training opens
    # every file_name it is given. Drop it, loudly, rather than write a json
    # that fails partway through the first epoch.
    if images.is_dir() and not keep_missing:
        on_disk = {path.name for path in images.iterdir() if path.is_file()}
        missing = [image for image in every if image["file_name"] not in on_disk]
        if missing:
            print(f"  ! {len(missing)} image(s) not in {images} -- dropped (first: {missing[0]['file_name']})")
            every = [image for image in every if image["file_name"] in on_disk]
    if not every:
        print(f"  nothing left to split -- is {images} the folder the file names?")
        return {}

    splits, annotated, empty = draw(every, boxes_of, val, test, seed)
    print(f"  {annotated} annotated, {empty} empty (background) -- stratified, seed {seed}")

    written = {}
    for name in SPLITS:
        chosen = splits[name]
        if not chosen:
            continue
        boxes = [box for image in chosen for box in boxes_of.get(image["id"], [])]
        written[name] = (len(chosen), len(boxes))

        share = len(chosen) / len(every) if every else 0
        print(f"\n[{name}] {len(chosen)} image(s) ({share:.0%}), {len(boxes)} box(es)")
        print(
            "  " + transfer_images([i["file_name"] for i in chosen], images, images_out / name, mode, clean, dry_run)
        )

        target = annotations_out / f"{name}_coco.json"
        if dry_run:
            print(f"  would write {target}")
            continue
        write_json(
            target,
            {
                "info": data.get("info") or {},
                "categories": data.get("categories") or [],
                "images": chosen,
                "annotations": boxes,
            },
        )
        print(f"  wrote {target}")
    return written


# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="split one COCO file + image folder into train / val (/ test)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--coco", default="../data/annotations/all_coco.json", help="the COCO file to split")
    parser.add_argument("--images", default="../data/images/all", help="the folder holding every image in it")
    parser.add_argument("--val", type=float, default=0.2, help="share going to val; 0.2 is train:val = 4:1")
    parser.add_argument("--test", type=float, default=0.0, help="share going to test; 0 writes no test split")
    parser.add_argument("--images-out", default="", help="split folders go here; default: beside --images")
    parser.add_argument("--annotations-out", default="", help="the json files go here; default: beside --coco")
    parser.add_argument("--seed", type=int, default=42, help="the draw; the same seed gives the same split back")
    parser.add_argument(
        "--mode",
        choices=("copy", "link", "move", "none"),
        default="copy",
        help="how images reach the split folders; link costs no disk, move empties --images",
    )
    parser.add_argument("--clean", action="store_true", help="empty the split folders before filling them")
    parser.add_argument("--keep-missing", action="store_true", help="keep images the folder does not hold")
    parser.add_argument("--dry-run", action="store_true", help="report the split, write nothing")
    args = parser.parse_args(argv)

    if not Path(args.coco).is_file():
        parser.error(f"COCO file not found: {args.coco}")
    if args.mode != "none" and not Path(args.images).is_dir():
        parser.error(f"images folder not found: {args.images}")
    if not 0 <= args.val < 1 or not 0 <= args.test < 1 or args.val + args.test >= 1:
        parser.error(f"--val + --test must leave something for train, got {args.val} + {args.test}")

    args.images_out = args.images_out or str(Path(args.images).parent)
    args.annotations_out = args.annotations_out or str(Path(args.coco).parent)
    if args.mode != "none":
        source = Path(args.images).resolve()
        if Path(args.images_out).resolve() == source:
            parser.error(f"--images-out is --images: the splits would nest inside it ({args.images})")
        for name in SPLITS:
            if Path(args.images_out, name).resolve() == source:
                parser.error(f"--images-out would write {name}/ over --images itself: {args.images}")
    return args


def main(argv=None):
    args = parse_args(argv)
    written = split_coco(
        coco=args.coco,
        images=args.images,
        images_out=args.images_out,
        annotations_out=args.annotations_out,
        val=args.val,
        test=args.test,
        seed=args.seed,
        mode=args.mode,
        clean=args.clean,
        keep_missing=args.keep_missing,
        dry_run=args.dry_run,
    )
    summary = ", ".join(f"{name} {images} image(s)/{boxes} box(es)" for name, (images, boxes) in written.items())
    print(f"\n{'[dry run] ' if args.dry_run else ''}{summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
