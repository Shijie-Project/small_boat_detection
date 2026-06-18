"""
Split a COCO file into train/val subsets based on which image folder each image
lives in, and report mismatches between the COCO file and the folders.

Every image in ``all_coco.json`` is routed to ``train_coco.json`` or
``val_coco.json`` depending on whether its image file is found in the train or
the val folder (annotations follow their image). It also prints:

  * unused images — files in a folder that don't appear in the COCO file.
  * unfound images — images in the COCO file not found in either folder.

The COCO ``file_name`` may be a Label Studio export path (URL-encoded, with an
upload-hash prefix); the real image basename is extracted before matching.
"""

import json
import re
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import unquote


IMAGE_EXTENSIONS: set[str] = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

# Label Studio prefixes uploaded files with an 8-hex-char hash + "__".
_LS_HASH_PREFIX = re.compile(r"^[0-9a-fA-F]{8}__")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def clean_image_name(file_name: str) -> str:
    """
    Real image basename from a (possibly Label Studio-mangled) COCO file_name.

    e.g. ``images\\ab12cd34__images%5Call%5CS3.png`` -> ``S3.png``.
    """
    decoded = unquote(str(file_name)).replace("\\", "/")
    base = decoded.rsplit("/", 1)[-1]
    return _LS_HASH_PREFIX.sub("", base)


def iter_folder_image_paths(folder: Path, recursive: bool = False) -> Iterable[Path]:
    """Yield image files from a folder."""
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    for path in iterator:
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def folder_image_names(folder: Path, recursive: bool = False) -> set[str]:
    """Set of image file names present in a folder."""
    return {path.name for path in iter_folder_image_paths(folder, recursive)}


def subset_coco(coco: dict, images: list[dict]) -> dict:
    """Build a COCO dict for `images`, carrying over their annotations + metadata."""
    image_ids = {img["id"] for img in images}
    annotations = [a for a in coco.get("annotations", []) if a.get("image_id") in image_ids]
    return {
        "info": coco.get("info", {}),
        "categories": coco.get("categories", []),
        "images": images,
        "annotations": annotations,
    }


def main(
    coco_json: str,
    train_folder: str,
    val_folder: str,
    out_dir: str | None = None,
    recursive: bool = False,
) -> None:
    coco_json = Path(coco_json)
    train_folder = Path(train_folder)
    val_folder = Path(val_folder)
    out_dir = Path(out_dir) if out_dir else coco_json.parent

    if not coco_json.is_file():
        raise FileNotFoundError(f"COCO file not found: {coco_json}")
    if not train_folder.is_dir():
        raise NotADirectoryError(f"Train image folder not found: {train_folder}")
    if not val_folder.is_dir():
        raise NotADirectoryError(f"Val image folder not found: {val_folder}")

    coco = load_json(coco_json)
    images = coco.get("images", [])

    train_names = folder_image_names(train_folder, recursive)
    val_names = folder_image_names(val_folder, recursive)

    # Route each COCO image to train / val by where its file lives.
    train_images: list[dict] = []
    val_images: list[dict] = []
    unfound: list[str] = []  # in COCO, but in no folder
    for img in images:
        name = clean_image_name(img["file_name"])
        if name in train_names:
            train_images.append(img)
        elif name in val_names:
            val_images.append(img)
        else:
            unfound.append(name)

    # Folder files that the COCO file never references.
    coco_names = {clean_image_name(img["file_name"]) for img in images}
    unused_train = sorted(train_names - coco_names)
    unused_val = sorted(val_names - coco_names)

    train_path = out_dir / "train_coco.json"
    val_path = out_dir / "val_coco.json"
    write_json(train_path, subset_coco(coco, train_images))
    write_json(val_path, subset_coco(coco, val_images))

    print(f"Loaded {len(images)} images / {len(coco.get('annotations', []))} annotations from {coco_json}")
    print(f"  -> train: {len(train_images)} images written to {train_path}")
    print(f"  -> val:   {len(val_images)} images written to {val_path}")

    print(f"\nUnused images in {train_folder} (not in COCO): {len(unused_train)}")
    for name in unused_train:
        print(f"  {name}")
    print(f"\nUnused images in {val_folder} (not in COCO): {len(unused_val)}")
    for name in unused_val:
        print(f"  {name}")

    print(f"\nUnfound images (in COCO but in neither folder): {len(unfound)}")
    for name in sorted(unfound):
        print(f"  {name}")


if __name__ == "__main__":
    main(
        coco_json="../data/annotations/all_coco.json",
        train_folder="../data/images/train",
        val_folder="../data/images/val",
    )
