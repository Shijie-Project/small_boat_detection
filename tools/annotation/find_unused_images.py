"""Print image filenames in a folder that are not used in a COCO-style JSON file."""

import json
from collections.abc import Iterable
from pathlib import Path


IMAGE_EXTENSIONS: set[str] = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def load_used_image_names(json_path: Path) -> set[str]:
    """Load image filenames from a COCO-style JSON file."""
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    images = data.get("images", [])
    return {str(item["file_name"]).strip() for item in images if isinstance(item, dict) and "file_name" in item}


def iter_folder_image_paths(folder: Path, recursive: bool = False) -> Iterable[Path]:
    """Yield image files from a folder."""
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    for path in iterator:
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def main(json_file: str, image_folder: str, recursive: bool = False, ignore_case: bool = False) -> None:
    json_file = Path(json_file)
    image_folder = Path(image_folder)

    if not json_file.is_file():
        raise FileNotFoundError(f"JSON file not found: {json_file}")

    if not image_folder.is_dir():
        raise NotADirectoryError(f"Image folder not found: {image_folder}")

    used_names = load_used_image_names(json_file)

    if ignore_case:
        used_lookup = {name.lower() for name in used_names}
        unused_names = sorted(
            path.name
            for path in iter_folder_image_paths(image_folder, recursive)
            if path.name.lower() not in used_lookup
        )
    else:
        unused_names = sorted(
            path.name for path in iter_folder_image_paths(image_folder, recursive) if path.name not in used_names
        )

    for name in unused_names:
        print(name)


if __name__ == "__main__":
    main("./annotations/train_coco.json", image_folder="./images/", recursive=True)
