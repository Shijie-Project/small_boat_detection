import json
import os
from typing import Literal
from urllib.parse import unquote


SplitType = Literal["train", "val", "test", "all"]


AITOD_SHIP_CATEGORY_ID = 3


def process_file(split: SplitType) -> None:
    """
    Strip directory prefixes from image filenames in a COCO-format JSON file,
    retaining only the base filename for each image entry.
    """
    assert split in ["train", "val", "test", "all"]

    input_file = f"../data/annotations/{split}_coco.json"

    try:
        with open(input_file, encoding="utf-8") as f:
            data: dict = json.load(f)

        images: list[dict] = data.get("images", [])
        annotations: list[dict] = data.get("annotations", [])

        for entry in images:
            file_name = entry["file_name"]
            decoded = unquote(file_name)
            normalized = decoded.replace("\\", "/")

            entry["file_name"] = os.path.basename(normalized)

        for anno in annotations:
            anno["category_id"] = AITOD_SHIP_CATEGORY_ID

        output_file = os.path.splitext(input_file)[0] + "_fixed.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Done. Processed {len(images)} image entry/entries → {output_file}")

    except (OSError, json.JSONDecodeError, KeyError) as exc:
        print(f"Error: {exc}")
        raise


if __name__ == "__main__":
    process_file(split="all")
