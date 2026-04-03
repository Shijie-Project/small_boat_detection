import json
import os
import urllib.parse
from typing import Literal


def normalize_label_studio_path(path_str: str, new_path: str | None = None) -> str:
    """
    Normalize a Label Studio image path for cross-platform compatibility.

    Steps:
        1. URL-decode the path (e.g. convert %5C back to backslash).
        2. Replace Windows-style backslashes with forward slashes.
        3. Optionally remap the directory portion to a new base path.

    Args:
        path_str: The raw path string extracted from the Label Studio task.
        new_path: If provided, replaces the directory portion of the path
                  while preserving the filename.

    Returns:
        The normalized path string.
    """
    if not path_str:
        return path_str

    # Step 1: URL-decode (e.g. %5C → \)
    decoded_path = urllib.parse.unquote(path_str)

    # Step 2: Normalise path separators to forward slashes
    normalized_path = decoded_path.replace("\\", "/")

    # Step 3: Optionally remap the directory
    if new_path is not None:
        prefix, old_path = normalized_path.split("=", maxsplit=1)
        basename = os.path.basename(old_path)
        normalized_path = f"{prefix}={new_path}/{basename}"

    return normalized_path


def process_file(split: Literal["train", "val", "test"]) -> None:
    """
    Read a Label Studio JSON export, normalize image paths, deduplicate tasks
    by filename, and write the result to a new file.
    """
    assert split in ["train", "val", "test"]

    input_file = f"./annotations/{split}_raw.json"
    remapped_image_dir = f"/images/{split}"

    output_file = os.path.splitext(input_file)[0] + "_fixed.json"

    try:
        with open(input_file, encoding="utf-8") as f:
            data: list[dict] = json.load(f)

        with open(input_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)  # Validate JSON structure

        print(f"Processing {len(data)} task(s)...")

        seen_filenames: set[str] = set()
        processed_tasks: list[dict] = []

        for task in data:
            raw_path: str = task["data"]["image"]
            fixed_path = normalize_label_studio_path(raw_path, new_path=remapped_image_dir)
            basename = os.path.basename(fixed_path)

            if basename in seen_filenames:
                continue

            task["data"]["image"] = fixed_path
            seen_filenames.add(basename)
            processed_tasks.append(task)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(processed_tasks, f, ensure_ascii=False, indent=2)

        print(f"Done. {len(processed_tasks)} task(s) written to: {output_file}")
        print("Re-import the fixed file into Label Studio.")

    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        print(f"Error: {exc}")
        raise


if __name__ == "__main__":
    process_file(split=None)
