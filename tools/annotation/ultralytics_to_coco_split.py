import json
import os
import shutil
from pathlib import Path


def resplit_dataset(old_train_json, old_val_json, ndjson_file, image_folder, output_dir):
    """
    Resplits a COCO dataset and its images based on an Ultralytics ndjson file.
    """
    print("Loading original COCO datasets...")
    all_images = {}
    all_annotations = []
    categories = []
    info = {}

    # 1. Combine all old data into dictionaries
    for json_file in [old_train_json, old_val_json]:
        with open(json_file) as f:
            data = json.load(f)
            # Grab categories and info only once
            if not categories and "categories" in data:
                categories = data["categories"]
            if not info and "info" in data:
                info = data["info"]

            # Map images by their stem (filename without extension) to handle png/jpg differences
            for img in data.get("images", []):
                stem = Path(img["file_name"]).stem
                all_images[stem] = img

            all_annotations.extend(data.get("annotations", []))

    # Group annotations by their image_id for fast lookup
    ann_by_image = {}
    for ann in all_annotations:
        img_id = ann["image_id"]
        if img_id not in ann_by_image:
            ann_by_image[img_id] = []
        ann_by_image[img_id].append(ann)

    print(f"Total images found in old JSONs: {len(all_images)}")

    # 2. Parse the new splits from the ndjson file
    splits = {"train": [], "val": [], "test": []}

    with open(ndjson_file) as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("type") == "image":
                stem = Path(obj["file"]).stem
                split = obj.get("split", "train")  # Default to train if missing

                if split not in splits:
                    splits[split] = []

                if stem in all_images:
                    splits[split].append(stem)

    # 3. Create output directories
    os.makedirs(output_dir, exist_ok=True)
    for split_name in splits:
        if splits[split_name]:
            os.makedirs(os.path.join(output_dir, split_name), exist_ok=True)

    # 4. Generate the new JSONs and copy the images into the corresponding folders
    for split_name, stems in splits.items():
        if not stems:
            continue

        split_images = []
        split_annotations = []

        print(f"\nProcessing '{split_name}' split ({len(stems)} images)...")

        for stem in stems:
            img = all_images[stem]
            split_images.append(img)

            # Add all annotations belonging to this image
            img_id = img["id"]
            if img_id in ann_by_image:
                split_annotations.extend(ann_by_image[img_id])

            # Copy image to the new folder
            src_img_path = os.path.join(image_folder, img["file_name"])
            dst_img_path = os.path.join(output_dir, split_name, img["file_name"])

            if os.path.exists(src_img_path):
                shutil.copy(src_img_path, dst_img_path)
            else:
                print(f"  -> Warning: Source image not found at {src_img_path}")

        # Save the new split COCO json
        new_coco = {"info": info, "categories": categories, "images": split_images, "annotations": split_annotations}

        out_json_path = os.path.join(output_dir, f"new_{split_name}_coco.json")
        with open(out_json_path, "w") as f:
            json.dump(new_coco, f, indent=4)

        print(
            f"Success! Created {out_json_path} with {len(split_images)} images and {len(split_annotations)} annotations."
        )


if __name__ == "__main__":
    OLD_TRAIN_JSON = Path("./annotations/train_coco.json")
    OLD_VAL_JSON = Path("./annotations/val_coco.json")
    NDJSON_FILE = Path("./annotations/ultralytics_split.ndjson")

    # The folder where your current images are stored
    ORIGINAL_IMAGE_FOLDER = "./images"

    # The folder where the newly structured dataset will be placed
    OUTPUT_DIRECTORY = "./resplit_dataset"

    resplit_dataset(
        old_train_json=OLD_TRAIN_JSON,
        old_val_json=OLD_VAL_JSON,
        ndjson_file=NDJSON_FILE,
        image_folder=ORIGINAL_IMAGE_FOLDER,
        output_dir=OUTPUT_DIRECTORY,
    )
