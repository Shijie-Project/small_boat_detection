from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


TRAIN_JSON = "../data/annotations/train_coco.json"
VAL_JSON = "../data/annotations/val_coco.json"
SAVE_DIR = Path("dataset_vis_compare")


def load_coco_json(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_stats(data: dict[str, Any]) -> dict[str, np.ndarray | dict[str, int] | int | float]:
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    image_ids = {img["id"] for img in images}
    ann_per_image = defaultdict(int)

    widths: list[float] = []
    heights: list[float] = []
    areas: list[float] = []
    aspect_ratios: list[float] = []
    centers_x: list[float] = []
    centers_y: list[float] = []
    category_counts = defaultdict(int)

    for image_id in image_ids:
        ann_per_image[image_id] = 0

    AREA_THRESHOLD = 1000

    for ann in annotations:
        image_id = ann["image_id"]
        ann_per_image[image_id] += 1

        x, y, w, h = ann["bbox"]
        area = float(w) * float(h)

        if area >= AREA_THRESHOLD:
            continue

        widths.append(float(w))
        heights.append(float(h))
        areas.append(area)
        aspect_ratios.append(float(w) / float(h) if h > 0 else np.nan)
        centers_x.append(float(x) + float(w) / 2.0)
        centers_y.append(float(y) + float(h) / 2.0)
        category_counts[ann["category_id"]] += 1

    category_id_to_name = {cat["id"]: cat["name"] for cat in categories}
    category_name_counts = {
        category_id_to_name.get(cat_id, str(cat_id)): count for cat_id, count in sorted(category_counts.items())
    }

    counts_array = np.array(list(ann_per_image.values()), dtype=float)
    widths_array = np.array(widths, dtype=float)
    heights_array = np.array(heights, dtype=float)
    areas_array = np.array(areas, dtype=float)
    aspect_ratios_array = np.array(aspect_ratios, dtype=float)
    centers_x_array = np.array(centers_x, dtype=float)
    centers_y_array = np.array(centers_y, dtype=float)

    return {
        "num_images": len(images),
        "num_annotations": len(annotations),
        "instances_per_image": counts_array,
        "widths": widths_array,
        "heights": heights_array,
        "areas": areas_array,
        "aspect_ratios": aspect_ratios_array,
        "centers_x": centers_x_array,
        "centers_y": centers_y_array,
        "category_name_counts": category_name_counts,
    }


def compute_area_bin_counts(areas: np.ndarray) -> dict[str, int]:
    area_bins = {
        "verytiny": (0, 8**2),
        "tiny": (8**2, 16**2),
        "small": (16**2, 32**2),
        "medium": (32**2, np.inf),
    }

    results: dict[str, int] = {}
    for name, (low, high) in area_bins.items():
        mask = (areas >= low) & (areas < high)
        results[name] = int(mask.sum())
    return results


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_overlaid_hist(
    train_values: np.ndarray,
    val_values: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
    bins: int | np.ndarray = 30,
    log_x: bool = False,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(train_values, bins=bins, alpha=0.55, label="Train")
    plt.hist(val_values, bins=bins, alpha=0.55, label="Val")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if log_x:
        plt.xscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(SAVE_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def save_area_bin_bar(train_areas: np.ndarray, val_areas: np.ndarray, filename: str) -> None:
    train_counts = compute_area_bin_counts(train_areas)
    val_counts = compute_area_bin_counts(val_areas)

    labels = list(train_counts.keys())
    train_values = [train_counts[k] for k in labels]
    val_values = [val_counts[k] for k in labels]

    x = np.arange(len(labels))
    width = 0.38

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, train_values, width=width, label="Train")
    plt.bar(x + width / 2, val_values, width=width, label="Val")
    plt.xticks(x, labels)
    plt.title("Object Size Distribution by Area Bin")
    plt.xlabel("Area Bin")
    plt.ylabel("Number of Instances")
    plt.legend()
    plt.tight_layout()
    plt.savefig(SAVE_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def save_category_bar(
    train_category_counts: dict[str, int],
    val_category_counts: dict[str, int],
    filename: str,
) -> None:
    labels = sorted(set(train_category_counts) | set(val_category_counts))
    train_values = [train_category_counts.get(label, 0) for label in labels]
    val_values = [val_category_counts.get(label, 0) for label in labels]

    x = np.arange(len(labels))
    width = 0.38

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, train_values, width=width, label="Train")
    plt.bar(x + width / 2, val_values, width=width, label="Val")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.title("Category Distribution")
    plt.xlabel("Category")
    plt.ylabel("Number of Instances")
    plt.legend()
    plt.tight_layout()
    plt.savefig(SAVE_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def save_summary_text(train_stats: dict[str, Any], val_stats: dict[str, Any], filename: str) -> None:
    def summarize(name: str, stats: dict[str, Any]) -> list[str]:
        counts = stats["instances_per_image"]
        areas = stats["areas"]
        widths = stats["widths"]
        heights = stats["heights"]

        return [
            f"{name}",
            f"  #Images: {stats['num_images']}",
            f"  #Annotations: {stats['num_annotations']}",
            f"  Avg instances/image: {counts.mean():.2f}",
            f"  Median instances/image: {np.median(counts):.2f}",
            f"  Max instances/image: {counts.max():.0f}",
            f"  Avg bbox width: {widths.mean():.2f}",
            f"  Avg bbox height: {heights.mean():.2f}",
            f"  Median bbox area: {np.median(areas):.2f}",
            f"  Mean bbox area: {areas.mean():.2f}",
            "",
        ]

    lines = []
    lines.extend(summarize("Train", train_stats))
    lines.extend(summarize("Val", val_stats))

    with open(SAVE_DIR / filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    ensure_dir(SAVE_DIR)

    train_data = load_coco_json(TRAIN_JSON)
    val_data = load_coco_json(VAL_JSON)

    train_stats = extract_stats(train_data)
    val_stats = extract_stats(val_data)

    save_overlaid_hist(
        train_stats["areas"],
        val_stats["areas"],
        title="BBox Area Distribution",
        xlabel="BBox Area (pixel^2)",
        ylabel="Frequency",
        filename="bbox_area_distribution.png",
        bins=40,
    )

    save_overlaid_hist(
        train_stats["areas"],
        val_stats["areas"],
        title="BBox Area Distribution (Log Scale)",
        xlabel="BBox Area (pixel^2)",
        ylabel="Frequency",
        filename="bbox_area_distribution_log.png",
        bins=np.logspace(0, 4, 40),
        log_x=True,
    )

    save_overlaid_hist(
        train_stats["widths"],
        val_stats["widths"],
        title="BBox Width Distribution",
        xlabel="Width (pixels)",
        ylabel="Frequency",
        filename="bbox_width_distribution.png",
        bins=40,
    )

    save_overlaid_hist(
        train_stats["heights"],
        val_stats["heights"],
        title="BBox Height Distribution",
        xlabel="Height (pixels)",
        ylabel="Frequency",
        filename="bbox_height_distribution.png",
        bins=40,
    )

    save_overlaid_hist(
        train_stats["instances_per_image"],
        val_stats["instances_per_image"],
        title="Instances per Image",
        xlabel="Number of Instances",
        ylabel="Frequency",
        filename="instances_per_image.png",
        bins=20,
    )

    save_overlaid_hist(
        train_stats["aspect_ratios"][~np.isnan(train_stats["aspect_ratios"])],
        val_stats["aspect_ratios"][~np.isnan(val_stats["aspect_ratios"])],
        title="Aspect Ratio Distribution",
        xlabel="Width / Height",
        ylabel="Frequency",
        filename="aspect_ratio_distribution.png",
        bins=40,
    )

    save_area_bin_bar(
        train_stats["areas"],
        val_stats["areas"],
        filename="area_bin_distribution.png",
    )

    save_category_bar(
        train_stats["category_name_counts"],
        val_stats["category_name_counts"],
        filename="category_distribution.png",
    )

    save_summary_text(
        train_stats=train_stats,
        val_stats=val_stats,
        filename="summary.txt",
    )

    print(f"Saved figures to: {SAVE_DIR.resolve()}")


if __name__ == "__main__":
    main()
