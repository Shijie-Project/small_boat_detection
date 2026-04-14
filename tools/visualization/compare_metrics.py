import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_coco_logs(log_path):
    """
    Parse the plain text log file to extract COCO-style AP/AR metrics and Optimal LRP.
    Returns a dictionary with the max value found for each metric in this log.
    """
    metrics_max = defaultdict(float)

    # Regex pattern to match lines like:
    # Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.161
    # Optimal LRP               @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.866
    pattern = re.compile(r"^(Average Precision.*?\]|Average Recall.*?\]|Optimal LRP.*?\])\s*=\s*([-0-9\.]+)")

    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            match = pattern.match(line)
            if match:
                # Group 1 is the metric name/description, Group 2 is the numerical value
                raw_metric_name = match.group(1).strip()
                # Clean up multiple spaces to make the metric name consistent and neat
                metric_name = re.sub(r"\s+", " ", raw_metric_name)

                try:
                    val = float(match.group(2))
                    # Usually -1.000 means no targets for that category (e.g., no 'verytiny' objects)
                    # We should ignore -1.0 when finding the maximum.
                    if val != -1.0:
                        if metric_name not in metrics_max or val > metrics_max[metric_name]:
                            metrics_max[metric_name] = val
                except ValueError:
                    continue

    return metrics_max


def main(root_dir):
    root_path = Path(root_dir)

    # Look for log.txt inside the first-level subdirectories (e.g., root/20260405-194835/log.txt)
    log_files = list(root_path.glob("*/eval_stage_1/log.txt"))

    if not log_files:
        print(f"No log.txt found in subdirectories of {root_dir}")
        return

    # Data structure: { metric_name: { run_name: max_value } }
    all_results = defaultdict(dict)
    run_names = []

    print(f"Found {len(log_files)} log files. Parsing data...")
    for log_file in sorted(log_files):
        run_name = log_file.parents[1].name
        run_names.append(run_name)

        # Parse the metrics from this specific run
        run_metrics = parse_coco_logs(log_file)

        # Save them into our global results dictionary
        for metric, val in run_metrics.items():
            all_results[metric][run_name] = val

    if not all_results:
        print("No valid COCO metrics found in the logs.")
        return

    # Get a sorted list of all unique metrics found across all logs
    metrics = sorted(all_results.keys())
    print(f"Successfully extracted {len(metrics)} unique metrics.")

    # ==========================================
    # Plotting: Create a grid of bar charts
    # ==========================================
    cols = 3  # 3 charts per row
    rows = (len(metrics) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    axes = axes.flatten()

    # Define some distinct colors for the bars
    colors = plt.cm.tab20(np.linspace(0, 1, len(run_names)))

    for i, metric in enumerate(metrics):
        ax = axes[i]

        # Extract values for this metric in the same order as run_names
        # If a run didn't record this metric, default to 0
        values = [all_results[metric].get(r, 0) for r in run_names]

        # Create Bar chart
        bars = ax.bar(run_names, values, color=colors, edgecolor="black", alpha=0.8)

        # Make the title wrap into multiple lines if it's too long
        title = metric.replace(" |", "\n|")
        ax.set_title(title, fontsize=10, fontweight="bold", pad=10)
        ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1.0)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        # Rotate x-axis labels so the folder timestamps don't overlap
        ax.set_xticks(range(len(run_names)))
        ax.set_xticklabels(run_names, rotation=45, ha="right", fontsize=8)

        # Add the exact value text on top of each bar
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(
                    f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="black",
                )

    # Hide any unused subplots if the number of metrics doesn't perfectly fit the grid
    for j in range(len(metrics), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    output_img = root_path / "coco_metrics_comparison.png"
    plt.savefig(output_img, dpi=200, bbox_inches="tight")
    print(f"\nSaved visualization to: {output_img}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot and compare COCO metrics across multiple subfolders.")
    parser.add_argument(
        "--root_dir",
        type=Path,
        required=True,
        help="Root directory containing the timestamped subfolders (e.g. ./train)",
    )
    args = parser.parse_args()

    main(args.root_dir)
