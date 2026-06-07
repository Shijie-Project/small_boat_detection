import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_logs(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    records.sort(key=lambda x: x.get("epoch", 0))
    return records


def get_all_metrics(records):
    keys = set()
    for r in records:
        for k, v in r.items():
            if k == "epoch":
                continue
            if isinstance(v, (int, float)):
                keys.add(k)
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, (int, float)):
                        keys.add(f"{k}_{i}")
    return sorted(keys)


def expand_record(record):
    """Flatten a record into a {key: value} dict, expanding list fields into key_i entries."""
    flat = {}
    for k, v in record.items():
        if isinstance(v, list):
            for i, item in enumerate(v):
                flat[f"{k}_{i}"] = item
        else:
            flat[k] = v
    return flat


def plot_all(records, metrics, output_dir):
    if not metrics:
        print("No numeric metrics found, exiting.")
        return

    epochs = [r.get("epoch", i) for i, r in enumerate(records)]
    flat_records = [expand_record(r) for r in records]

    cols = 5
    rows = (len(metrics) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, 4 * rows))
    axes = axes.flatten()

    for i, key in enumerate(metrics):
        values = [fr.get(key) for fr in flat_records]
        valid = [(e, v) for e, v in zip(epochs, values) if v is not None]
        if not valid:
            axes[i].set_visible(False)
            continue
        x, y = zip(*valid)
        axes[i].plot(x, y, linewidth=1.5, color=f"C{i % 10}")
        axes[i].set_title(key, fontsize=9)
        axes[i].set_xlabel("epoch", fontsize=8)
        axes[i].tick_params(labelsize=7)
        axes[i].grid(True, alpha=0.3)
        axes[i].ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))

        # Annotate the maximum value
        max_y = max(y)
        max_x = x[y.index(max_y)]
        axes[i].annotate(
            f"max={max_y:.4g}",
            xy=(max_x, max_y),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color="red",
        )
        axes[i].scatter([max_x], [max_y], color="red", s=20, zorder=5)

    # Hide any unused subplots; use len(metrics) to avoid relying on loop variable i
    for j in range(len(metrics), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    out = output_dir / "metrics.png"
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"Plotted {len(metrics)} metrics, saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training metrics from a log file.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Training output directory containing log.txt",
    )
    args = parser.parse_args()

    log_file = args.output_dir / "log.txt"

    records = load_logs(log_file)
    if not records:
        print("Log file is empty or could not be parsed, exiting.")
        exit(1)
    print(f"Loaded {len(records)} records, epoch range: {records[0]['epoch']} ~ {records[-1]['epoch']}")
    metrics = get_all_metrics(records)
    print(f"Found {len(metrics)} numeric metrics")
    plot_all(records, metrics, args.output_dir)
