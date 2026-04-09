import json
from pathlib import Path

import matplotlib.pyplot as plt


LOG_FILE = Path("../../output/dome_m_custom/train/20260405-194835/log.txt")


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


def plot_all(records, metrics):
    epochs = [r.get("epoch", i) for i, r in enumerate(records)]

    def get_value(record, key):
        # check if key is a list element e.g. "test_bbox_0"
        for k, v in record.items():
            if isinstance(v, list):
                for i, item in enumerate(v):
                    if f"{k}_{i}" == key:
                        return item
        return record.get(key)

    cols = 5
    rows = (len(metrics) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, 4 * rows))
    axes = axes.flatten()

    for i, key in enumerate(metrics):
        values = [get_value(r, key) for r in records]
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

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    out = LOG_FILE.parent.joinpath("metrics.png")
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"共绘制 {len(metrics)} 个指标，已保存到 {out}")


if __name__ == "__main__":
    records = load_logs(LOG_FILE)
    print(f"共加载 {len(records)} 条记录，epoch 范围: {records[0]['epoch']} ~ {records[-1]['epoch']}")
    metrics = get_all_metrics(records)
    print(f"发现 {len(metrics)} 个数值指标")
    plot_all(records, metrics)
