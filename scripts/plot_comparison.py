"""Create report-ready charts from the five-method experiment summary.

Run after ``scripts/run_all_experiments.py``:
    python scripts/plot_comparison.py
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np


INPUT_PATH = PROJECT_ROOT / "results" / "comparison" / "summary.csv"
OUTPUT_PATH = PROJECT_ROOT / "results" / "comparison" / "method_comparison.png"


def read_summary(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(
            f"Missing {path.relative_to(PROJECT_ROOT)}. Run scripts/run_all_experiments.py first."
        )
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def values(rows: list[dict[str, str]], metric: str, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    means = np.array([float(row[f"{metric}_mean"]) * scale for row in rows])
    stds = np.array([float(row[f"{metric}_std"]) * scale for row in rows])
    return means, stds


def add_bar_chart(axis, labels, means, stds, title: str, ylabel: str, lower_is_better: bool) -> None:
    colors = ["#2a9d8f", "#457b9d", "#f4a261", "#8d99ae", "#e76f51"]
    bars = axis.bar(labels, means, yerr=stds, capsize=4, color=colors[: len(labels)])
    axis.set_title(title, weight="bold")
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=18)
    axis.grid(axis="y", alpha=0.25)
    winner = int(np.argmin(means) if lower_is_better else np.argmax(means))
    bars[winner].set_edgecolor("#111111")
    bars[winner].set_linewidth(2.2)
    offset = max(float(np.max(means + stds)) * 0.02, 0.001)
    for bar, value, std in zip(bars, means, stds):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + std + offset,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def main() -> None:
    rows = read_summary(INPUT_PATH)
    labels = [row["method"] for row in rows]
    latency, latency_std = values(rows, "average_latency_s", scale=1000.0)
    throughput, throughput_std = values(rows, "throughput_mbps")
    utilization, utilization_std = values(rows, "resource_utilization", scale=100.0)
    jitter, jitter_std = values(rows, "jitter_s", scale=1000.0)

    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    figure.suptitle("5G/6G NTN Network-Slicing Method Comparison", fontsize=16, weight="bold")
    add_bar_chart(axes[0, 0], labels, latency, latency_std, "Average latency", "Milliseconds (lower is better)", True)
    add_bar_chart(axes[0, 1], labels, throughput, throughput_std, "Average throughput", "Mbps (higher is better)", False)
    add_bar_chart(axes[1, 0], labels, jitter, jitter_std, "Average jitter", "Milliseconds (lower is better)", True)
    add_bar_chart(axes[1, 1], labels, utilization, utilization_std, "Resource utilization", "Percent (higher is better)", False)
    figure.text(0.5, 0.005, "Error bars show standard deviation across evaluation episodes.", ha="center", fontsize=9)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight")
    print(f"Saved comparison chart to {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
