"""Create a chart for the DT-DDPG synchronization sensitivity study."""

from __future__ import annotations

import csv
import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT = PROJECT_ROOT / "results" / "tuning" / "dt_ddpg_sync_sensitivity_summary.csv"


def metric(rows: list[dict[str, str]], name: str, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([float(row[f"{name}_mean"]) * scale for row in rows]),
        np.array([float(row[f"{name}_std"]) * scale for row in rows]),
    )


def panel(axis, labels, means, stds, title: str, ylabel: str, lower_is_better: bool) -> None:
    colors = ["#2a9d8f", "#457b9d", "#e76f51"]
    bars = axis.bar(labels, means, yerr=stds, capsize=5, color=colors)
    best = int(np.argmin(means) if lower_is_better else np.argmax(means))
    bars[best].set_edgecolor("#111111")
    bars[best].set_linewidth(2.2)
    axis.set_title(title, weight="bold")
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=12)
    offset = max(float(np.max(means + stds)) * 0.025, 0.001)
    for bar, mean, std in zip(bars, means, stds):
        axis.text(bar.get_x() + bar.get_width() / 2, mean + std + offset, f"{mean:.1f}", ha="center", fontsize=9)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a DT-DDPG tuning summary CSV")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Tuning summary CSV to plot")
    parser.add_argument("--output", type=Path, default=None, help="Output PNG path (default: beside input)")
    args = parser.parse_args()
    input_path = args.input if args.input.is_absolute() else PROJECT_ROOT / args.input
    if not input_path.exists():
        raise SystemExit(f"Missing {input_path}. Run scripts/tune_dt_ddpg.py first.")
    output_path = args.output
    if output_path is None:
        output_path = input_path.with_suffix(".png")
    elif not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    with input_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    labels = [f"Sync every\n{row['sync_interval_steps']} step(s)" for row in rows]
    latency, latency_std = metric(rows, "average_latency_s", 1000.0)
    throughput, throughput_std = metric(rows, "throughput_mbps")
    jitter, jitter_std = metric(rows, "jitter_s", 1000.0)
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    seeds = rows[0]["training_seeds"]
    episodes = rows[0]["episodes_per_seed"]
    figure.suptitle(f"DT-DDPG: Digital-Twin Synchronization Study ({seeds} seeds, {episodes} episodes each)", weight="bold")
    panel(axes[0], labels, latency, latency_std, "Latency", "Milliseconds (lower is better)", True)
    panel(axes[1], labels, throughput, throughput_std, "Throughput", "Mbps (higher is better)", False)
    panel(axes[2], labels, jitter, jitter_std, "Jitter", "Milliseconds (lower is better)", True)
    figure.text(0.5, 0.01, "Error bars show standard deviation across independent training seeds.", ha="center", fontsize=9)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    print(f"Saved tuning chart to {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
