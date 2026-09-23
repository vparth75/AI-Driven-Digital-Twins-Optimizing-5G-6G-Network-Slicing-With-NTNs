"""Static topology visualization for Phase 1 inspection."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.network.models import NetworkSnapshot


def plot_topology(snapshot: NetworkSnapshot, output_path: str | Path) -> Path:
    """Save FBS/UE locations, FBS links, and channel quality to a PNG."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    gains_db = 10.0 * np.log10(snapshot.channel_gains)
    normalizer = plt.Normalize(gains_db.min(), gains_db.max())
    cmap = plt.colormaps["viridis"]
    for ue, gain in zip(snapshot.ue_positions_m, gains_db):
        ax.plot([snapshot.fbs_position_m[0], ue[0]], [snapshot.fbs_position_m[1], ue[1]], color=cmap(normalizer(gain)), alpha=0.38, linewidth=0.8)
    points = ax.scatter(snapshot.ue_positions_m[:, 0], snapshot.ue_positions_m[:, 1], c=gains_db, cmap=cmap, s=34, label="UE")
    ax.scatter(snapshot.fbs_position_m[0], snapshot.fbs_position_m[1], marker="^", s=220, c="#d95f02", edgecolors="black", label="UAV/FBS")
    ax.set(title="Phase 1 - FBS-to-UE topology and Rician channel quality", xlabel="x (m)", ylabel="y (m)", aspect="equal")
    ax.legend(loc="upper right")
    colorbar = fig.colorbar(points, ax=ax, pad=0.02)
    colorbar.set_label("Channel power gain (dB)")
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output
