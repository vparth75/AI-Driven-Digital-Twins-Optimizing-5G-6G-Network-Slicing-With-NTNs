"""Run the Phase 1 physical-network simulation and save an inspection plot."""

from pathlib import Path
import os
import sys

# Make `python3 scripts/run_phase1.py` work from a source checkout without an
# editable installation. The package root remains the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/dt_ddpg_matplotlib")

import numpy as np

from src.config import load_config
from src.network import NetworkSimulator
from src.visualization import plot_topology


def main() -> None:
    config = load_config("configs/paper_config.yaml")
    simulator = NetworkSimulator(config)
    initial = simulator.reset()
    history = simulator.run(config["simulation"]["duration_steps"])
    final = history[-1]
    output = plot_topology(initial, Path("results/phase1_topology.png"))
    print("Phase 1 physical-network simulation complete")
    print(f"UEs: {simulator.M}; total bandwidth: {simulator.network['total_bandwidth_hz'] / 1e6:.1f} MHz")
    print(f"Mean latency: {np.mean([item.average_latency_s for item in history]) * 1e3:.2f} ms")
    print(f"Mean throughput: {np.mean([item.throughput_mbps for item in history]):.2f} Mbps")
    print(f"Mean resource utilization: {np.mean([item.resource_utilization for item in history]) * 100:.1f}%")
    print(f"Final jitter: {final.jitter_s * 1e3:.2f} ms")
    print(f"Topology plot: {output}")


if __name__ == "__main__":
    main()
