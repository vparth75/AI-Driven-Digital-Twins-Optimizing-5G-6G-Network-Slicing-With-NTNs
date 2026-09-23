"""Phase 4 integration of physical network, Digital Twin, and continuous actions."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from src.digital_twin import DigitalTwin
from src.environment.action_mapping import decode_continuous_action
from src.network import NetworkSimulator


class DTDDPGEnvironment:
    """DT-mediated continuous-control environment for the future DDPG agent.

    A raw action contains M bandwidth preferences followed by M power controls.
    It is decoded and constrained before the Digital Twin predicts its outcome.
    The same feasible allocation is then applied to the physical simulator to
    produce the next synchronized/predicted twin state.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.physical = NetworkSimulator(config)
        self.twin = DigitalTwin(self.physical)
        self.M = self.physical.M
        self.state_dim = self.M * 10
        self.action_dim = self.M * 2
        self.max_steps = int(config["training"]["steps_per_episode"])
        self.episode_step = 0
        self.history: list[dict[str, float | int | bool]] = []

    def reset(self, seed: int | None = None) -> np.ndarray:
        physical_state = self.physical.reset(seed)
        self.twin.reset(physical_state)
        self.episode_step = 0
        self.history = []
        return self.twin.build_state()

    def decode_action(self, raw_action: np.ndarray):
        """Map [-1, 1] actor output to continuous bandwidth and bounded power."""
        return decode_continuous_action(self.physical, raw_action)

    def step(self, raw_action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Evaluate in the twin, apply physically, then obtain the next twin state."""
        if self.twin.state is None:
            raise RuntimeError("Call reset() before step()")
        allocation = self.decode_action(raw_action)
        predicted = self.twin.simulate(allocation)
        actual = self.physical.step(predicted.allocation)
        next_twin_state = self.twin.advance(actual)
        self.episode_step += 1
        done = self.episode_step >= self.max_steps
        info = {
            "step": self.episode_step,
            "was_synchronized": next_twin_state.is_synchronized,
            "predicted_latency_s": predicted.average_latency_s,
            "actual_latency_s": actual.average_latency_s,
            "predicted_throughput_mbps": predicted.throughput_mbps,
            "actual_throughput_mbps": actual.throughput_mbps,
            "resource_utilization": actual.resource_utilization,
            "actual_jitter_s": actual.jitter_s,
            "mean_channel_gain": float(np.mean(next_twin_state.channel_gains)),
            "mean_packet_bits": float(np.mean(next_twin_state.predicted_packet_bits)),
            "mean_sync_error_bits": float(np.mean(np.abs(next_twin_state.synchronization_error_bits))),
            "bandwidth_sum_hz": float(np.sum(predicted.allocation.bandwidth_hz)),
            "mean_power_w": float(np.mean(predicted.allocation.power_w)),
        }
        self.history.append({**info, "reward": predicted.reward})
        return self.twin.build_state(), predicted.reward, done, {**info, "allocation": predicted.allocation}

    def save_history(self, output_path: str | Path) -> Path:
        """Write per-step DT-DDPG logs to CSV for later experiment analysis."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if not self.history:
            raise ValueError("No episode history is available to save")
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.history[0]))
            writer.writeheader()
            writer.writerows(self.history)
        return output
