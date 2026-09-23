"""Physical-only MDP used by standalone DDPG, Q-learning, and DQN baselines."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from src.environment.action_mapping import decode_continuous_action
from src.network import NetworkSimulator
from src.network.models import Allocation, NetworkSnapshot


class DirectNetworkEnvironment:
    """Same simulator and action constraints as DT-DDPG, without a Digital Twin."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.physical = NetworkSimulator(config)
        self.M = self.physical.M
        self.state_dim = self.M * 10
        self.action_dim = self.M * 2
        self.max_steps = int(config["training"]["steps_per_episode"])
        self.snapshot: NetworkSnapshot | None = None
        self.previous_bandwidth_hz = np.zeros(self.M)
        self.episode_step = 0
        self.history: list[dict[str, float | int]] = []

    def reset(self, seed: int | None = None) -> np.ndarray:
        self.snapshot = self.physical.reset(seed)
        self.previous_bandwidth_hz = self.snapshot.allocation.bandwidth_hz.copy()
        self.episode_step = 0
        self.history = []
        return self._build_state(self.snapshot)

    def decode_action(self, raw_action: np.ndarray) -> Allocation:
        return decode_continuous_action(self.physical, raw_action)

    def step(self, raw_action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        return self.apply_allocation(self.decode_action(raw_action))

    def apply_allocation(self, allocation: Allocation) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Apply a feasible allocation directly to the physical network."""
        allocation = self.physical.validate_and_project_allocation(allocation)
        self.snapshot = self.physical.step(allocation)
        self.episode_step += 1
        reward = self._reward(self.snapshot, allocation)
        done = self.episode_step >= self.max_steps
        info = {
            "step": self.episode_step, "actual_latency_s": self.snapshot.average_latency_s,
            "actual_throughput_mbps": self.snapshot.throughput_mbps, "resource_utilization": self.snapshot.resource_utilization,
            "actual_jitter_s": self.snapshot.jitter_s, "mean_channel_gain": float(np.mean(self.snapshot.channel_gains)),
            "mean_packet_bits": float(np.mean(self.snapshot.packet_bits)), "bandwidth_sum_hz": float(np.sum(allocation.bandwidth_hz)),
            "mean_power_w": float(np.mean(allocation.power_w)),
        }
        self.history.append({**info, "reward": reward})
        self.previous_bandwidth_hz = allocation.bandwidth_hz.copy()
        return self._build_state(self.snapshot), reward, done, {**info, "allocation": allocation}

    def save_history(self, output_path: str | Path) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if not self.history:
            raise ValueError("No episode history is available to save")
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.history[0]))
            writer.writeheader()
            writer.writerows(self.history)
        return output

    def _build_state(self, snapshot: NetworkSnapshot) -> np.ndarray:
        xmin, xmax, ymin, ymax = self.physical.bounds
        mean_packet = self.physical.traffic_config["mean_packet_bits"]
        csi_db = 10.0 * np.log10(np.maximum(snapshot.channel_gains, np.finfo(float).tiny))
        rows = np.column_stack((
            csi_db / 100.0, snapshot.packet_bits / mean_packet,
            (snapshot.ue_positions_m[:, 0] - xmin) / (xmax - xmin),
            (snapshot.ue_positions_m[:, 1] - ymin) / (ymax - ymin),
            snapshot.ue_positions_m[:, 2] / self.physical.network["fbs_altitude_m"],
            np.full(self.M, (snapshot.fbs_position_m[0] - xmin) / (xmax - xmin)),
            np.full(self.M, (snapshot.fbs_position_m[1] - ymin) / (ymax - ymin)),
            np.full(self.M, snapshot.fbs_position_m[2] / self.physical.network["fbs_altitude_m"]),
            np.zeros(self.M), csi_db / 100.0,
        ))
        return rows.astype(np.float32).ravel()

    def _reward(self, snapshot: NetworkSnapshot, allocation: Allocation) -> float:
        settings = self.config["reward"]
        bandwidth_change = np.sum(np.abs(allocation.bandwidth_hz - self.previous_bandwidth_hz)) / settings["bandwidth_change_unit_hz"]
        power = np.sum(allocation.power_w) / settings["power_unit_w"]
        return float(
            -np.sum(snapshot.latencies_s) - settings["bandwidth_change_weight"] * bandwidth_change
            - settings["power_weight"] * power
        )
