"""Scalable discrete allocation choices for Q-learning and DQN baselines."""

from __future__ import annotations

import numpy as np

from src.network.models import Allocation, NetworkSnapshot
from src.network.simulator import NetworkSimulator


class DiscreteAllocationCatalog:
    """Ten demand/channel-aware bandwidth policies on the shared simulator.

    The paper states ten bandwidth levels per UE, whose joint 50-UE action
    space is 10^50 and cannot be tabulated. This centralized catalog retains
    ten discrete actions while making each action a valid full allocation.
    """

    def __init__(self, simulator: NetworkSimulator, levels: int = 10):
        self.simulator = simulator
        self.levels = levels

    def allocation(self, snapshot: NetworkSnapshot, action: int) -> Allocation:
        if not 0 <= action < self.levels:
            raise ValueError(f"Action must be in [0, {self.levels - 1}]")
        priority_strength = action / max(self.levels - 1, 1)
        demand = snapshot.packet_bits / np.maximum(np.mean(snapshot.packet_bits), 1.0)
        channel = snapshot.channel_gains / np.maximum(np.mean(snapshot.channel_gains), np.finfo(float).tiny)
        priority = np.maximum(demand * channel, 1e-6)
        priority /= priority.sum()
        equal = np.full(self.simulator.M, 1.0 / self.simulator.M)
        weights = (1.0 - priority_strength) * equal + priority_strength * priority
        bandwidth = self.simulator.network["total_bandwidth_hz"] * weights
        power = np.full(self.simulator.M, self.simulator.default_power_w)
        return self.simulator.validate_and_project_allocation(Allocation(bandwidth, power))

    def encode_network_state(self, snapshot: NetworkSnapshot) -> int:
        """Three coarse bins each for load, channel quality, and load variability."""
        demand_ratio = np.mean(snapshot.packet_bits) / self.simulator.traffic_config["mean_packet_bits"]
        gain_db = 10.0 * np.log10(np.mean(snapshot.channel_gains))
        variability = np.std(snapshot.packet_bits) / np.maximum(np.mean(snapshot.packet_bits), 1.0)
        load_bin = int(np.digitize(demand_ratio, [0.8, 1.2]))
        gain_bin = int(np.digitize(gain_db, [-105.0, -95.0]))
        variability_bin = int(np.digitize(variability, [0.2, 0.5]))
        return load_bin * 9 + gain_bin * 3 + variability_bin
