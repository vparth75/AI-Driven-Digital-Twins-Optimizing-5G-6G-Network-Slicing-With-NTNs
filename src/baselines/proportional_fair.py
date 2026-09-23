"""Proportional-fair resource allocation baseline."""

import numpy as np

from src.network.models import Allocation, NetworkSnapshot
from src.network.simulator import NetworkSimulator


class ProportionalFairPolicy:
    """Allocate bandwidth by instantaneous demand divided by moving throughput."""

    def __init__(self, simulator: NetworkSimulator, averaging_alpha: float = 0.05):
        self.simulator = simulator
        self.alpha = averaging_alpha
        self.average_rate_bps = np.ones(simulator.M)

    def reset(self) -> None:
        self.average_rate_bps.fill(1.0)

    def select(self, snapshot: NetworkSnapshot) -> Allocation:
        utility = snapshot.packet_bits / np.maximum(self.average_rate_bps, 1.0)
        weights = utility / utility.sum()
        bandwidth = self.simulator.network["total_bandwidth_hz"] * weights
        return self.simulator.validate_and_project_allocation(
            Allocation(bandwidth, np.full(self.simulator.M, self.simulator.default_power_w))
        )

    def observe(self, snapshot: NetworkSnapshot) -> None:
        self.average_rate_bps = (1.0 - self.alpha) * self.average_rate_bps + self.alpha * snapshot.rates_bps
