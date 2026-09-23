"""Shared continuous-action decoding used by DT-DDPG and standalone DDPG."""

import numpy as np

from src.network.models import Allocation
from src.network.simulator import NetworkSimulator, dbm_to_watts


def decode_continuous_action(simulator: NetworkSimulator, raw_action: np.ndarray) -> Allocation:
    """Map M bandwidth preferences and M power controls in [-1, 1] to an allocation."""
    action = np.asarray(raw_action, dtype=np.float32)
    expected = simulator.M * 2
    if action.shape != (expected,):
        raise ValueError(f"Expected action shape ({expected},), got {action.shape}")
    action = np.clip(action, -1.0, 1.0)
    preferences = (action[: simulator.M] + 1.0) / 2.0
    weights = preferences + 1e-6
    bandwidth = simulator.network["total_bandwidth_hz"] * weights / weights.sum()
    power_fraction = (action[simulator.M :] + 1.0) / 2.0
    min_power = float(dbm_to_watts(simulator.power["min_dbm"]))
    max_power = float(dbm_to_watts(simulator.power["max_dbm"]))
    power = min_power + power_fraction * (max_power - min_power)
    return simulator.validate_and_project_allocation(Allocation(bandwidth, power))
