"""Small state containers for Phase 1's physical-network simulator."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Allocation:
    bandwidth_hz: np.ndarray
    power_w: np.ndarray


@dataclass
class NetworkSnapshot:
    step: int
    ue_positions_m: np.ndarray
    ue_velocities_mps: np.ndarray
    fbs_position_m: np.ndarray
    fbs_velocity_mps: np.ndarray
    distances_m: np.ndarray
    channel_gains: np.ndarray
    packet_bits: np.ndarray
    allocation: Allocation
    rates_bps: np.ndarray
    latencies_s: np.ndarray
    average_latency_s: float
    throughput_mbps: float
    resource_utilization: float
    jitter_s: float
