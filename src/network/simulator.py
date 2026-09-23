"""Phase 1 physical 5G/6G NTN network simulator."""

from __future__ import annotations

import numpy as np

from src.channel import rician_power_gain
from src.metrics import average_latency, jitter, latency_seconds, resource_utilization, shannon_rates_bps
from src.mobility import move_fbs_toward, move_ues_with_reflection
from src.network.geometry import distances_3d
from src.network.models import Allocation, NetworkSnapshot
from src.traffic import PacketTrafficGenerator


def dbm_to_watts(dbm: float | np.ndarray) -> np.ndarray:
    return 1e-3 * 10.0 ** (np.asarray(dbm, dtype=float) / 10.0)


class NetworkSimulator:
    """Simulates the physical components that later phases will virtualize.

    Phase 1 deliberately has no Digital Twin and no learning agent. Allocation
    defaults to equal RB shares solely to make the simulator runnable.
    """

    def __init__(self, config: dict):
        self.config = config
        self.network = config["network"]
        self.channel = config["channel"]
        self.power = config["power"]
        self.traffic_config = config["traffic"]
        self.dt_s = float(config["simulation"]["step_seconds"])
        self.M = int(self.network["num_ues"])
        self.bounds = np.asarray(self.network["area_bounds_m"], dtype=float)
        self.rb_bandwidth_hz = self.network["total_bandwidth_hz"] / self.network["num_resource_blocks"]
        self.min_power_w = float(dbm_to_watts(self.power["min_dbm"]))
        self.max_power_w = float(dbm_to_watts(self.power["max_dbm"]))
        self.default_power_w = float(dbm_to_watts(self.power["default_dbm"]))
        self.reset()

    def reset(self, seed: int | None = None) -> NetworkSnapshot:
        if seed is None:
            seed = int(self.config["simulation"]["seed"])
        self.rng = np.random.default_rng(seed)
        xmin, xmax, ymin, ymax = self.bounds
        self.ue_positions_m = np.column_stack((
            self.rng.uniform(xmin, xmax, self.M),
            self.rng.uniform(ymin, ymax, self.M),
            np.full(self.M, self.network["ue_height_m"]),
        ))
        headings = self.rng.uniform(0.0, 2.0 * np.pi, self.M)
        speeds = self.rng.uniform(0.0, self.network["ue_max_speed_mps"], self.M)
        self.ue_velocities_mps = np.column_stack((speeds * np.cos(headings), speeds * np.sin(headings), np.zeros(self.M)))
        self.fbs_position_m = np.asarray(self.network["fbs_initial_position_m"], dtype=float)
        self.fbs_velocity_mps = np.zeros(3)
        self.traffic = PacketTrafficGenerator(self.traffic_config["mean_packet_bits"], self.traffic_config["coefficient_of_variation"], self.rng)
        self.step_index = 0
        self.previous_latencies_s: np.ndarray | None = None
        return self._observe(self.default_allocation())

    def default_allocation(self) -> Allocation:
        """Equal active-user RB allocation; a non-learning Phase 1 reference policy."""
        rbs_per_ue = self.network["num_resource_blocks"] // self.M
        remainder = self.network["num_resource_blocks"] % self.M
        rbs = np.full(self.M, rbs_per_ue, dtype=int)
        rbs[:remainder] += 1
        return Allocation(rbs * self.rb_bandwidth_hz, np.full(self.M, self.default_power_w))

    def validate_and_project_allocation(self, allocation: Allocation) -> Allocation:
        """Clip power and project requested bandwidth to valid, non-starving RB totals."""
        bandwidth = np.asarray(allocation.bandwidth_hz, dtype=float)
        power = np.asarray(allocation.power_w, dtype=float)
        if bandwidth.shape != (self.M,) or power.shape != (self.M,):
            raise ValueError(f"Allocation vectors must each have shape ({self.M},)")
        requested_rbs = np.maximum(np.rint(bandwidth / self.rb_bandwidth_hz).astype(int), 0)
        total_rbs = self.network["num_resource_blocks"]
        minimum_rbs = int(self.network.get("minimum_resource_blocks_per_ue", 0))
        if minimum_rbs * self.M > total_rbs:
            raise ValueError("minimum_resource_blocks_per_ue exceeds available resource blocks")
        requested_rbs = np.maximum(requested_rbs, minimum_rbs)
        if requested_rbs.sum() > total_rbs:
            base = np.full(self.M, minimum_rbs, dtype=int)
            available = total_rbs - base.sum()
            priorities = np.maximum(requested_rbs - base, 0)
            if available and priorities.sum():
                scaled = priorities * available / priorities.sum()
                projected_extra = np.floor(scaled).astype(int)
                spare = available - projected_extra.sum()
                if spare:
                    projected_extra[np.argsort(-(scaled - projected_extra))[:spare]] += 1
                requested_rbs = base + projected_extra
            else:
                requested_rbs = base
        return Allocation(requested_rbs * self.rb_bandwidth_hz, np.clip(power, self.min_power_w, self.max_power_w))

    def step(self, allocation: Allocation | None = None) -> NetworkSnapshot:
        """Advance mobility and traffic one step, then evaluate an allocation."""
        self.ue_positions_m, self.ue_velocities_mps = move_ues_with_reflection(
            self.ue_positions_m, self.ue_velocities_mps, self.dt_s, self.bounds
        )
        target_xy = np.mean(self.ue_positions_m[:, :2], axis=0)
        previous_fbs_position = self.fbs_position_m.copy()
        self.fbs_position_m = move_fbs_toward(
            self.fbs_position_m, target_xy, self.network["fbs_speed_mps"], self.dt_s, self.bounds
        )
        self.fbs_velocity_mps = (self.fbs_position_m - previous_fbs_position) / self.dt_s
        self.step_index += 1
        return self._observe(self.default_allocation() if allocation is None else allocation)

    def _observe(self, allocation: Allocation) -> NetworkSnapshot:
        allocation = self.validate_and_project_allocation(allocation)
        distances = distances_3d(self.fbs_position_m, self.ue_positions_m)
        gains = rician_power_gain(
            distances, self.channel["carrier_frequency_hz"], self.channel["rician_k_factor"],
            self.channel["path_loss_exponent"], self.rng,
        )
        packets = self.traffic.sample(self.M)
        noise_psd_w_per_hz = float(dbm_to_watts(self.channel["noise_psd_dbm_per_hz"]))
        rates = shannon_rates_bps(
            allocation.bandwidth_hz, allocation.power_w, gains, noise_psd_w_per_hz,
            self.channel["noise_figure_db"], self.channel["interference_w"],
        )
        latencies = latency_seconds(
            packets, rates, self.traffic_config["processing_delay_s"], np.full(self.M, self.traffic_config["sync_error_m"])
        )
        result = NetworkSnapshot(
            step=self.step_index,
            ue_positions_m=self.ue_positions_m.copy(), ue_velocities_mps=self.ue_velocities_mps.copy(),
            fbs_position_m=self.fbs_position_m.copy(), fbs_velocity_mps=self.fbs_velocity_mps.copy(),
            distances_m=distances, channel_gains=gains, packet_bits=packets, allocation=allocation,
            rates_bps=rates, latencies_s=latencies, average_latency_s=average_latency(latencies),
            throughput_mbps=float(np.sum(rates) / 1e6),
            resource_utilization=resource_utilization(allocation.bandwidth_hz, self.network["total_bandwidth_hz"]),
            jitter_s=jitter(self.previous_latencies_s, latencies),
        )
        self.previous_latencies_s = latencies.copy()
        return result

    def run(self, steps: int) -> list[NetworkSnapshot]:
        return [self.step() for _ in range(steps)]
