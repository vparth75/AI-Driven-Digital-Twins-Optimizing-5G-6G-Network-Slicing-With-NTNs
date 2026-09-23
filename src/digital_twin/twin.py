"""Digital Twin synchronization, prediction, state construction, and simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.digital_twin.kalman import KalmanFilter3D
from src.metrics import average_latency, latency_seconds, resource_utilization, shannon_rates_bps
from src.network.geometry import distances_3d
from src.network.models import Allocation, NetworkSnapshot
from src.network.simulator import NetworkSimulator, dbm_to_watts


@dataclass
class TwinState:
    """Virtual network representation supplied to the future DRL agent."""

    step: int
    ue_positions_m: np.ndarray
    ue_velocities_mps: np.ndarray
    fbs_position_m: np.ndarray
    fbs_velocity_mps: np.ndarray
    distances_m: np.ndarray
    channel_gains: np.ndarray
    csi_db: np.ndarray
    observed_packet_bits: np.ndarray
    predicted_packet_bits: np.ndarray
    synchronization_error_bits: np.ndarray
    is_synchronized: bool


@dataclass
class TwinSimulationResult:
    """Predicted QoS from safely evaluating an allocation in the twin."""

    allocation: Allocation
    rates_bps: np.ndarray
    latencies_s: np.ndarray
    average_latency_s: float
    throughput_mbps: float
    resource_utilization: float
    reward: float


class DigitalTwin:
    """A predictive virtual representation of ``NetworkSimulator`` state.

    The physical simulator remains authoritative. On synchronization steps the
    twin corrects its virtual state with the physical snapshot; between them it
    extrapolates mobility with a Kalman filter, predicts traffic with an EMA,
    and estimates distance-driven channel evolution.
    """

    def __init__(self, simulator: NetworkSimulator):
        self.simulator = simulator
        self.config = simulator.config
        self.settings = self.config["digital_twin"]
        self.reward_settings = self.config["reward"]
        self.kf = KalmanFilter3D(
            simulator.dt_s, self.settings["fbs_measurement_std_m"], self.settings["fbs_process_std_mps2"]
        )
        self.state: TwinState | None = None
        self.last_sync_step = 0
        self.last_observed_packet_bits: np.ndarray | None = None
        self.previous_bandwidth_hz = np.zeros(simulator.M)

    def reset(self, physical: NetworkSnapshot) -> TwinState:
        """Initialize the twin from a freshly observed physical-network state."""
        self.kf.reset(physical.fbs_position_m, physical.fbs_velocity_mps)
        self.last_sync_step = physical.step
        self.last_observed_packet_bits = physical.packet_bits.copy()
        self.previous_bandwidth_hz = physical.allocation.bandwidth_hz.copy()
        self.state = self._state_from_observation(physical, physical.packet_bits.copy(), np.zeros(self.simulator.M), True)
        return self.state

    def advance(self, physical: NetworkSnapshot) -> TwinState:
        """Consume one physical snapshot and either synchronize or predict.

        The caller may retain the returned state for a policy action. The
        physical snapshot is used only for synchronization/error evaluation;
        no allocation is applied to the physical simulator here.
        """
        if self.state is None:
            return self.reset(physical)
        predicted_position, predicted_velocity = self.kf.predict()
        due_for_sync = physical.step - self.last_sync_step >= self.settings["sync_interval_steps"]
        predicted_packets = self._ema_predict(self.state.predicted_packet_bits)
        sync_error = np.asarray(physical.packet_bits) - predicted_packets
        if due_for_sync:
            corrected_position, corrected_velocity = self.kf.update(physical.fbs_position_m)
            # The updated virtual state receives fresh position/CSI data. Its
            # demand remains an EMA forecast so the error is observable.
            observed = self._state_from_observation(physical, predicted_packets, sync_error, True)
            self.state = TwinState(
                **{**observed.__dict__, "fbs_position_m": corrected_position, "fbs_velocity_mps": corrected_velocity}
            )
            self.last_sync_step = physical.step
            self.last_observed_packet_bits = physical.packet_bits.copy()
        else:
            ue_positions = self.state.ue_positions_m + self.state.ue_velocities_mps * self.simulator.dt_s
            self._clip_ue_positions(ue_positions)
            distances = distances_3d(predicted_position, ue_positions)
            # Without a CSI update, retain the fading realization and predict
            # only its path-loss change as distance varies.
            scale = (self.state.distances_m / np.maximum(distances, 1.0)) ** self.simulator.channel["path_loss_exponent"]
            gains = np.maximum(self.state.channel_gains * scale, np.finfo(float).tiny)
            self.state = TwinState(
                step=physical.step, ue_positions_m=ue_positions, ue_velocities_mps=self.state.ue_velocities_mps.copy(),
                fbs_position_m=predicted_position, fbs_velocity_mps=predicted_velocity, distances_m=distances,
                channel_gains=gains, csi_db=10.0 * np.log10(gains), observed_packet_bits=self.last_observed_packet_bits.copy(),
                predicted_packet_bits=predicted_packets,
                synchronization_error_bits=sync_error, is_synchronized=False,
            )
        return self.state

    def build_state(self) -> np.ndarray:
        """Build normalized {h, D, UE position, FBS position, e, CSI} for all UEs.

        This follows Algorithm 1's state contents. One ten-feature row per UE
        is flattened so the future DDPG actor receives a fixed 10*M vector.
        """
        if self.state is None:
            raise RuntimeError("Call reset() before build_state()")
        xmin, xmax, ymin, ymax = self.simulator.bounds
        mean_packet = self.simulator.traffic_config["mean_packet_bits"]
        rows = np.column_stack((
            self.state.csi_db / 100.0,
            self.state.predicted_packet_bits / mean_packet,
            (self.state.ue_positions_m[:, 0] - xmin) / (xmax - xmin),
            (self.state.ue_positions_m[:, 1] - ymin) / (ymax - ymin),
            self.state.ue_positions_m[:, 2] / self.simulator.network["fbs_altitude_m"],
            np.full(self.simulator.M, (self.state.fbs_position_m[0] - xmin) / (xmax - xmin)),
            np.full(self.simulator.M, (self.state.fbs_position_m[1] - ymin) / (ymax - ymin)),
            np.full(self.simulator.M, self.state.fbs_position_m[2] / self.simulator.network["fbs_altitude_m"]),
            self.state.synchronization_error_bits / mean_packet,
            self.state.csi_db / 100.0,
        ))
        return rows.astype(np.float32).ravel()

    def simulate(self, proposed_allocation: Allocation) -> TwinSimulationResult:
        """Evaluate a feasible allocation inside the virtual network only."""
        if self.state is None:
            raise RuntimeError("Call reset() before simulate()")
        allocation = self.simulator.validate_and_project_allocation(proposed_allocation)
        noise_psd_w_per_hz = float(dbm_to_watts(self.simulator.channel["noise_psd_dbm_per_hz"]))
        rates = shannon_rates_bps(
            allocation.bandwidth_hz, allocation.power_w, self.state.channel_gains, noise_psd_w_per_hz,
            self.simulator.channel["noise_figure_db"], self.simulator.channel["interference_w"],
        )
        propagation_error_m = np.abs(self.state.synchronization_error_bits) * self.settings["sync_error_m_per_bit"]
        latencies = latency_seconds(
            self.state.predicted_packet_bits, rates, self.simulator.traffic_config["processing_delay_s"], propagation_error_m
        )
        result = TwinSimulationResult(
            allocation=allocation, rates_bps=rates, latencies_s=latencies, average_latency_s=average_latency(latencies),
            throughput_mbps=float(np.sum(rates) / 1e6),
            resource_utilization=resource_utilization(allocation.bandwidth_hz, self.simulator.network["total_bandwidth_hz"]),
            reward=self._reward(latencies, allocation),
        )
        self.previous_bandwidth_hz = allocation.bandwidth_hz.copy()
        return result

    def _ema_predict(self, previous_prediction: np.ndarray) -> np.ndarray:
        """EMA forecast; α is configurable because Eq. 12 notation is ambiguous."""
        alpha = self.settings["traffic_ema_alpha"]
        return np.maximum(alpha * self.last_observed_packet_bits + (1.0 - alpha) * previous_prediction, 1.0)

    def _state_from_observation(
        self, physical: NetworkSnapshot, predicted_packets: np.ndarray, sync_error: np.ndarray, synchronized: bool
    ) -> TwinState:
        gains = np.maximum(physical.channel_gains, np.finfo(float).tiny)
        return TwinState(
            step=physical.step, ue_positions_m=physical.ue_positions_m.copy(), ue_velocities_mps=physical.ue_velocities_mps.copy(),
            fbs_position_m=physical.fbs_position_m.copy(), fbs_velocity_mps=physical.fbs_velocity_mps.copy(),
            distances_m=physical.distances_m.copy(), channel_gains=gains, csi_db=10.0 * np.log10(gains),
            observed_packet_bits=physical.packet_bits.copy(),
            predicted_packet_bits=np.asarray(predicted_packets, dtype=float).copy(),
            synchronization_error_bits=np.asarray(sync_error, dtype=float).copy(), is_synchronized=synchronized,
        )

    def _reward(self, latencies_s: np.ndarray, allocation: Allocation) -> float:
        bandwidth_change = np.sum(np.abs(allocation.bandwidth_hz - self.previous_bandwidth_hz)) / self.reward_settings["bandwidth_change_unit_hz"]
        sync_error = np.sum(np.abs(self.state.synchronization_error_bits)) / self.reward_settings["sync_error_unit_bits"]
        power = np.sum(allocation.power_w) / self.reward_settings["power_unit_w"]
        return float(
            -np.sum(latencies_s)
            - self.reward_settings["bandwidth_change_weight"] * bandwidth_change
            - self.reward_settings["sync_error_weight"] * sync_error
            - self.reward_settings["power_weight"] * power
        )

    def _clip_ue_positions(self, positions_m: np.ndarray) -> None:
        xmin, xmax, ymin, ymax = self.simulator.bounds
        positions_m[:, 0] = np.clip(positions_m[:, 0], xmin, xmax)
        positions_m[:, 1] = np.clip(positions_m[:, 1], ymin, ymax)
