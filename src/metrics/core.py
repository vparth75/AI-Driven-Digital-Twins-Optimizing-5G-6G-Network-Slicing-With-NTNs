"""Rate and QoS metrics corresponding to paper Eqs. 8-10."""

import numpy as np

SPEED_OF_LIGHT_MPS = 299_792_458.0


def shannon_rates_bps(
    bandwidth_hz: np.ndarray,
    power_w: np.ndarray,
    channel_gain: np.ndarray,
    noise_psd_w_per_hz: float,
    noise_figure_db: float,
    interference_w: float | np.ndarray = 0.0,
) -> np.ndarray:
    """Compute R_i = B_i log2(1 + P_i h_i / (N_i + I_i))."""
    bandwidth = np.maximum(np.asarray(bandwidth_hz, dtype=float), 0.0)
    power = np.maximum(np.asarray(power_w, dtype=float), 0.0)
    gain = np.maximum(np.asarray(channel_gain, dtype=float), 0.0)
    noise_w = noise_psd_w_per_hz * bandwidth * 10 ** (noise_figure_db / 10.0)
    denominator = np.maximum(noise_w + np.asarray(interference_w, dtype=float), np.finfo(float).tiny)
    return bandwidth * np.log2(1.0 + power * gain / denominator)


def latency_seconds(packet_bits: np.ndarray, rates_bps: np.ndarray, processing_delay_s: float, sync_error_m: np.ndarray) -> np.ndarray:
    """Compute L_i = k_i/R_i + T_proc + e_i/c (paper Eq. 9)."""
    rates = np.maximum(np.asarray(rates_bps, dtype=float), np.finfo(float).tiny)
    return np.asarray(packet_bits, dtype=float) / rates + processing_delay_s + np.asarray(sync_error_m, dtype=float) / SPEED_OF_LIGHT_MPS


def average_latency(latencies_s: np.ndarray) -> float:
    return float(np.mean(latencies_s))


def resource_utilization(bandwidth_hz: np.ndarray, total_bandwidth_hz: float) -> float:
    return float(np.clip(np.sum(bandwidth_hz) / total_bandwidth_hz, 0.0, 1.0))


def jitter(previous_latencies_s: np.ndarray | None, current_latencies_s: np.ndarray) -> float:
    """Mean absolute inter-step latency variation; 0 for the first sample."""
    if previous_latencies_s is None:
        return 0.0
    return float(np.mean(np.abs(np.asarray(current_latencies_s) - np.asarray(previous_latencies_s))))
