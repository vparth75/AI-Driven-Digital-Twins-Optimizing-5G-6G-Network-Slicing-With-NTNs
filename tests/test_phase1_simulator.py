import numpy as np

from src.channel import rician_power_gain
from src.config import load_config
from src.metrics import latency_seconds, shannon_rates_bps
from src.network import NetworkSimulator
from src.network.geometry import distances_3d
from src.network.models import Allocation


def config():
    return load_config("configs/paper_config.yaml")


def test_3d_distance_matches_paper_geometry():
    result = distances_3d(np.array([0.0, 0.0, 10.0]), np.array([[3.0, 4.0, 10.0], [0.0, 0.0, 0.0]]))
    np.testing.assert_allclose(result, [5.0, 10.0])


def test_rician_gain_is_positive_and_reproducible():
    distances = np.array([100.0, 200.0])
    first = rician_power_gain(distances, 3.5e9, 6.0, 2.2, np.random.default_rng(11))
    second = rician_power_gain(distances, 3.5e9, 6.0, 2.2, np.random.default_rng(11))
    assert np.all(first > 0)
    np.testing.assert_allclose(first, second)


def test_higher_channel_gain_increases_rate():
    low = shannon_rates_bps(np.array([1e6]), np.array([0.1]), np.array([1e-10]), 4e-21, 7.0)
    high = shannon_rates_bps(np.array([1e6]), np.array([0.1]), np.array([1e-8]), 4e-21, 7.0)
    assert high[0] > low[0] > 0


def test_latency_decreases_with_rate():
    slow = latency_seconds(np.array([1e6]), np.array([1e6]), 0.001, np.array([0.0]))
    fast = latency_seconds(np.array([1e6]), np.array([10e6]), 0.001, np.array([0.0]))
    assert fast[0] < slow[0]


def test_bandwidth_projection_and_power_bounds():
    simulator = NetworkSimulator(config())
    excessive = Allocation(np.full(simulator.M, simulator.network["total_bandwidth_hz"]), np.zeros(simulator.M))
    projected = simulator.validate_and_project_allocation(excessive)
    assert projected.bandwidth_hz.sum() <= simulator.network["total_bandwidth_hz"]
    assert np.all(np.mod(projected.bandwidth_hz, simulator.rb_bandwidth_hz) == 0)
    assert np.all(projected.power_w >= simulator.min_power_w)
    assert np.all(projected.power_w <= simulator.max_power_w)


def test_bandwidth_projection_preserves_minimum_rb_per_ue():
    simulator = NetworkSimulator(config())
    projected = simulator.validate_and_project_allocation(Allocation(np.zeros(simulator.M), np.ones(simulator.M)))
    assert np.all(projected.bandwidth_hz >= simulator.rb_bandwidth_hz)


def test_fbs_speed_limit_is_respected():
    simulator = NetworkSimulator(config())
    start = simulator.fbs_position_m.copy()
    simulator.step()
    movement = np.linalg.norm(simulator.fbs_position_m[:2] - start[:2])
    assert movement <= simulator.network["fbs_speed_mps"] * simulator.dt_s + 1e-10
