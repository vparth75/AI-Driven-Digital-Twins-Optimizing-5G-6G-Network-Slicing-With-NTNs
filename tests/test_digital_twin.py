import numpy as np

from src.config import load_config
from src.digital_twin import DigitalTwin
from src.digital_twin.kalman import KalmanFilter3D
from src.network import NetworkSimulator
from src.network.models import Allocation


def fresh_twin():
    simulator = NetworkSimulator(load_config("configs/paper_config.yaml"))
    twin = DigitalTwin(simulator)
    twin.reset(simulator.reset())
    return simulator, twin


def test_kalman_filter_predicts_constant_velocity():
    kf = KalmanFilter3D(1.0, measurement_std_m=0.1, process_std_mps2=0.01)
    kf.reset(np.array([1.0, 2.0, 3.0]), np.array([4.0, 0.0, -1.0]))
    position, velocity = kf.predict()
    np.testing.assert_allclose(position, [5.0, 2.0, 2.0])
    np.testing.assert_allclose(velocity, [4.0, 0.0, -1.0])


def test_twin_constructs_the_paper_state_shape():
    simulator, twin = fresh_twin()
    state = twin.build_state()
    assert state.shape == (simulator.M * 10,)
    assert np.isfinite(state).all()


def test_twin_predicts_between_synchronizations_and_resynchronizes():
    simulator, twin = fresh_twin()
    for _ in range(4):
        state = twin.advance(simulator.step())
        assert not state.is_synchronized
    state = twin.advance(simulator.step())
    assert state.is_synchronized


def test_twin_simulation_projects_action_and_returns_finite_qos():
    simulator, twin = fresh_twin()
    state = twin.advance(simulator.step())
    proposed = Allocation(
        bandwidth_hz=np.full(simulator.M, simulator.network["total_bandwidth_hz"]),
        power_w=np.zeros(simulator.M),
    )
    result = twin.simulate(proposed)
    assert result.allocation.bandwidth_hz.sum() <= simulator.network["total_bandwidth_hz"]
    assert np.all(result.allocation.power_w >= simulator.min_power_w)
    assert np.isfinite(result.average_latency_s)
    assert np.isfinite(result.reward)
    assert np.any(np.abs(state.synchronization_error_bits) > 0)
