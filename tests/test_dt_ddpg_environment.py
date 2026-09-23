import numpy as np

from src.config import load_config
from src.environment import DTDDPGEnvironment


def test_dt_ddpg_environment_executes_a_dt_mediated_step(tmp_path):
    environment = DTDDPGEnvironment(load_config("configs/paper_config.yaml"))
    state = environment.reset(seed=3)
    next_state, reward, done, info = environment.step(np.zeros(environment.action_dim, dtype=np.float32))
    assert state.shape == (environment.state_dim,)
    assert next_state.shape == (environment.state_dim,)
    assert np.isfinite(reward)
    assert not done
    assert info["bandwidth_sum_hz"] <= environment.physical.network["total_bandwidth_hz"]
    assert info["actual_latency_s"] > 0
    saved = environment.save_history(tmp_path / "episode.csv")
    assert saved.exists()
    assert "actual_throughput_mbps" in saved.read_text(encoding="utf-8")
