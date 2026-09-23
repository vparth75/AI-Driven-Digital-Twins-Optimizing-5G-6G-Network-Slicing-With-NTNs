import numpy as np

from src.baselines import DiscreteAllocationCatalog, ProportionalFairPolicy, TabularQLearning
from src.config import load_config
from src.environment import DirectNetworkEnvironment


def fresh_environment():
    env = DirectNetworkEnvironment(load_config("configs/paper_config.yaml"))
    env.reset(seed=17)
    return env


def test_direct_environment_uses_same_constraints_as_dt_ddpg():
    env = fresh_environment()
    _, reward, _, info = env.step(np.zeros(env.action_dim, dtype=np.float32))
    assert np.isfinite(reward)
    assert info["bandwidth_sum_hz"] <= env.physical.network["total_bandwidth_hz"]


def test_discrete_catalog_returns_feasible_allocations_for_all_levels():
    env = fresh_environment()
    catalog = DiscreteAllocationCatalog(env.physical, levels=10)
    for action in range(catalog.levels):
        allocation = catalog.allocation(env.snapshot, action)
        assert allocation.bandwidth_hz.sum() <= env.physical.network["total_bandwidth_hz"]
        assert np.all(allocation.power_w >= env.physical.min_power_w)


def test_proportional_fair_policy_tracks_rates_after_observation():
    env = fresh_environment()
    policy = ProportionalFairPolicy(env.physical)
    allocation = policy.select(env.snapshot)
    env.apply_allocation(allocation)
    before = policy.average_rate_bps.copy()
    policy.observe(env.snapshot)
    assert not np.array_equal(before, policy.average_rate_bps)


def test_q_learning_updates_selected_action_value():
    agent = TabularQLearning(num_states=27, num_actions=10, learning_rate=0.01, epsilon=0.0, seed=1)
    agent.update(state=3, action=2, reward=5.0, next_state=4, done=True)
    assert agent.select(3, explore=False) == 2
