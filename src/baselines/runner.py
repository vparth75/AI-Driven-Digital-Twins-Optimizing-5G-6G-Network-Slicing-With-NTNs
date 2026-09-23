"""Shared episode runners for fair baseline measurements."""

from __future__ import annotations

import numpy as np

from src.baselines.discrete import DiscreteAllocationCatalog
from src.baselines.proportional_fair import ProportionalFairPolicy
from src.baselines.q_learning import TabularQLearning
from src.environment import DirectNetworkEnvironment


def summarize(history: list[dict]) -> dict[str, float]:
    return {
        "average_latency_s": float(np.mean([row["actual_latency_s"] for row in history])),
        "throughput_mbps": float(np.mean([row["actual_throughput_mbps"] for row in history])),
        "resource_utilization": float(np.mean([row["resource_utilization"] for row in history])),
        "jitter_s": float(np.mean([row["actual_jitter_s"] for row in history])),
    }


def run_proportional_fair(config: dict, seed: int) -> tuple[dict[str, float], list[dict]]:
    env = DirectNetworkEnvironment(config)
    env.reset(seed)
    policy = ProportionalFairPolicy(env.physical)
    policy.reset()
    for _ in range(env.max_steps):
        allocation = policy.select(env.snapshot)
        _, _, done, _ = env.apply_allocation(allocation)
        policy.observe(env.snapshot)
        if done:
            break
    return summarize(env.history), env.history


def run_q_learning(config: dict, seed: int, train: bool = True) -> tuple[dict[str, float], list[dict]]:
    env = DirectNetworkEnvironment(config)
    catalog = DiscreteAllocationCatalog(env.physical, config["baselines"]["discrete_bandwidth_levels"])
    agent = TabularQLearning(27, catalog.levels, config["baselines"]["q_learning_rate"], config["rl"]["discount_factor"], config["baselines"]["q_epsilon"], seed)
    env.reset(seed)
    for _ in range(env.max_steps):
        state_id = catalog.encode_network_state(env.snapshot)
        action = agent.select(state_id, explore=train)
        _, reward, done, _ = env.apply_allocation(catalog.allocation(env.snapshot, action))
        next_state_id = catalog.encode_network_state(env.snapshot)
        if train:
            agent.update(state_id, action, reward, next_state_id, done)
        if done:
            break
    return summarize(env.history), env.history
