"""Run the five paper methods under identical seeds and save raw summaries.

This intentionally does no charting; Phase 6 turns these unmodified metrics
into paper-comparable figures.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.baselines import DiscreteAllocationCatalog, ProportionalFairPolicy, TabularQLearning
from src.baselines.runner import summarize
from src.config import load_config
from src.environment import DTDDPGEnvironment, DirectNetworkEnvironment


def _aggregate(method: str, episode_metrics: list[dict[str, float]]) -> dict:
    fields = ("average_latency_s", "throughput_mbps", "resource_utilization", "jitter_s")
    result = {"method": method}
    for field in fields:
        values = [item[field] for item in episode_metrics]
        result[f"{field}_mean"] = float(np.mean(values))
        result[f"{field}_std"] = float(np.std(values, ddof=0))
    return result


def _train_continuous(config: dict, environment_type, label: str) -> dict:
    from src.rl import DDPGAgent, DDPGConfig

    env = environment_type(config)
    rl = config["rl"]
    agent = DDPGAgent(DDPGConfig(
        state_dim=env.state_dim, action_dim=env.action_dim, actor_hidden_layers=tuple(rl["actor_hidden_layers"]),
        critic_hidden_layers=tuple(rl["critic_hidden_layers"]), batch_size=rl["batch_size"],
        discount_factor=rl["discount_factor"], target_tau=rl["target_tau"], actor_learning_rate=rl["actor_learning_rate"],
        critic_learning_rate=rl["critic_learning_rate"], replay_capacity=rl["replay_capacity"], ou_theta=rl["ou_theta"],
        ou_sigma_start=rl["ou_sigma_start"], ou_sigma_end=rl["ou_sigma_end"], ou_decay_episodes=rl["ou_decay_episodes"],
        seed=config["simulation"]["seed"],
    ))
    for episode in range(config["training"]["episodes"]):
        state = env.reset(config["simulation"]["seed"] + episode)
        agent.reset_exploration()
        while True:
            action = agent.act(state, episode=episode, explore=True)
            next_state, reward, done, _ = env.step(action)
            agent.observe(state, action, reward, next_state, done)
            agent.train_step()
            state = next_state
            if done:
                break
    metrics = []
    for offset in range(config["baselines"]["evaluation_episodes"]):
        state = env.reset(config["simulation"]["seed"] + 10_000 + offset)
        while True:
            state, _, done, _ = env.step(agent.act(state, explore=False))
            if done:
                break
        metrics.append({
            "average_latency_s": float(np.mean([x["actual_latency_s"] for x in env.history])),
            "throughput_mbps": float(np.mean([x["actual_throughput_mbps"] for x in env.history])),
            "resource_utilization": float(np.mean([x["resource_utilization"] for x in env.history])),
            "jitter_s": float(np.mean([x["actual_jitter_s"] for x in env.history])),
        })
    return _aggregate(label, metrics)


def _evaluate_pf(config: dict) -> dict:
    metrics = []
    for offset in range(config["baselines"]["evaluation_episodes"]):
        env = DirectNetworkEnvironment(config)
        env.reset(config["simulation"]["seed"] + 10_000 + offset)
        policy = ProportionalFairPolicy(env.physical)
        while True:
            _, _, done, _ = env.apply_allocation(policy.select(env.snapshot))
            policy.observe(env.snapshot)
            if done:
                break
        metrics.append(summarize(env.history))
    return _aggregate("Proportional Fairness", metrics)


def _train_q_learning(config: dict) -> dict:
    env = DirectNetworkEnvironment(config)
    catalog = DiscreteAllocationCatalog(env.physical, config["baselines"]["discrete_bandwidth_levels"])
    agent = TabularQLearning(27, catalog.levels, config["baselines"]["q_learning_rate"], config["rl"]["discount_factor"], config["baselines"]["q_epsilon"], config["simulation"]["seed"])
    for episode in range(config["training"]["episodes"]):
        env.reset(config["simulation"]["seed"] + episode)
        while True:
            state_id = catalog.encode_network_state(env.snapshot)
            action = agent.select(state_id, explore=True)
            _, reward, done, _ = env.apply_allocation(catalog.allocation(env.snapshot, action))
            agent.update(state_id, action, reward, catalog.encode_network_state(env.snapshot), done)
            if done:
                break
    metrics = []
    for offset in range(config["baselines"]["evaluation_episodes"]):
        env.reset(config["simulation"]["seed"] + 10_000 + offset)
        while True:
            action = agent.select(catalog.encode_network_state(env.snapshot), explore=False)
            _, _, done, _ = env.apply_allocation(catalog.allocation(env.snapshot, action))
            if done:
                break
        metrics.append(summarize(env.history))
    return _aggregate("Q-learning", metrics)


def _train_dqn(config: dict) -> dict:
    from src.baselines.dqn import DQNAgent

    env = DirectNetworkEnvironment(config)
    catalog = DiscreteAllocationCatalog(env.physical, config["baselines"]["discrete_bandwidth_levels"])
    agent = DQNAgent(env.state_dim, catalog.levels, tuple(config["baselines"]["dqn_hidden_layers"]), batch_size=config["rl"]["batch_size"], seed=config["simulation"]["seed"])
    for episode in range(config["training"]["episodes"]):
        state = env.reset(config["simulation"]["seed"] + episode)
        while True:
            action = agent.select(state, config["baselines"]["q_epsilon"])
            next_state, reward, done, _ = env.apply_allocation(catalog.allocation(env.snapshot, action))
            agent.observe(state, action, reward, next_state, done)
            agent.train_step()
            state = next_state
            if done:
                break
        if (episode + 1) % 100 == 0:
            agent.update_target()
    metrics = []
    for offset in range(config["baselines"]["evaluation_episodes"]):
        state = env.reset(config["simulation"]["seed"] + 10_000 + offset)
        while True:
            action = agent.select(state, epsilon=0.0)
            state, _, done, _ = env.apply_allocation(catalog.allocation(env.snapshot, action))
            if done:
                break
        metrics.append(summarize(env.history))
    return _aggregate("DQN", metrics)


def main() -> None:
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError as error:
        raise SystemExit("PyTorch is required for the complete five-method comparison. Install requirements.txt; no partial comparison was written.") from error
    config = load_config("configs/paper_config.yaml")
    results = [
        _train_continuous(config, DTDDPGEnvironment, "DT-DDPG"),
        _train_continuous(config, DirectNetworkEnvironment, "Standalone DDPG"),
        _evaluate_pf(config), _train_q_learning(config), _train_dqn(config),
    ]
    output_dir = Path("results/comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved five-method comparison to {output_dir}")


if __name__ == "__main__":
    main()
