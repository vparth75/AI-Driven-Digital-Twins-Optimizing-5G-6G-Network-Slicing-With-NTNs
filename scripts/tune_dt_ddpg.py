"""Measure the effect of Digital-Twin synchronization frequency on DT-DDPG.

This is a sensitivity study, not a claim that a candidate is better until its
saved results are inspected. Each candidate uses the same training/evaluation
seeds and the same DDPG architecture. Only ``sync_interval_steps`` changes.

Example screening run:
    python scripts/tune_dt_ddpg.py --episodes 300 --seeds 3

Paper-length run:
    python scripts/tune_dt_ddpg.py --episodes 1000 --seeds 3
"""

from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.config import load_config
from src.environment import DTDDPGEnvironment
from src.rl import DDPGAgent, DDPGConfig


def make_agent(environment: DTDDPGEnvironment, config: dict, seed: int) -> DDPGAgent:
    rl = config["rl"]
    return DDPGAgent(DDPGConfig(
        state_dim=environment.state_dim,
        action_dim=environment.action_dim,
        actor_hidden_layers=tuple(rl["actor_hidden_layers"]),
        critic_hidden_layers=tuple(rl["critic_hidden_layers"]),
        batch_size=rl["batch_size"],
        discount_factor=rl["discount_factor"],
        target_tau=rl["target_tau"],
        actor_learning_rate=rl["actor_learning_rate"],
        critic_learning_rate=rl["critic_learning_rate"],
        replay_capacity=rl["replay_capacity"],
        ou_theta=rl["ou_theta"],
        ou_sigma_start=rl["ou_sigma_start"],
        ou_sigma_end=rl["ou_sigma_end"],
        ou_decay_episodes=rl["ou_decay_episodes"],
        seed=seed,
    ))


def train_and_evaluate(config: dict, training_seed: int, episodes: int, evaluation_episodes: int) -> dict[str, float]:
    environment = DTDDPGEnvironment(config)
    agent = make_agent(environment, config, training_seed)
    for episode in range(episodes):
        state = environment.reset(seed=training_seed + episode)
        agent.reset_exploration()
        while True:
            action = agent.act(state, episode=episode, explore=True)
            next_state, reward, done, _ = environment.step(action)
            agent.observe(state, action, reward, next_state, done)
            agent.train_step()
            state = next_state
            if done:
                break

    per_episode: list[dict[str, float]] = []
    for offset in range(evaluation_episodes):
        state = environment.reset(seed=10_000 + offset)
        while True:
            state, _, done, _ = environment.step(agent.act(state, explore=False))
            if done:
                break
        per_episode.append({
            "average_latency_s": float(np.mean([row["actual_latency_s"] for row in environment.history])),
            "throughput_mbps": float(np.mean([row["actual_throughput_mbps"] for row in environment.history])),
            "jitter_s": float(np.mean([row["actual_jitter_s"] for row in environment.history])),
        })
    return {key: float(np.mean([row[key] for row in per_episode])) for key in per_episode[0]}


def write_csv(rows: list[dict[str, float | int | str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="DT-DDPG synchronization sensitivity study")
    parser.add_argument("--episodes", type=int, default=300, help="Training episodes per candidate/seed (default: 300)")
    parser.add_argument("--seeds", type=int, default=3, help="Independent training seeds per candidate (default: 3)")
    parser.add_argument("--evaluation-episodes", type=int, default=10, help="Evaluation seeds per trained agent (default: 10)")
    parser.add_argument(
        "--intervals",
        type=int,
        nargs="+",
        choices=(1, 2, 5),
        default=(1, 2, 5),
        help="Synchronization intervals to evaluate; choose from 1, 2, and 5 (default: all three)",
    )
    args = parser.parse_args()
    if args.episodes <= 0 or args.seeds <= 0 or args.evaluation_episodes <= 0:
        raise SystemExit("All numeric arguments must be positive.")

    base = load_config(PROJECT_ROOT / "configs" / "paper_config.yaml")
    candidate_names = {1: "sync_1_step", 2: "sync_2_steps", 5: "baseline_sync_5_steps"}
    candidates = tuple((candidate_names[interval], interval) for interval in args.intervals)
    raw_rows: list[dict[str, float | int | str]] = []
    for name, interval in candidates:
        candidate = copy.deepcopy(base)
        candidate["digital_twin"]["sync_interval_steps"] = interval
        for seed_offset in range(args.seeds):
            training_seed = int(base["simulation"]["seed"]) + seed_offset * 1_000
            print(f"Running {name}: seed {seed_offset + 1}/{args.seeds}, {args.episodes} episodes...", flush=True)
            metrics = train_and_evaluate(candidate, training_seed, args.episodes, args.evaluation_episodes)
            raw_rows.append({"candidate": name, "sync_interval_steps": interval, "training_seed": training_seed, **metrics})
            print(f"  latency={metrics['average_latency_s'] * 1e3:.2f} ms | throughput={metrics['throughput_mbps']:.2f} Mbps", flush=True)

    output_dir = PROJECT_ROOT / "results" / "tuning"
    interval_label = "-".join(str(interval) for interval in args.intervals)
    run_label = f"{args.episodes}ep_{args.seeds}seeds_sync{interval_label}"
    raw_output = output_dir / f"dt_ddpg_sync_sensitivity_{run_label}_raw.csv"
    write_csv(raw_rows, raw_output)
    aggregate_rows: list[dict[str, float | int | str]] = []
    for name, interval in candidates:
        rows = [row for row in raw_rows if row["candidate"] == name]
        aggregate_rows.append({
            "candidate": name,
            "sync_interval_steps": interval,
            "training_seeds": args.seeds,
            "episodes_per_seed": args.episodes,
            "average_latency_s_mean": float(np.mean([row["average_latency_s"] for row in rows])),
            "average_latency_s_std": float(np.std([row["average_latency_s"] for row in rows])),
            "throughput_mbps_mean": float(np.mean([row["throughput_mbps"] for row in rows])),
            "throughput_mbps_std": float(np.std([row["throughput_mbps"] for row in rows])),
            "jitter_s_mean": float(np.mean([row["jitter_s"] for row in rows])),
            "jitter_s_std": float(np.std([row["jitter_s"] for row in rows])),
        })
    summary_output = output_dir / f"dt_ddpg_sync_sensitivity_{run_label}_summary.csv"
    write_csv(aggregate_rows, summary_output)
    print(f"Saved raw results to {raw_output.relative_to(PROJECT_ROOT)}")
    print(f"Saved summary to {summary_output.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
