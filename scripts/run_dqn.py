"""Run DQN with the common discrete allocation catalog. Requires PyTorch."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.discrete import DiscreteAllocationCatalog
from src.baselines.dqn import DQNAgent
from src.config import load_config
from src.environment import DirectNetworkEnvironment


if __name__ == "__main__":
    config = load_config("configs/paper_config.yaml")
    env = DirectNetworkEnvironment(config)
    catalog = DiscreteAllocationCatalog(env.physical, config["baselines"]["discrete_bandwidth_levels"])
    agent = DQNAgent(env.state_dim, catalog.levels, tuple(config["baselines"]["dqn_hidden_layers"]))
    state = env.reset(config["simulation"]["seed"])
    for _ in range(env.max_steps):
        action = agent.select(state, config["baselines"]["q_epsilon"])
        next_state, reward, done, _ = env.apply_allocation(catalog.allocation(env.snapshot, action))
        agent.observe(state, action, reward, next_state, done)
        agent.train_step()
        state = next_state
        if done:
            break
    env.save_history("results/dqn_last_episode.csv")
