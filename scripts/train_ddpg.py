"""Train the standalone, physical-only DDPG baseline. Requires PyTorch."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.environment import DirectNetworkEnvironment
from src.rl import DDPGAgent, DDPGConfig


if __name__ == "__main__":
    config = load_config("configs/paper_config.yaml")
    env = DirectNetworkEnvironment(config)
    agent = DDPGAgent(DDPGConfig(state_dim=env.state_dim, action_dim=env.action_dim))
    for episode in range(config["training"]["episodes"]):
        state = env.reset(config["simulation"]["seed"] + episode)
        agent.reset_exploration()
        for _ in range(env.max_steps):
            action = agent.act(state, episode=episode)
            next_state, reward, done, _ = env.step(action)
            agent.observe(state, action, reward, next_state, done)
            agent.train_step()
            state = next_state
            if done:
                break
    env.save_history("results/standalone_ddpg_last_episode.csv")
