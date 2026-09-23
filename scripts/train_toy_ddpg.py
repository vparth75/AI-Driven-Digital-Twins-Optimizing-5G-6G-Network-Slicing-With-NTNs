"""Train DDPG on a tiny continuous-control task before network integration."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.rl import DDPGAgent, DDPGConfig
from src.rl.toy_env import QuadraticControlEnv


def train(episodes: int = 800) -> dict[str, float]:
    env = QuadraticControlEnv()
    # Smaller hidden layers are an explicit toy-task optimization. The actual
    # DT-DDPG agent will use the paper's 256/128 and 512/256/128 defaults.
    agent = DDPGAgent(DDPGConfig(state_dim=1, action_dim=1, actor_hidden_layers=(64, 32), critic_hidden_layers=(64, 32), seed=7))
    rewards = []
    for episode in range(episodes):
        state = env.reset()
        agent.reset_exploration()
        action = agent.act(state, episode=episode, explore=True)
        next_state, reward, done, _ = env.step(action)
        agent.observe(state, action, reward, next_state, done)
        agent.train_step()
        rewards.append(reward)
    learned_action = float(agent.act(env.reset(), explore=False)[0])
    return {"initial_mean_reward": float(np.mean(rewards[:100])), "final_mean_reward": float(np.mean(rewards[-100:])), "learned_action": learned_action, "target": env.target}


if __name__ == "__main__":
    report = train()
    print("Toy DDPG validation")
    print(f"Initial mean reward: {report['initial_mean_reward']:.4f}")
    print(f"Final mean reward:   {report['final_mean_reward']:.4f}")
    print(f"Learned action:      {report['learned_action']:.3f} (target {report['target']:.3f})")
