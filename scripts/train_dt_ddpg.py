"""Phase 4 DT-DDPG training loop. Requires PyTorch from requirements.txt."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.environment import DTDDPGEnvironment
from src.rl import DDPGAgent, DDPGConfig


def main() -> None:
    config = load_config("configs/paper_config.yaml")
    env = DTDDPGEnvironment(config)
    rl = config["rl"]
    agent = DDPGAgent(DDPGConfig(
        state_dim=env.state_dim, action_dim=env.action_dim,
        actor_hidden_layers=tuple(rl["actor_hidden_layers"]), critic_hidden_layers=tuple(rl["critic_hidden_layers"]),
        batch_size=rl["batch_size"], discount_factor=rl["discount_factor"], target_tau=rl["target_tau"],
        actor_learning_rate=rl["actor_learning_rate"], critic_learning_rate=rl["critic_learning_rate"],
        replay_capacity=rl["replay_capacity"], ou_theta=rl["ou_theta"], ou_sigma_start=rl["ou_sigma_start"],
        ou_sigma_end=rl["ou_sigma_end"], ou_decay_episodes=rl["ou_decay_episodes"], seed=config["simulation"]["seed"],
    ))
    for episode in range(config["training"]["episodes"]):
        state = env.reset(seed=config["simulation"]["seed"] + episode)
        agent.reset_exploration()
        reward_total = 0.0
        for _ in range(config["training"]["steps_per_episode"]):
            action = agent.act(state, episode=episode, explore=True)
            next_state, reward, done, _ = env.step(action)
            agent.observe(state, action, reward, next_state, done)
            agent.train_step()
            state = next_state
            reward_total += reward
            if done:
                break
        if (episode + 1) % 25 == 0:
            print(f"episode {episode + 1:4d} | reward {reward_total:9.2f} | latency {env.history[-1]['actual_latency_s'] * 1e3:7.2f} ms")
        if (episode + 1) % config["training"]["checkpoint_interval"] == 0:
            Path("results/checkpoints").mkdir(parents=True, exist_ok=True)
            agent.save_checkpoint(f"results/checkpoints/dt_ddpg_episode_{episode + 1}.pt")
    log_path = env.save_history("results/dt_ddpg_last_episode.csv")
    print(f"Saved final episode logs to {log_path}")


if __name__ == "__main__":
    main()
