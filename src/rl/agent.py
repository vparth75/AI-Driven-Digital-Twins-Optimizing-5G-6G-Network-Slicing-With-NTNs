"""DDPG implementation from first principles with PyTorch autograd."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

from src.rl.networks import Actor, Critic
from src.rl.noise import OrnsteinUhlenbeckNoise
from src.rl.replay_buffer import ReplayBuffer


@dataclass(frozen=True)
class DDPGConfig:
    state_dim: int
    action_dim: int
    actor_hidden_layers: tuple[int, ...] = (256, 128)
    critic_hidden_layers: tuple[int, ...] = (512, 256, 128)
    batch_size: int = 64
    discount_factor: float = 0.99
    target_tau: float = 0.001
    actor_learning_rate: float = 1e-4
    critic_learning_rate: float = 1e-3
    replay_capacity: int = 100_000
    ou_theta: float = 0.15
    ou_sigma_start: float = 0.2
    ou_sigma_end: float = 0.05
    ou_decay_episodes: int = 500
    seed: int = 42
    device: str = "cpu"


class DDPGAgent:
    """Actor, critic, targets, replay, OU exploration, and soft updates."""

    def __init__(self, config: DDPGConfig):
        self.config = config
        self.device = torch.device(config.device)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        self.actor = Actor(config.state_dim, config.action_dim, config.actor_hidden_layers).to(self.device)
        self.critic = Critic(config.state_dim, config.action_dim, config.critic_hidden_layers).to(self.device)
        self.target_actor = Actor(config.state_dim, config.action_dim, config.actor_hidden_layers).to(self.device)
        self.target_critic = Critic(config.state_dim, config.action_dim, config.critic_hidden_layers).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_learning_rate)
        self.replay = ReplayBuffer(config.replay_capacity, config.state_dim, config.action_dim, config.seed)
        self.noise = OrnsteinUhlenbeckNoise(
            config.action_dim, config.ou_theta, config.ou_sigma_start, config.ou_sigma_end, config.ou_decay_episodes, config.seed
        )

    def reset_exploration(self) -> None:
        self.noise.reset()

    def act(self, state: np.ndarray, episode: int = 0, explore: bool = True) -> np.ndarray:
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(self._tensor(np.asarray(state, dtype=np.float32)[None, :])).cpu().numpy()[0]
        self.actor.train()
        if explore:
            action += self.noise.sample(episode)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def observe(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.replay.add(state, action, reward, next_state, done)

    def train_step(self) -> dict[str, float] | None:
        """One Bellman critic update, deterministic policy update, and soft target update."""
        if len(self.replay) < self.config.batch_size:
            return None
        states, actions, rewards, next_states, dones = (self._tensor(item) for item in self.replay.sample(self.config.batch_size))
        with torch.no_grad():
            target_actions = self.target_actor(next_states)
            target_q = rewards + self.config.discount_factor * (1.0 - dones) * self.target_critic(next_states, target_actions)
        critic_loss = nn.functional.mse_loss(self.critic(states, actions), target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.soft_update_targets()
        return {"actor_loss": float(actor_loss.item()), "critic_loss": float(critic_loss.item())}

    def soft_update_targets(self) -> None:
        tau = self.config.target_tau
        with torch.no_grad():
            for target, source in zip(self.target_actor.parameters(), self.actor.parameters()):
                target.mul_(1.0 - tau).add_(source, alpha=tau)
            for target, source in zip(self.target_critic.parameters(), self.critic.parameters()):
                target.mul_(1.0 - tau).add_(source, alpha=tau)

    def save_checkpoint(self, path: str | Path) -> None:
        """Save model, optimizer, and configuration needed to resume training."""
        torch.save({
            "config": asdict(self.config), "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
            "target_actor": self.target_actor.state_dict(), "target_critic": self.target_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(), "critic_optimizer": self.critic_optimizer.state_dict(),
        }, Path(path))

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.target_actor.load_state_dict(checkpoint["target_actor"])
        self.target_critic.load_state_dict(checkpoint["target_critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

    def _tensor(self, values: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)
