"""DQN baseline for the discrete allocation catalog; requires PyTorch."""

from __future__ import annotations

import copy
import numpy as np
import torch
from torch import nn

from src.rl.replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(self, state_dim: int, num_actions: int, hidden_layers: tuple[int, ...] = (256, 128), learning_rate: float = 1e-3, discount_factor: float = 0.99, batch_size: int = 64, seed: int = 42):
        torch.manual_seed(seed)
        layers: list[nn.Module] = []
        width = state_dim
        for hidden in hidden_layers:
            layers.extend((nn.Linear(width, hidden), nn.ReLU()))
            width = hidden
        layers.append(nn.Linear(width, num_actions))
        self.network = nn.Sequential(*layers)
        self.target_network = copy.deepcopy(self.network)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.replay = ReplayBuffer(100_000, state_dim, 1, seed)
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.rng = np.random.default_rng(seed)

    def select(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.num_actions))
        with torch.no_grad():
            return int(self.network(torch.as_tensor(state[None, :], dtype=torch.float32)).argmax(dim=1).item())

    def observe(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.replay.add(state, np.array([action], dtype=np.float32), reward, next_state, done)

    def train_step(self) -> float | None:
        if len(self.replay) < self.batch_size:
            return None
        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)
        states_t = torch.as_tensor(states)
        actions_t = torch.as_tensor(actions, dtype=torch.long)
        rewards_t = torch.as_tensor(rewards)
        next_states_t = torch.as_tensor(next_states)
        dones_t = torch.as_tensor(dones)
        chosen_q = self.network(states_t).gather(1, actions_t)
        with torch.no_grad():
            target = rewards_t + self.discount_factor * (1.0 - dones_t) * self.target_network(next_states_t).max(dim=1, keepdim=True).values
        loss = nn.functional.mse_loss(chosen_q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def update_target(self) -> None:
        self.target_network.load_state_dict(self.network.state_dict())
