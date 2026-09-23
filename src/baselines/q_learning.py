"""Tabular Q-learning over the scalable discrete allocation catalog."""

import numpy as np


class TabularQLearning:
    def __init__(self, num_states: int, num_actions: int, learning_rate: float = 0.01, discount_factor: float = 0.99, epsilon: float = 0.1, seed: int | None = None):
        self.values = np.zeros((num_states, num_actions), dtype=float)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

    def select(self, state: int, explore: bool = True) -> int:
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.values.shape[1]))
        return int(np.argmax(self.values[state]))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        target = reward if done else reward + self.discount_factor * np.max(self.values[next_state])
        self.values[state, action] += self.learning_rate * (target - self.values[state, action])
