"""Ornstein-Uhlenbeck exploration noise from the paper's simulation setup."""

from __future__ import annotations

import numpy as np


class OrnsteinUhlenbeckNoise:
    """Temporally correlated exploration noise with episode-level sigma decay."""

    def __init__(
        self, dimension: int, theta: float = 0.15, sigma_start: float = 0.2,
        sigma_end: float = 0.05, decay_episodes: int = 500, seed: int | None = None,
    ):
        self.dimension = dimension
        self.theta = theta
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.decay_episodes = decay_episodes
        self.rng = np.random.default_rng(seed)
        self.state = np.zeros(dimension, dtype=np.float32)

    def reset(self) -> None:
        self.state.fill(0.0)

    def sigma(self, episode: int) -> float:
        fraction = min(max(episode, 0) / self.decay_episodes, 1.0)
        return self.sigma_start + fraction * (self.sigma_end - self.sigma_start)

    def sample(self, episode: int) -> np.ndarray:
        self.state += self.theta * (-self.state) + self.sigma(episode) * self.rng.standard_normal(self.dimension)
        return self.state.copy()
