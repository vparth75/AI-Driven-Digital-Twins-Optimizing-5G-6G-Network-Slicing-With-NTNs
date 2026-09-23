"""Minimal continuous-control task used to validate DDPG before Phase 4."""

from __future__ import annotations

import numpy as np


class QuadraticControlEnv:
    """One-step task: select an action matching a fixed continuous target."""

    state_dim = 1
    action_dim = 1

    def __init__(self, target: float = 0.65):
        self.target = float(target)

    def reset(self) -> np.ndarray:
        return np.array([self.target], dtype=np.float32)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        bounded = float(np.clip(action[0], -1.0, 1.0))
        reward = -(bounded - self.target) ** 2
        return self.reset(), reward, True, {"action_error": abs(bounded - self.target)}
