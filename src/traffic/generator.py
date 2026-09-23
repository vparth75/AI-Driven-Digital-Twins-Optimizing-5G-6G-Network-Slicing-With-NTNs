"""Reproducible packet-demand process used by the physical simulator."""

import numpy as np


class PacketTrafficGenerator:
    """Log-normal packet sizes with a configured mean and variation.

    This is an implementation assumption because the paper describes dynamic
    demand but gives no traffic trace or distribution.
    """

    def __init__(self, mean_bits: float, coefficient_of_variation: float, rng: np.random.Generator):
        self.mean_bits = mean_bits
        self.cv = coefficient_of_variation
        self.rng = rng
        self.sigma = float(np.sqrt(np.log1p(coefficient_of_variation**2)))
        self.mu = float(np.log(mean_bits) - self.sigma**2 / 2.0)

    def sample(self, num_ues: int) -> np.ndarray:
        return self.rng.lognormal(self.mu, self.sigma, size=num_ues)
