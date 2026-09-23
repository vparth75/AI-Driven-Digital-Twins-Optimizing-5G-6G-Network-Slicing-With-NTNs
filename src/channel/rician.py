"""Rician FBS-to-UE channel model based on paper Eq. 2."""

import numpy as np

SPEED_OF_LIGHT_MPS = 299_792_458.0


def rician_power_gain(
    distances_m: np.ndarray,
    carrier_frequency_hz: float,
    k_factor: float,
    path_loss_exponent: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return linear channel power gains for a LoS/NLoS Rician link.

    The paper supplies the Rician composition but no path-loss model. We use
    free-space reference gain at 1 m followed by configurable distance decay;
    this is deliberately an explicit implementation assumption.
    """
    distance = np.maximum(np.asarray(distances_m, dtype=float), 1.0)
    if k_factor < 0 or path_loss_exponent <= 0:
        raise ValueError("k_factor must be non-negative and exponent positive")
    wavelength = SPEED_OF_LIGHT_MPS / carrier_frequency_hz
    reference_gain = (wavelength / (4.0 * np.pi)) ** 2
    path_gain = reference_gain * distance ** (-path_loss_exponent)
    los = np.exp(-1j * 2.0 * np.pi * distance / wavelength)
    nlos = (rng.normal(size=distance.size) + 1j * rng.normal(size=distance.size)) / np.sqrt(2.0)
    fading = np.sqrt(k_factor / (k_factor + 1.0)) * los
    fading += np.sqrt(1.0 / (k_factor + 1.0)) * nlos
    return np.maximum(path_gain * np.abs(fading) ** 2, np.finfo(float).tiny)
