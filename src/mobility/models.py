"""Bounded motion models for FBS and ground UEs."""

import numpy as np


def move_fbs_toward(
    position_m: np.ndarray,
    target_xy_m: np.ndarray,
    speed_mps: float,
    dt_s: float,
    bounds_m: np.ndarray,
) -> np.ndarray:
    """Move the FBS toward a target without exceeding its speed bound."""
    result = np.asarray(position_m, dtype=float).copy()
    target = np.asarray(target_xy_m, dtype=float)
    delta = target - result[:2]
    length = float(np.linalg.norm(delta))
    if length:
        result[:2] += delta / length * min(length, speed_mps * dt_s)
    xmin, xmax, ymin, ymax = bounds_m
    result[0] = np.clip(result[0], xmin, xmax)
    result[1] = np.clip(result[1], ymin, ymax)
    return result


def move_ues_with_reflection(
    positions_m: np.ndarray, velocities_mps: np.ndarray, dt_s: float, bounds_m: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Advance UE positions and reflect velocity at rectangular-area edges."""
    positions = np.asarray(positions_m, dtype=float).copy()
    velocities = np.asarray(velocities_mps, dtype=float).copy()
    positions[:, :2] += velocities[:, :2] * dt_s
    xmin, xmax, ymin, ymax = bounds_m
    for axis, lower, upper in ((0, xmin, xmax), (1, ymin, ymax)):
        lower_hits = positions[:, axis] < lower
        upper_hits = positions[:, axis] > upper
        positions[lower_hits, axis] = 2 * lower - positions[lower_hits, axis]
        positions[upper_hits, axis] = 2 * upper - positions[upper_hits, axis]
        velocities[lower_hits | upper_hits, axis] *= -1.0
    return positions, velocities
