"""Geometry for the FBS-to-UE links (paper Eq. 1)."""

import numpy as np


def distances_3d(fbs_position_m: np.ndarray, ue_positions_m: np.ndarray) -> np.ndarray:
    """Return Euclidean 3D distance from the FBS to each UE in metres."""
    fbs = np.asarray(fbs_position_m, dtype=float)
    ues = np.asarray(ue_positions_m, dtype=float)
    if fbs.shape != (3,) or ues.ndim != 2 or ues.shape[1] != 3:
        raise ValueError("Expected FBS shape (3,) and UE positions shape (M, 3)")
    return np.linalg.norm(ues - fbs, axis=1)
