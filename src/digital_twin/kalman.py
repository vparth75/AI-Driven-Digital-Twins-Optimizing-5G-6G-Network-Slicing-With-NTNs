"""Six-state Kalman filter for the paper's FBS trajectory model (Eqs. 3-4)."""

import numpy as np


class KalmanFilter3D:
    """Track 3D position and velocity: [x, y, z, vx, vy, vz]."""

    def __init__(self, dt_s: float, measurement_std_m: float, process_std_mps2: float):
        self.dt_s = dt_s
        self.F = np.eye(6)
        self.F[:3, 3:] = np.eye(3) * dt_s
        self.H = np.zeros((3, 6))
        self.H[:3, :3] = np.eye(3)
        self.Q = np.eye(6) * process_std_mps2**2
        self.R = np.eye(3) * measurement_std_m**2
        self.x = np.zeros(6)
        self.P = np.eye(6)

    def reset(self, position_m: np.ndarray, velocity_mps: np.ndarray) -> None:
        self.x = np.concatenate((np.asarray(position_m, dtype=float), np.asarray(velocity_mps, dtype=float)))
        self.P = np.eye(6)

    def predict(self) -> tuple[np.ndarray, np.ndarray]:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.position, self.velocity

    def update(self, measured_position_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        innovation = np.asarray(measured_position_m, dtype=float) - self.H @ self.x
        innovation_covariance = self.H @ self.P @ self.H.T + self.R
        gain = np.linalg.solve(innovation_covariance, self.H @ self.P).T
        self.x = self.x + gain @ innovation
        self.P = (np.eye(6) - gain @ self.H) @ self.P
        return self.position, self.velocity

    @property
    def position(self) -> np.ndarray:
        return self.x[:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:].copy()
