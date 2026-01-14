"""
Mathematical operations for the Ab Initio RL Simulation System.

This module provides core numerical algorithms:
- RK4 integration for continuous dynamics (Homeostasis environment)
- Sherman-Morrison update for O(d²) matrix inverse updates (LinUCB)
- Online normalizer using Welford's algorithm (PPO observation normalization)
"""

from typing import Callable

import numpy as np


def rk4_step(
    state: np.ndarray,
    dt: float,
    derivatives_func: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Perform one step of Runge-Kutta 4 integration.

    This provides 4th-order accurate numerical integration for stiff ODEs,
    suitable for the Bergman glucose-insulin model.

    Args:
        state: Current state vector.
        dt: Time step size.
        derivatives_func: Function that computes derivatives given state.
                          Signature: f(state) -> d_state/dt

    Returns:
        State vector after dt time has passed.

    Example:
        >>> def simple_decay(y):
        ...     return -0.1 * y  # dy/dt = -0.1 * y
        >>> y0 = np.array([1.0])
        >>> y1 = rk4_step(y0, 0.1, simple_decay)
    """
    k1 = derivatives_func(state)
    k2 = derivatives_func(state + 0.5 * dt * k1)
    k3 = derivatives_func(state + 0.5 * dt * k2)
    k4 = derivatives_func(state + dt * k3)

    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def rk4_step_with_control(
    state: np.ndarray,
    dt: float,
    derivatives_func: Callable[[np.ndarray, float, float], np.ndarray],
    control: float,
    disturbance: float = 0.0,
) -> np.ndarray:
    """RK4 integration with control input and disturbance.

    Extended version for controlled systems like the Bergman model,
    where derivatives depend on insulin infusion (control) and meal (disturbance).

    Args:
        state: Current state vector [G, X, I].
        dt: Time step size.
        derivatives_func: f(state, control, disturbance) -> d_state/dt
        control: Control input (e.g., insulin infusion rate).
        disturbance: External disturbance (e.g., meal glucose).

    Returns:
        State vector after dt time has passed.
    """
    k1 = derivatives_func(state, control, disturbance)
    k2 = derivatives_func(state + 0.5 * dt * k1, control, disturbance)
    k3 = derivatives_func(state + 0.5 * dt * k2, control, disturbance)
    k4 = derivatives_func(state + dt * k3, control, disturbance)

    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def sherman_morrison_update(
    A_inv: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """Update matrix inverse using Sherman-Morrison formula.

    Given A_inv = A^{-1} and a rank-1 update A_new = A + x @ x.T,
    computes A_new^{-1} in O(d²) instead of O(d³).

    Formula:
        A_new^{-1} = A_inv - (A_inv @ x @ x.T @ A_inv) / (1 + x.T @ A_inv @ x)

    Args:
        A_inv: Current inverse matrix, shape (d, d).
        x: Update vector, shape (d,).

    Returns:
        Updated inverse matrix, shape (d, d).

    Note:
        This is the core update for LinUCB's ridge regression.
    """
    x = x.reshape(-1, 1)  # Ensure column vector (d, 1)
    A_inv_x = A_inv @ x  # (d, 1)
    denominator = 1.0 + (x.T @ A_inv_x).item()  # Scalar

    if denominator < 1e-12:
        # Numerical stability: skip update if denominator is too small
        return A_inv

    return A_inv - (A_inv_x @ A_inv_x.T) / denominator


class OnlineNormalizer:
    """Online running mean and variance estimation using Welford's algorithm.

    This is used for normalizing observations in PPO and other neural network
    agents. It maintains running statistics without storing all samples.

    Attributes:
        mean: Current running mean, shape (dim,).
        var: Current running variance, shape (dim,).
        count: Number of samples seen.
    """

    def __init__(self, dim: int, epsilon: float = 1e-8):
        """Initialize the normalizer.

        Args:
            dim: Dimension of the observations.
            epsilon: Small constant to prevent division by zero.
        """
        self.dim = dim
        self.epsilon = epsilon
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)
        self.count = 0
        self._m2 = np.zeros(dim, dtype=np.float64)  # Running sum of squared deviations

    def update(self, x: np.ndarray) -> None:
        """Update running statistics with a new observation.

        Args:
            x: New observation, shape (dim,) or (batch, dim).
        """
        x = np.asarray(x, dtype=np.float64)

        if x.ndim == 1:
            x = x.reshape(1, -1)

        for sample in x:
            self.count += 1
            delta = sample - self.mean
            self.mean += delta / self.count
            delta2 = sample - self.mean
            self._m2 += delta * delta2

        if self.count > 1:
            self.var = self._m2 / (self.count - 1)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize observations using current statistics.

        Args:
            x: Observation to normalize, shape (dim,) or (batch, dim).

        Returns:
            Normalized observation with zero mean and unit variance.
        """
        return (x - self.mean) / np.sqrt(self.var + self.epsilon)

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        """Reverse normalization.

        Args:
            x: Normalized observation.

        Returns:
            Original-scale observation.
        """
        return x * np.sqrt(self.var + self.epsilon) + self.mean

    def reset(self) -> None:
        """Reset all statistics."""
        self.mean = np.zeros(self.dim, dtype=np.float64)
        self.var = np.ones(self.dim, dtype=np.float64)
        self.count = 0
        self._m2 = np.zeros(self.dim, dtype=np.float64)

    def get_state(self) -> dict:
        """Get current state for saving."""
        return {
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "count": self.count,
            "_m2": self._m2.copy(),
        }

    def set_state(self, state: dict) -> None:
        """Restore state from saved dict."""
        self.mean = state["mean"].copy()
        self.var = state["var"].copy()
        self.count = state["count"]
        self._m2 = state["_m2"].copy()
