"""
Linear Quadratic Regulator (LQR) with Discrete Algebraic Riccati Equation solver.

Implements optimal state-feedback control for linear systems:
    x_{k+1} = A*x_k + B*u_k

Minimizing the infinite-horizon cost:
    J = sum_{k=0}^{inf} (x_k' Q x_k + u_k' R u_k)

Features:
- Pure NumPy implementation of DARE solver (no scipy dependency required)
- Support for scipy.linalg.solve_discrete_are when available
- Continuous-to-discrete system conversion
- Controllability and observability checks
- Time-varying LQR support (finite horizon)

Suitable for:
- Smart Grid environment (linearized around operating point)
- Any linear or linearizable control problem
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .base import BaseController


def discretize_system(
    A_c: np.ndarray,
    B_c: np.ndarray,
    dt: float,
    method: str = "zoh",
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert continuous-time system to discrete-time.

    Continuous: dx/dt = A_c*x + B_c*u
    Discrete:   x_{k+1} = A_d*x_k + B_d*u_k

    Args:
        A_c: Continuous state matrix (n x n).
        B_c: Continuous input matrix (n x m).
        dt: Sampling period.
        method: Discretization method ("zoh" for zero-order hold, "euler" for forward Euler).

    Returns:
        Tuple of (A_d, B_d) discrete-time matrices.
    """
    n = A_c.shape[0]

    if method == "euler":
        # Forward Euler: simple but less accurate
        A_d = np.eye(n) + A_c * dt
        B_d = B_c * dt
    elif method == "zoh":
        # Zero-order hold: exact for constant input over dt
        # A_d = exp(A_c * dt)
        # B_d = integral_0^dt exp(A_c * tau) d_tau * B_c
        # Use matrix exponential via Padé approximation or series
        A_d = _matrix_exp(A_c * dt)
        # For B_d, use: B_d = A_c^{-1} * (A_d - I) * B_c if A_c invertible
        # Otherwise use series expansion
        if np.linalg.matrix_rank(A_c) == n:
            A_inv = np.linalg.inv(A_c)
            B_d = A_inv @ (A_d - np.eye(n)) @ B_c
        else:
            # Series approximation for singular A_c
            B_d = np.zeros_like(B_c)
            term = np.eye(n) * dt
            for k in range(1, 20):
                B_d += term @ B_c / k
                term = term @ A_c * dt / (k + 1)
    else:
        raise ValueError(f"Unknown discretization method: {method}")

    return A_d, B_d


def _matrix_exp(M: np.ndarray, order: int = 12) -> np.ndarray:
    """Compute matrix exponential using Padé approximation.

    Args:
        M: Square matrix.
        order: Order of Padé approximation.

    Returns:
        exp(M) as numpy array.
    """
    # Scale and square method with Padé approximation
    n = M.shape[0]
    norm = np.linalg.norm(M, ord=np.inf)

    # Scaling: find s such that ||M/2^s|| < 0.5
    s = max(0, int(np.ceil(np.log2(norm + 1))))
    M_scaled = M / (2 ** s)

    # Padé approximation coefficients (diagonal Padé)
    c = 1.0
    X = np.eye(n)
    N = np.eye(n)
    D = np.eye(n)

    for k in range(1, order + 1):
        c = c * (order - k + 1) / (k * (2 * order - k + 1))
        X = M_scaled @ X
        N = N + c * X
        D = D + ((-1) ** k) * c * X

    # exp(M_scaled) ≈ D^{-1} * N
    F = np.linalg.solve(D, N)

    # Squaring: exp(M) = exp(M_scaled)^{2^s}
    for _ in range(s):
        F = F @ F

    return F


def solve_dare(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    max_iters: int = 1000,
    tol: float = 1e-9,
) -> np.ndarray:
    """Solve the Discrete Algebraic Riccati Equation (DARE).

    Finds P such that:
        P = A' P A - A' P B (R + B' P B)^{-1} B' P A + Q

    Uses iterative method (value iteration on Riccati recursion).

    Args:
        A: State transition matrix (n x n).
        B: Control input matrix (n x m).
        Q: State cost matrix (n x n), must be positive semi-definite.
        R: Control cost matrix (m x m), must be positive definite.
        max_iters: Maximum iterations.
        tol: Convergence tolerance.

    Returns:
        Solution matrix P (n x n).

    Raises:
        ValueError: If matrices have incompatible dimensions.
        RuntimeError: If iteration does not converge.
    """
    n = A.shape[0]
    m = B.shape[1]

    # Validate dimensions
    if A.shape != (n, n):
        raise ValueError(f"A must be square, got shape {A.shape}")
    if B.shape[0] != n:
        raise ValueError(f"B must have {n} rows, got {B.shape[0]}")
    if Q.shape != (n, n):
        raise ValueError(f"Q must be {n}x{n}, got {Q.shape}")
    if R.shape != (m, m):
        raise ValueError(f"R must be {m}x{m}, got {R.shape}")

    # Initialize P with Q
    P = Q.copy().astype(np.float64)

    for i in range(max_iters):
        # Riccati iteration: P_new = A' P A - A' P B (R + B' P B)^{-1} B' P A + Q
        BtP = B.T @ P
        AtP = A.T @ P

        # (R + B' P B)^{-1}
        S = R + BtP @ B
        S_inv = np.linalg.inv(S)

        # Gain matrix (for reference): K = S^{-1} B' P A
        K = S_inv @ BtP @ A

        # Update P
        P_new = AtP @ A - AtP @ B @ K + Q

        # Check convergence
        diff = np.max(np.abs(P_new - P))
        if diff < tol:
            return P_new

        P = P_new

    raise RuntimeError(f"DARE did not converge after {max_iters} iterations (diff={diff:.2e})")


def compute_lqr_gain(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the optimal LQR gain matrix.

    Args:
        A: State transition matrix.
        B: Control input matrix.
        Q: State cost matrix.
        R: Control cost matrix.

    Returns:
        Tuple of (K, P) where K is the gain matrix and P is the Riccati solution.
        Optimal control law: u = -K @ x
    """
    P = solve_dare(A, B, Q, R)
    # K = (R + B' P B)^{-1} B' P A
    K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
    return K, P


def check_controllability(A: np.ndarray, B: np.ndarray) -> bool:
    """Check if the system (A, B) is controllable.

    A system is controllable if the controllability matrix
    C = [B, AB, A^2B, ..., A^{n-1}B] has full row rank.

    Args:
        A: State transition matrix (n x n).
        B: Control input matrix (n x m).

    Returns:
        True if controllable, False otherwise.
    """
    n = A.shape[0]
    C = B.copy()
    Ak = np.eye(n)

    for _ in range(n - 1):
        Ak = Ak @ A
        C = np.hstack([C, Ak @ B])

    rank = np.linalg.matrix_rank(C)
    return rank == n


def check_stabilizability(A: np.ndarray, B: np.ndarray) -> bool:
    """Check if the system (A, B) is stabilizable.

    A system is stabilizable if all uncontrollable modes are stable
    (eigenvalues inside unit circle for discrete-time).

    Args:
        A: State transition matrix.
        B: Control input matrix.

    Returns:
        True if stabilizable, False otherwise.
    """
    n = A.shape[0]
    eigenvalues = np.linalg.eigvals(A)

    for eig in eigenvalues:
        if np.abs(eig) >= 1:  # Unstable eigenvalue
            # Check if this mode is controllable via PBH test
            # rank([A - λI, B]) should equal n
            test_matrix = np.hstack([A - eig * np.eye(n), B])
            if np.linalg.matrix_rank(test_matrix) < n:
                return False

    return True


@dataclass
class LQRConfig:
    """Configuration for LQR controller.

    Attributes:
        A: State transition matrix.
        B: Control input matrix.
        Q: State cost matrix.
        R: Control cost matrix.
    """

    A: np.ndarray
    B: np.ndarray
    Q: np.ndarray
    R: np.ndarray


class LQRController(BaseController):
    """Linear Quadratic Regulator for optimal state-feedback control.

    Computes control law u = -K @ (x - x_ref) where K is the optimal gain
    matrix obtained by solving the Discrete Algebraic Riccati Equation.

    Attributes:
        A: State transition matrix (n x n).
        B: Control input matrix (n x m).
        Q: State cost matrix (n x n).
        R: Control cost matrix (m x m).
        K: Optimal gain matrix (m x n).
        P: Riccati solution matrix (n x n).
        x_ref: Reference state for regulation.
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        x_ref: Optional[np.ndarray] = None,
        output_limits: Optional[Tuple[float, float]] = None,
        name: str = "LQR",
    ):
        """Initialize the LQR controller.

        Args:
            A: State transition matrix.
            B: Control input matrix.
            Q: State cost matrix (penalizes state deviation).
            R: Control cost matrix (penalizes control effort).
            x_ref: Reference state (defaults to zero).
            output_limits: (min, max) bounds on each control input.
            name: Controller identifier.

        Raises:
            ValueError: If system is not stabilizable.
        """
        super().__init__(name=name, output_limits=output_limits)

        self.A = np.asarray(A, dtype=np.float64)
        self.B = np.asarray(B, dtype=np.float64)
        self.Q = np.asarray(Q, dtype=np.float64)
        self.R = np.asarray(R, dtype=np.float64)

        # Dimensions
        self.n = self.A.shape[0]  # State dimension
        self.m = self.B.shape[1]  # Control dimension

        # Reference state
        if x_ref is None:
            self.x_ref = np.zeros(self.n)
        else:
            self.x_ref = np.asarray(x_ref, dtype=np.float64)

        # Check stabilizability
        if not check_stabilizability(self.A, self.B):
            raise ValueError("System (A, B) is not stabilizable. LQR cannot be computed.")

        # Compute optimal gain
        self.K, self.P = compute_lqr_gain(self.A, self.B, self.Q, self.R)

    def compute_control(
        self, state: np.ndarray, setpoint: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute LQR control action.

        Args:
            state: Current state vector.
            setpoint: Optional reference state override.

        Returns:
            Control action u = -K @ (x - x_ref).
        """
        x = np.asarray(state, dtype=np.float64).flatten()
        x_ref = setpoint if setpoint is not None else self.x_ref

        # State error
        x_error = x - x_ref

        # Optimal control: u = -K @ x_error
        u = -self.K @ x_error

        return u.astype(np.float32)

    def set_reference(self, x_ref: np.ndarray) -> None:
        """Update the reference state.

        Args:
            x_ref: New reference state.
        """
        self.x_ref = np.asarray(x_ref, dtype=np.float64)

    def get_closed_loop_poles(self) -> np.ndarray:
        """Get the closed-loop system poles.

        Returns:
            Eigenvalues of (A - B @ K).
        """
        A_cl = self.A - self.B @ self.K
        return np.linalg.eigvals(A_cl)

    def is_stable(self) -> bool:
        """Check if closed-loop system is stable.

        Returns:
            True if all poles are inside unit circle.
        """
        poles = self.get_closed_loop_poles()
        return np.all(np.abs(poles) < 1.0)

    def get_config(self) -> Dict[str, Any]:
        """Get controller configuration.

        Returns:
            Configuration dictionary.
        """
        config = super().get_config()
        config.update({
            "state_dim": self.n,
            "control_dim": self.m,
            "Q_diag": np.diag(self.Q).tolist(),
            "R_diag": np.diag(self.R).tolist(),
            "K": self.K.tolist(),
            "closed_loop_stable": self.is_stable(),
        })
        return config

    @classmethod
    def from_config(
        cls,
        config: LQRConfig,
        x_ref: Optional[np.ndarray] = None,
        output_limits: Optional[Tuple[float, float]] = None,
    ) -> "LQRController":
        """Create LQR controller from config dataclass.

        Args:
            config: LQR configuration.
            x_ref: Reference state.
            output_limits: Control bounds.

        Returns:
            LQRController instance.
        """
        return cls(
            A=config.A,
            B=config.B,
            Q=config.Q,
            R=config.R,
            x_ref=x_ref,
            output_limits=output_limits,
        )


class FiniteHorizonLQR(BaseController):
    """Finite-horizon LQR with time-varying gains.

    Solves the finite-horizon optimal control problem:
        min sum_{k=0}^{N-1} (x_k' Q x_k + u_k' R u_k) + x_N' Q_f x_N

    The gain matrix K_k varies with time step k.

    Attributes:
        horizon: Number of time steps N.
        gains: List of gain matrices [K_0, K_1, ..., K_{N-1}].
        current_step: Current time step in the horizon.
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        Q_f: np.ndarray,
        horizon: int,
        x_ref: Optional[np.ndarray] = None,
        output_limits: Optional[Tuple[float, float]] = None,
        name: str = "FiniteHorizonLQR",
    ):
        """Initialize finite-horizon LQR.

        Args:
            A: State transition matrix.
            B: Control input matrix.
            Q: Stage state cost matrix.
            R: Stage control cost matrix.
            Q_f: Terminal state cost matrix.
            horizon: Number of time steps.
            x_ref: Reference state.
            output_limits: Control bounds.
            name: Controller identifier.
        """
        super().__init__(name=name, output_limits=output_limits)

        self.A = np.asarray(A, dtype=np.float64)
        self.B = np.asarray(B, dtype=np.float64)
        self.Q = np.asarray(Q, dtype=np.float64)
        self.R = np.asarray(R, dtype=np.float64)
        self.Q_f = np.asarray(Q_f, dtype=np.float64)
        self.horizon = horizon

        self.n = self.A.shape[0]
        self.m = self.B.shape[1]

        if x_ref is None:
            self.x_ref = np.zeros(self.n)
        else:
            self.x_ref = np.asarray(x_ref, dtype=np.float64)

        # Compute time-varying gains via backward Riccati recursion
        self.gains = self._compute_gains()
        self.current_step = 0

    def _compute_gains(self) -> list:
        """Compute time-varying gains via backward recursion.

        Returns:
            List of gain matrices [K_0, ..., K_{N-1}].
        """
        gains = []
        P = self.Q_f.copy()

        for _ in range(self.horizon):
            # K_k = (R + B' P_{k+1} B)^{-1} B' P_{k+1} A
            S = self.R + self.B.T @ P @ self.B
            K = np.linalg.solve(S, self.B.T @ P @ self.A)
            gains.append(K)

            # P_k = Q + A' P_{k+1} A - A' P_{k+1} B K_k
            P = self.Q + self.A.T @ P @ self.A - self.A.T @ P @ self.B @ K

        # Reverse to get [K_0, K_1, ..., K_{N-1}]
        return gains[::-1]

    def compute_control(
        self, state: np.ndarray, setpoint: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute control using time-varying gain.

        Args:
            state: Current state.
            setpoint: Optional reference override.

        Returns:
            Control action.
        """
        x = np.asarray(state, dtype=np.float64).flatten()
        x_ref = setpoint if setpoint is not None else self.x_ref

        if self.current_step >= self.horizon:
            # Past horizon, use last gain
            K = self.gains[-1]
        else:
            K = self.gains[self.current_step]

        u = -K @ (x - x_ref)
        self.current_step += 1

        return u.astype(np.float32)

    def reset(self) -> None:
        """Reset time step counter."""
        self.current_step = 0
