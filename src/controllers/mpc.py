"""
Model Predictive Control (MPC) with receding horizon optimization.

Implements nonlinear MPC using sequential optimization:
    min_{u_0,...,u_{N-1}} sum_{k=0}^{N-1} l(x_k, u_k) + V_f(x_N)
    s.t. x_{k+1} = f(x_k, u_k)
         x_k in X, u_k in U
         x_0 = current_state

Features:
- Nonlinear dynamics via user-provided function
- State and input constraints
- Quadratic and custom cost functions
- Multiple solver backends (scipy, simple gradient descent)
- Warm-starting from previous solution

Suitable for:
- Homeostasis environment (glucose regulation with Bergman model)
- Smart Grid environment (economic dispatch)
- Any constrained optimal control problem
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import minimize, Bounds

from .base import BaseController


@dataclass
class MPCConfig:
    """Configuration for MPC controller.

    Attributes:
        horizon: Prediction horizon N.
        state_dim: Dimension of state vector.
        control_dim: Dimension of control vector.
        dt: Time step for dynamics.
        state_bounds: (lower, upper) bounds on state.
        control_bounds: (lower, upper) bounds on control.
        Q: State cost matrix (or diagonal).
        R: Control cost matrix (or diagonal).
        Q_f: Terminal cost matrix.
        x_ref: Reference state trajectory or single state.
    """

    horizon: int = 10
    state_dim: int = 1
    control_dim: int = 1
    dt: float = 1.0
    state_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    control_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    Q: Optional[np.ndarray] = None
    R: Optional[np.ndarray] = None
    Q_f: Optional[np.ndarray] = None
    x_ref: Optional[np.ndarray] = None


class MPCController(BaseController):
    """Model Predictive Control with receding horizon.

    Solves a finite-horizon optimal control problem at each time step,
    applies the first control action, then re-solves at the next step.

    Attributes:
        dynamics_fn: Function x_{k+1} = f(x_k, u_k, dt).
        cost_fn: Stage cost l(x, u, x_ref) (optional, uses quadratic if None).
        terminal_cost_fn: Terminal cost V_f(x_N, x_ref) (optional).
        horizon: Prediction horizon N.
        state_dim: State dimension.
        control_dim: Control dimension.
        Q: State cost matrix.
        R: Control cost matrix.
        Q_f: Terminal state cost matrix.
    """

    def __init__(
        self,
        dynamics_fn: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
        horizon: int,
        state_dim: int,
        control_dim: int,
        dt: float = 1.0,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        Q_f: Optional[np.ndarray] = None,
        x_ref: Optional[np.ndarray] = None,
        state_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        control_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        cost_fn: Optional[Callable] = None,
        terminal_cost_fn: Optional[Callable] = None,
        solver: str = "SLSQP",
        max_iter: int = 100,
        warm_start: bool = True,
        name: str = "MPC",
    ):
        """Initialize the MPC controller.

        Args:
            dynamics_fn: System dynamics x_{k+1} = f(x_k, u_k, dt).
            horizon: Prediction horizon (number of steps).
            state_dim: Dimension of state vector.
            control_dim: Dimension of control input.
            dt: Time step for dynamics integration.
            Q: State cost matrix (n x n), default identity.
            R: Control cost matrix (m x m), default identity.
            Q_f: Terminal cost matrix (n x n), default 10*Q.
            x_ref: Reference state (n,) or trajectory (N+1, n).
            state_bounds: Tuple of (lower, upper) state bounds.
            control_bounds: Tuple of (lower, upper) control bounds.
            cost_fn: Custom stage cost function l(x, u, x_ref) -> float.
            terminal_cost_fn: Custom terminal cost V_f(x, x_ref) -> float.
            solver: Scipy solver method ("SLSQP", "trust-constr", etc.).
            max_iter: Maximum solver iterations.
            warm_start: Use previous solution as initial guess.
            name: Controller identifier.
        """
        # Determine output limits from control bounds
        output_limits = None
        if control_bounds is not None:
            output_limits = (
                float(control_bounds[0].min()),
                float(control_bounds[1].max()),
            )

        super().__init__(name=name, output_limits=output_limits)

        self.dynamics_fn = dynamics_fn
        self.horizon = horizon
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.dt = dt
        self.solver = solver
        self.max_iter = max_iter
        self.warm_start = warm_start

        # Cost matrices (default to identity)
        self.Q = Q if Q is not None else np.eye(state_dim)
        self.R = R if R is not None else np.eye(control_dim)
        self.Q_f = Q_f if Q_f is not None else 10 * self.Q

        # Reference state
        if x_ref is None:
            self.x_ref = np.zeros(state_dim)
        else:
            self.x_ref = np.asarray(x_ref)

        # Bounds
        self.state_bounds = state_bounds
        self.control_bounds = control_bounds

        # Custom cost functions
        self.cost_fn = cost_fn
        self.terminal_cost_fn = terminal_cost_fn

        # Warm start storage
        self._prev_solution = None

    def _get_reference(self, k: int) -> np.ndarray:
        """Get reference state at step k.

        Args:
            k: Time step index.

        Returns:
            Reference state vector.
        """
        if self.x_ref.ndim == 1:
            return self.x_ref
        else:
            # Time-varying reference
            idx = min(k, len(self.x_ref) - 1)
            return self.x_ref[idx]

    def _stage_cost(self, x: np.ndarray, u: np.ndarray, k: int) -> float:
        """Compute stage cost l(x, u).

        Args:
            x: State vector.
            u: Control vector.
            k: Time step.

        Returns:
            Stage cost value.
        """
        x_ref = self._get_reference(k)

        if self.cost_fn is not None:
            return self.cost_fn(x, u, x_ref)

        # Default quadratic cost
        x_err = x - x_ref
        return float(x_err @ self.Q @ x_err + u @ self.R @ u)

    def _terminal_cost(self, x: np.ndarray) -> float:
        """Compute terminal cost V_f(x_N).

        Args:
            x: Terminal state.

        Returns:
            Terminal cost value.
        """
        x_ref = self._get_reference(self.horizon)

        if self.terminal_cost_fn is not None:
            return self.terminal_cost_fn(x, x_ref)

        # Default quadratic terminal cost
        x_err = x - x_ref
        return float(x_err @ self.Q_f @ x_err)

    def _rollout(
        self, x0: np.ndarray, u_sequence: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Simulate system with control sequence.

        Args:
            x0: Initial state.
            u_sequence: Control sequence (N, m).

        Returns:
            Tuple of (state trajectory (N+1, n), total cost).
        """
        N = self.horizon
        x_traj = np.zeros((N + 1, self.state_dim))
        x_traj[0] = x0

        total_cost = 0.0
        x = x0.copy()

        for k in range(N):
            u = u_sequence[k]
            total_cost += self._stage_cost(x, u, k)
            x = self.dynamics_fn(x, u, self.dt)
            x_traj[k + 1] = x

        total_cost += self._terminal_cost(x)

        return x_traj, total_cost

    def _objective(self, u_flat: np.ndarray, x0: np.ndarray) -> float:
        """Objective function for optimizer.

        Args:
            u_flat: Flattened control sequence (N * m,).
            x0: Initial state.

        Returns:
            Total cost.
        """
        u_sequence = u_flat.reshape(self.horizon, self.control_dim)
        _, cost = self._rollout(x0, u_sequence)
        return cost

    def _objective_with_grad(
        self, u_flat: np.ndarray, x0: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Objective with numerical gradient.

        Args:
            u_flat: Flattened control sequence.
            x0: Initial state.

        Returns:
            Tuple of (cost, gradient).
        """
        cost = self._objective(u_flat, x0)

        # Numerical gradient
        eps = 1e-6
        grad = np.zeros_like(u_flat)
        for i in range(len(u_flat)):
            u_plus = u_flat.copy()
            u_plus[i] += eps
            cost_plus = self._objective(u_plus, x0)
            grad[i] = (cost_plus - cost) / eps

        return cost, grad

    def _state_constraint(
        self, u_flat: np.ndarray, x0: np.ndarray
    ) -> np.ndarray:
        """Compute state constraint violations.

        Returns array where positive values indicate violations.
        """
        if self.state_bounds is None:
            return np.array([0.0])

        u_sequence = u_flat.reshape(self.horizon, self.control_dim)
        x_traj, _ = self._rollout(x0, u_sequence)

        violations = []
        for x in x_traj:
            # Lower bound violations: lb - x <= 0
            violations.extend(self.state_bounds[0] - x)
            # Upper bound violations: x - ub <= 0
            violations.extend(x - self.state_bounds[1])

        return np.array(violations)

    def _solve_ocp(self, x0: np.ndarray) -> np.ndarray:
        """Solve the optimal control problem.

        Args:
            x0: Current state.

        Returns:
            Optimal control sequence (N, m).
        """
        N = self.horizon
        m = self.control_dim

        # Initial guess
        if self.warm_start and self._prev_solution is not None:
            # Shift previous solution and append last control
            u_init = np.vstack([
                self._prev_solution[1:],
                self._prev_solution[-1:]
            ])
        else:
            u_init = np.zeros((N, m))

        u_flat = u_init.flatten()

        # Control bounds
        if self.control_bounds is not None:
            lb = np.tile(self.control_bounds[0], N)
            ub = np.tile(self.control_bounds[1], N)
            bounds = Bounds(lb, ub)
        else:
            bounds = None

        # Constraints
        constraints = []
        if self.state_bounds is not None:
            constraints.append({
                "type": "ineq",
                "fun": lambda u, x0=x0: -self._state_constraint(u, x0),
            })

        # Solve
        result = minimize(
            self._objective,
            u_flat,
            args=(x0,),
            method=self.solver,
            bounds=bounds,
            constraints=constraints if constraints else None,
            options={"maxiter": self.max_iter, "disp": False},
        )

        u_opt = result.x.reshape(N, m)

        # Store for warm start
        self._prev_solution = u_opt

        return u_opt

    def compute_control(
        self, state: np.ndarray, setpoint: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute MPC control action.

        Solves the OCP and returns the first control action.

        Args:
            state: Current state vector.
            setpoint: Optional reference override.

        Returns:
            First control action from optimal sequence.
        """
        x0 = np.asarray(state, dtype=np.float64).flatten()

        # Update reference if provided
        if setpoint is not None:
            self.x_ref = np.asarray(setpoint)

        # Solve optimal control problem
        u_sequence = self._solve_ocp(x0)

        # Return first control action (receding horizon)
        return u_sequence[0].astype(np.float32)

    def get_planned_trajectory(
        self, state: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get the full planned state and control trajectory.

        Useful for visualization and debugging.

        Args:
            state: Current state.

        Returns:
            Tuple of (state trajectory (N+1, n), control sequence (N, m)).
        """
        x0 = np.asarray(state, dtype=np.float64).flatten()
        u_sequence = self._solve_ocp(x0)
        x_traj, _ = self._rollout(x0, u_sequence)
        return x_traj, u_sequence

    def reset(self) -> None:
        """Reset warm start storage."""
        self._prev_solution = None

    def set_reference(self, x_ref: np.ndarray) -> None:
        """Update reference state/trajectory.

        Args:
            x_ref: New reference (single state or trajectory).
        """
        self.x_ref = np.asarray(x_ref)

    def get_config(self) -> Dict[str, Any]:
        """Get controller configuration.

        Returns:
            Configuration dictionary.
        """
        config = super().get_config()
        config.update({
            "horizon": self.horizon,
            "state_dim": self.state_dim,
            "control_dim": self.control_dim,
            "dt": self.dt,
            "solver": self.solver,
            "max_iter": self.max_iter,
            "warm_start": self.warm_start,
            "Q_diag": np.diag(self.Q).tolist(),
            "R_diag": np.diag(self.R).tolist(),
        })
        return config


class LinearMPC(MPCController):
    """MPC specialized for linear systems.

    For linear dynamics x_{k+1} = A*x + B*u, the optimization
    problem becomes a quadratic program which can be solved more efficiently.

    Attributes:
        A: State transition matrix.
        B: Control input matrix.
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        horizon: int,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        Q_f: Optional[np.ndarray] = None,
        x_ref: Optional[np.ndarray] = None,
        state_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        control_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        name: str = "LinearMPC",
    ):
        """Initialize Linear MPC.

        Args:
            A: State transition matrix (n x n).
            B: Control input matrix (n x m).
            horizon: Prediction horizon.
            Q: State cost matrix.
            R: Control cost matrix.
            Q_f: Terminal cost matrix.
            x_ref: Reference state.
            state_bounds: State constraints.
            control_bounds: Control constraints.
            name: Controller identifier.
        """
        self.A = np.asarray(A, dtype=np.float64)
        self.B = np.asarray(B, dtype=np.float64)

        state_dim = A.shape[0]
        control_dim = B.shape[1]

        def linear_dynamics(x, u, dt):
            return self.A @ x + self.B @ u

        super().__init__(
            dynamics_fn=linear_dynamics,
            horizon=horizon,
            state_dim=state_dim,
            control_dim=control_dim,
            dt=1.0,  # Already discrete
            Q=Q,
            R=R,
            Q_f=Q_f,
            x_ref=x_ref,
            state_bounds=state_bounds,
            control_bounds=control_bounds,
            name=name,
        )

    def _build_qp_matrices(
        self, x0: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build QP matrices for efficient solving.

        The problem can be written as:
            min (1/2) u' H u + f' u
            s.t. A_ineq u <= b_ineq

        Args:
            x0: Initial state.

        Returns:
            Tuple of (H, f, A_ineq, b_ineq).
        """
        N = self.horizon
        n = self.state_dim
        m = self.control_dim

        # Build prediction matrices: x = Px * x0 + Pu * u
        Px = np.zeros((N * n, n))
        Pu = np.zeros((N * n, N * m))

        A_power = np.eye(n)
        for k in range(N):
            A_power = A_power @ self.A
            Px[k * n:(k + 1) * n, :] = A_power

            for j in range(k + 1):
                A_diff = np.linalg.matrix_power(self.A, k - j)
                Pu[k * n:(k + 1) * n, j * m:(j + 1) * m] = A_diff @ self.B

        # Build block diagonal cost matrices
        Q_bar = np.kron(np.eye(N - 1), self.Q)
        Q_bar = np.block([
            [Q_bar, np.zeros((Q_bar.shape[0], n))],
            [np.zeros((n, Q_bar.shape[1])), self.Q_f]
        ])
        R_bar = np.kron(np.eye(N), self.R)

        # Reference trajectory
        if self.x_ref.ndim == 1:
            x_ref_stacked = np.tile(self.x_ref, N)
        else:
            x_ref_stacked = self.x_ref[:N].flatten()

        # QP cost: (1/2) u' H u + f' u
        H = Pu.T @ Q_bar @ Pu + R_bar
        f = Pu.T @ Q_bar @ (Px @ x0 - x_ref_stacked)

        return H, f, Px, Pu
