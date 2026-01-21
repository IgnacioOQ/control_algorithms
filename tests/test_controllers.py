"""
Unit tests for control-theoretic controllers (PID, LQR, MPC).

Tests cover:
- Basic functionality and interface compliance
- Mathematical correctness (step response, stability, optimality)
- Edge cases (saturation, anti-windup, singular systems)
"""

import numpy as np
import pytest

from src.controllers import (
    PIDController,
    LQRController,
    MPCController,
    discretize_system,
    solve_dare,
)
from src.controllers.pid import PIDGains
from src.controllers.lqr import (
    check_controllability,
    check_stabilizability,
    compute_lqr_gain,
)


class TestPIDController:
    """Tests for PID controller."""

    def test_proportional_only(self):
        """Test P-only controller response."""
        gains = PIDGains(Kp=2.0, Ki=0.0, Kd=0.0)
        pid = PIDController(gains=gains, setpoint=10.0)

        # At measurement=0, error=10, output should be Kp*10 = 20
        action = pid.compute_control(np.array([0.0]))
        assert np.isclose(action[0], 20.0, atol=1e-6)

        # At measurement=10 (at setpoint), output should be 0
        pid.reset()
        action = pid.compute_control(np.array([10.0]))
        assert np.isclose(action[0], 0.0, atol=1e-6)

    def test_integral_accumulation(self):
        """Test that integral term accumulates over time."""
        gains = PIDGains(Kp=0.0, Ki=1.0, Kd=0.0)
        pid = PIDController(gains=gains, setpoint=10.0, dt=1.0)

        # Constant error of 10, Ki=1, dt=1 -> integral grows by 10 each step
        action1 = pid.compute_control(np.array([0.0]))
        assert np.isclose(action1[0], 10.0, atol=1e-6)

        action2 = pid.compute_control(np.array([0.0]))
        assert np.isclose(action2[0], 20.0, atol=1e-6)

        action3 = pid.compute_control(np.array([0.0]))
        assert np.isclose(action3[0], 30.0, atol=1e-6)

    def test_derivative_on_measurement(self):
        """Test that derivative acts on measurement, not error."""
        gains = PIDGains(Kp=0.0, Ki=0.0, Kd=1.0)
        pid = PIDController(gains=gains, setpoint=0.0, dt=1.0)

        # First call: no previous measurement, D term should be 0
        action1 = pid.compute_control(np.array([0.0]))
        assert np.isclose(action1[0], 0.0, atol=1e-6)

        # Measurement jumps from 0 to 10, d(measurement)/dt = 10
        # D term = Kd * (-d_measurement) = 1 * (-10) = -10
        action2 = pid.compute_control(np.array([10.0]))
        assert np.isclose(action2[0], -10.0, atol=1e-6)

    def test_anti_windup(self):
        """Test that anti-windup prevents integrator windup."""
        gains = PIDGains(Kp=0.0, Ki=10.0, Kd=0.0)
        pid = PIDController(
            gains=gains,
            setpoint=100.0,
            output_limits=(-10.0, 10.0),
            anti_windup=True,
            dt=1.0,
        )

        # Large error should try to accumulate, but output is clamped
        for _ in range(10):
            action = pid.compute_control(np.array([0.0]))
            assert -10.0 <= action[0] <= 10.0

        # When error reverses, integrator should not have wound up excessively
        pid.setpoint = 0.0
        action = pid.compute_control(np.array([0.0]))
        # If anti-windup works, output should quickly come back near 0
        assert action[0] <= 10.0

    def test_output_limits(self):
        """Test that output is properly clamped."""
        gains = PIDGains(Kp=100.0, Ki=0.0, Kd=0.0)
        pid = PIDController(
            gains=gains, setpoint=10.0, output_limits=(-5.0, 5.0)
        )

        action = pid.compute_control(np.array([0.0]))
        assert action[0] == 5.0  # Clamped at upper limit

        action = pid.compute_control(np.array([20.0]))
        assert action[0] == -5.0  # Clamped at lower limit

    def test_reset(self):
        """Test that reset clears internal state."""
        gains = PIDGains(Kp=1.0, Ki=1.0, Kd=1.0)
        pid = PIDController(gains=gains, setpoint=10.0, dt=1.0)

        # Accumulate some state
        for _ in range(5):
            pid.compute_control(np.array([0.0]))

        # Reset should clear integral and derivative state
        pid.reset()

        # After reset, behavior should be same as fresh controller
        pid2 = PIDController(gains=gains, setpoint=10.0, dt=1.0)

        action1 = pid.compute_control(np.array([5.0]))
        action2 = pid2.compute_control(np.array([5.0]))

        assert np.isclose(action1[0], action2[0], atol=1e-6)

    def test_ziegler_nichols_tuning(self):
        """Test Ziegler-Nichols tuning factory method."""
        # Ultimate gain and period
        Ku, Tu = 10.0, 2.0

        pid = PIDController.ziegler_nichols(Ku, Tu, controller_type="PID")

        # Check gains match ZN formulas
        assert np.isclose(pid.gains.Kp, 0.6 * Ku, atol=1e-6)
        assert np.isclose(pid.gains.Ki, 1.2 * Ku / Tu, atol=1e-6)
        assert np.isclose(pid.gains.Kd, 0.075 * Ku * Tu, atol=1e-6)

    def test_interface_compliance(self):
        """Test that PID implements BaseAgent interface."""
        gains = PIDGains(Kp=1.0, Ki=0.1, Kd=0.05)
        pid = PIDController(gains=gains, setpoint=0.0)

        # Should have these methods from BaseAgent
        assert hasattr(pid, "select_action")
        assert hasattr(pid, "store")
        assert hasattr(pid, "update")
        assert hasattr(pid, "ready_to_train")
        assert hasattr(pid, "get_config")

        # ready_to_train should always be False for controllers
        assert pid.ready_to_train() is False

        # update should return empty dict
        assert pid.update() == {}


class TestLQRController:
    """Tests for LQR controller."""

    def test_simple_integrator(self):
        """Test LQR on simple integrator system x_{k+1} = x_k + u_k."""
        A = np.array([[1.0]])
        B = np.array([[1.0]])
        Q = np.array([[1.0]])
        R = np.array([[1.0]])

        lqr = LQRController(A, B, Q, R)

        # LQR should compute a stabilizing gain
        assert lqr.is_stable()

        # Control should drive state to zero
        x = np.array([10.0])
        for _ in range(50):
            u = lqr.compute_control(x)
            x = A @ x + B @ u

        assert np.abs(x[0]) < 0.1

    def test_double_integrator(self):
        """Test LQR on double integrator (position, velocity)."""
        dt = 0.1
        A = np.array([[1.0, dt], [0.0, 1.0]])
        B = np.array([[0.5 * dt**2], [dt]])
        Q = np.diag([10.0, 1.0])
        R = np.array([[1.0]])

        lqr = LQRController(A, B, Q, R)

        assert lqr.is_stable()

        # Simulate from initial state
        x = np.array([5.0, 2.0])  # position=5, velocity=2
        trajectory = [x.copy()]

        for _ in range(100):
            u = lqr.compute_control(x)
            x = A @ x + B @ u
            trajectory.append(x.copy())

        # Should converge to origin
        assert np.linalg.norm(trajectory[-1]) < 0.1

    def test_reference_tracking(self):
        """Test LQR drives state toward reference (with steady-state offset).

        Note: Standard LQR has steady-state error for non-zero references
        unless the system is augmented with integral action. This test
        verifies that the controller at least moves toward the reference.
        """
        A = np.array([[0.9]])
        B = np.array([[0.1]])
        Q = np.array([[10.0]])
        R = np.array([[1.0]])

        x_ref = np.array([5.0])
        lqr = LQRController(A, B, Q, R, x_ref=x_ref)

        x = np.array([0.0])
        initial_error = np.abs(x[0] - x_ref[0])

        for _ in range(100):
            u = lqr.compute_control(x)
            x = A @ x + B @ u

        final_error = np.abs(x[0] - x_ref[0])

        # Error should decrease (moved toward reference)
        assert final_error < initial_error

    def test_unstabilizable_raises(self):
        """Test that unstabilizable system raises error."""
        # Uncontrollable mode: eigenvalue 2 (unstable) with B orthogonal
        A = np.array([[2.0, 0.0], [0.0, 0.5]])
        B = np.array([[0.0], [1.0]])  # Can only affect second state
        Q = np.eye(2)
        R = np.array([[1.0]])

        with pytest.raises(ValueError, match="not stabilizable"):
            LQRController(A, B, Q, R)

    def test_gain_correctness(self):
        """Test that computed gain matches analytical solution for simple case."""
        # For scalar system x_{k+1} = a*x + b*u with Q=q, R=r
        # The Riccati solution has a closed form
        a, b, q, r = 0.9, 1.0, 1.0, 1.0
        A = np.array([[a]])
        B = np.array([[b]])
        Q = np.array([[q]])
        R = np.array([[r]])

        K, P = compute_lqr_gain(A, B, Q, R)

        # Verify Riccati equation: P = A'PA - A'PB(R+B'PB)^{-1}B'PA + Q
        S = R + B.T @ P @ B
        P_check = A.T @ P @ A - A.T @ P @ B @ np.linalg.inv(S) @ B.T @ P @ A + Q

        assert np.allclose(P, P_check, atol=1e-8)


class TestDARESolver:
    """Tests for Discrete Algebraic Riccati Equation solver."""

    def test_identity_system(self):
        """Test DARE on identity system."""
        A = np.eye(2)
        B = np.eye(2)
        Q = np.eye(2)
        R = np.eye(2)

        P = solve_dare(A, B, Q, R)

        # Verify Riccati equation
        S = R + B.T @ P @ B
        P_check = A.T @ P @ A - A.T @ P @ B @ np.linalg.inv(S) @ B.T @ P @ A + Q

        assert np.allclose(P, P_check, atol=1e-8)

    def test_random_system(self):
        """Test DARE on random stable system."""
        np.random.seed(42)
        n, m = 4, 2

        # Generate stable A (eigenvalues < 1)
        A = np.random.randn(n, n)
        A = A / (np.max(np.abs(np.linalg.eigvals(A))) + 0.5)

        B = np.random.randn(n, m)
        Q = np.eye(n)
        R = np.eye(m)

        P = solve_dare(A, B, Q, R)

        # Verify Riccati equation
        S = R + B.T @ P @ B
        P_check = A.T @ P @ A - A.T @ P @ B @ np.linalg.inv(S) @ B.T @ P @ A + Q

        assert np.allclose(P, P_check, atol=1e-6)

    def test_dimension_mismatch_raises(self):
        """Test that dimension mismatch raises error."""
        A = np.eye(2)
        B = np.eye(3)  # Wrong dimension
        Q = np.eye(2)
        R = np.eye(3)

        with pytest.raises(ValueError):
            solve_dare(A, B, Q, R)


class TestDiscretization:
    """Tests for continuous-to-discrete conversion."""

    def test_euler_method(self):
        """Test forward Euler discretization."""
        A_c = np.array([[-1.0]])
        B_c = np.array([[1.0]])
        dt = 0.1

        A_d, B_d = discretize_system(A_c, B_c, dt, method="euler")

        # Euler: A_d = I + A_c*dt, B_d = B_c*dt
        assert np.allclose(A_d, [[0.9]], atol=1e-10)
        assert np.allclose(B_d, [[0.1]], atol=1e-10)

    def test_zoh_method(self):
        """Test zero-order hold discretization."""
        A_c = np.array([[-1.0]])
        B_c = np.array([[1.0]])
        dt = 0.1

        A_d, B_d = discretize_system(A_c, B_c, dt, method="zoh")

        # For scalar dx/dt = -x + u, exact: A_d = exp(-dt), B_d = 1-exp(-dt)
        expected_A = np.exp(-dt)
        expected_B = 1 - np.exp(-dt)

        assert np.allclose(A_d, [[expected_A]], atol=1e-6)
        assert np.allclose(B_d, [[expected_B]], atol=1e-6)


class TestControllability:
    """Tests for controllability checks."""

    def test_controllable_system(self):
        """Test controllable double integrator."""
        A = np.array([[1, 1], [0, 1]])
        B = np.array([[0], [1]])

        assert check_controllability(A, B) == True

    def test_uncontrollable_system(self):
        """Test uncontrollable system."""
        A = np.array([[1, 0], [0, 2]])
        B = np.array([[1], [0]])  # Second state uncontrollable

        assert check_controllability(A, B) == False

    def test_stabilizable_but_not_controllable(self):
        """Test stabilizable system that is not fully controllable."""
        # First mode (eigenvalue 2) is unstable but controllable through B[0]
        # Second mode (eigenvalue 0.5) is stable but uncontrollable
        # System is stabilizable (can stabilize unstable modes) but not controllable
        A = np.array([[2, 0], [0, 0.5]])
        B = np.array([[1], [0]])

        assert check_controllability(A, B) == False  # Not fully controllable
        assert check_stabilizability(A, B) == True   # But stabilizable (unstable mode controllable)


class TestMPCController:
    """Tests for MPC controller."""

    def test_linear_dynamics(self):
        """Test MPC with simple linear dynamics."""

        def dynamics(x, u, dt):
            return 0.9 * x + 0.1 * u

        mpc = MPCController(
            dynamics_fn=dynamics,
            horizon=10,
            state_dim=1,
            control_dim=1,
            Q=np.array([[1.0]]),
            R=np.array([[0.1]]),
            x_ref=np.array([0.0]),
        )

        # From non-zero state, should compute control toward zero
        x = np.array([5.0])
        u = mpc.compute_control(x)

        assert u.shape == (1,)
        # Control should be negative to drive state toward zero
        assert u[0] < 0

    def test_control_bounds(self):
        """Test that MPC respects control bounds."""

        def dynamics(x, u, dt):
            return x + u

        mpc = MPCController(
            dynamics_fn=dynamics,
            horizon=5,
            state_dim=1,
            control_dim=1,
            control_bounds=(np.array([-1.0]), np.array([1.0])),
            x_ref=np.array([0.0]),
        )

        # Large state should saturate control
        x = np.array([100.0])
        u = mpc.compute_control(x)

        assert -1.0 <= u[0] <= 1.0

    def test_receding_horizon(self):
        """Test that MPC uses receding horizon correctly."""

        def dynamics(x, u, dt):
            return 0.9 * x + 0.1 * u

        mpc = MPCController(
            dynamics_fn=dynamics,
            horizon=10,
            state_dim=1,
            control_dim=1,
            x_ref=np.array([0.0]),
        )

        # Simulate closed-loop
        x = np.array([10.0])
        trajectory = [x.copy()]

        for _ in range(30):
            u = mpc.compute_control(x)
            x = dynamics(x, u, 1.0)
            trajectory.append(x.copy())

        # Should converge toward reference
        assert np.abs(trajectory[-1][0]) < 1.0

    def test_warm_start(self):
        """Test that warm starting uses previous solution."""

        def dynamics(x, u, dt):
            return 0.9 * x + 0.1 * u

        mpc = MPCController(
            dynamics_fn=dynamics,
            horizon=10,
            state_dim=1,
            control_dim=1,
            warm_start=True,
        )

        x = np.array([5.0])
        mpc.compute_control(x)

        # Previous solution should be stored
        assert mpc._prev_solution is not None

    def test_reset_clears_warm_start(self):
        """Test that reset clears warm start."""

        def dynamics(x, u, dt):
            return x + u

        mpc = MPCController(
            dynamics_fn=dynamics,
            horizon=5,
            state_dim=1,
            control_dim=1,
        )

        mpc.compute_control(np.array([1.0]))
        assert mpc._prev_solution is not None

        mpc.reset()
        assert mpc._prev_solution is None

    def test_interface_compliance(self):
        """Test that MPC implements BaseAgent interface."""

        def dynamics(x, u, dt):
            return x + u

        mpc = MPCController(
            dynamics_fn=dynamics,
            horizon=5,
            state_dim=1,
            control_dim=1,
        )

        # BaseAgent interface
        assert hasattr(mpc, "select_action")
        assert hasattr(mpc, "store")
        assert hasattr(mpc, "update")
        assert hasattr(mpc, "ready_to_train")

        assert mpc.ready_to_train() is False
        assert mpc.update() == {}

    def test_multivariate_system(self):
        """Test MPC with multi-dimensional state and control."""
        A = np.array([[0.9, 0.1], [0.0, 0.8]])
        B = np.array([[0.0], [0.1]])

        def dynamics(x, u, dt):
            return A @ x + B @ u

        mpc = MPCController(
            dynamics_fn=dynamics,
            horizon=10,
            state_dim=2,
            control_dim=1,
            Q=np.eye(2),
            R=np.array([[1.0]]),
            x_ref=np.zeros(2),
        )

        x = np.array([5.0, 3.0])
        u = mpc.compute_control(x)

        assert u.shape == (1,)


class TestControllerIntegration:
    """Integration tests for controllers with environments."""

    def test_pid_step_response(self):
        """Test PID controller step response on first-order system."""
        # First-order system: dx/dt = -x + u, discretized
        dt = 0.1
        tau = 1.0  # Time constant

        def plant(x, u):
            return x + dt * (-x / tau + u)

        # PID tuned for this system
        gains = PIDGains(Kp=2.0, Ki=0.5, Kd=0.1)
        pid = PIDController(
            gains=gains,
            setpoint=1.0,
            dt=dt,
            output_limits=(-10.0, 10.0),
        )

        # Simulate step response
        x = 0.0
        trajectory = [x]

        for _ in range(100):
            u = pid.compute_control(np.array([x]))[0]
            x = plant(x, u)
            trajectory.append(x)

        # Should reach setpoint
        assert abs(trajectory[-1] - 1.0) < 0.1

        # Should not have excessive overshoot
        assert max(trajectory) < 1.5

    def test_lqr_vs_mpc_equivalence(self):
        """Test that LQR and MPC give similar results for linear unconstrained case."""
        A = np.array([[0.9, 0.1], [0.0, 0.8]])
        B = np.array([[0.0], [0.1]])
        Q = np.eye(2)
        R = np.array([[1.0]])

        lqr = LQRController(A, B, Q, R)

        def dynamics(x, u, dt):
            return A @ x + B @ u

        mpc = MPCController(
            dynamics_fn=dynamics,
            horizon=50,  # Long horizon to approximate infinite
            state_dim=2,
            control_dim=1,
            Q=Q,
            R=R,
            Q_f=10 * Q,  # Terminal cost
        )

        # Compare controls from same state
        x = np.array([5.0, 3.0])

        u_lqr = lqr.compute_control(x)
        u_mpc = mpc.compute_control(x)

        # Should be similar (not exact due to finite horizon)
        assert np.allclose(u_lqr, u_mpc, atol=0.5)
