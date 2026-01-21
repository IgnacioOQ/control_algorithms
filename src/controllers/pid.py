"""
PID Controller with anti-windup and derivative filtering.

Implements a discrete-time PID controller suitable for:
- Homeostasis environment (glucose regulation via insulin)
- Any setpoint tracking control problem

Features:
- Derivative-on-measurement (avoids derivative kick on setpoint changes)
- Integrator anti-windup via clamping and back-calculation
- Optional derivative filtering (low-pass filter on D term)
- Configurable setpoint weighting
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .base import BaseController


@dataclass
class PIDGains:
    """PID controller gains.

    Attributes:
        Kp: Proportional gain.
        Ki: Integral gain.
        Kd: Derivative gain.
    """

    Kp: float = 1.0
    Ki: float = 0.0
    Kd: float = 0.0


class PIDController(BaseController):
    """Discrete-time PID controller with anti-windup.

    Implements the velocity (incremental) form of PID:
        u(k) = u(k-1) + Kp*(e(k) - e(k-1)) + Ki*e(k) + Kd*(e(k) - 2*e(k-1) + e(k-2))

    Or equivalently, the positional form with integrator:
        P = Kp * e(k)
        I = I + Ki * e(k) * dt
        D = Kd * (y(k-1) - y(k)) / dt  (derivative on measurement)
        u = P + I + D

    Attributes:
        gains: PID gains (Kp, Ki, Kd).
        dt: Sampling time (for proper scaling).
        setpoint: Target value for regulation.
        output_limits: (min, max) bounds on control output.
        anti_windup: Whether to use anti-windup.
        derivative_filter: Time constant for derivative low-pass filter (0 = no filter).
        setpoint_weight_b: Setpoint weight for proportional term (0-1).
        setpoint_weight_c: Setpoint weight for derivative term (0-1).
    """

    def __init__(
        self,
        gains: PIDGains,
        dt: float = 1.0,
        setpoint: float = 0.0,
        output_limits: Optional[Tuple[float, float]] = None,
        anti_windup: bool = True,
        derivative_filter: float = 0.0,
        setpoint_weight_b: float = 1.0,
        setpoint_weight_c: float = 0.0,
        name: str = "PID",
    ):
        """Initialize the PID controller.

        Args:
            gains: PID gains dataclass.
            dt: Sampling/control time step.
            setpoint: Target value.
            output_limits: (min, max) control output bounds.
            anti_windup: Enable integrator anti-windup.
            derivative_filter: Derivative filter time constant (0 disables).
            setpoint_weight_b: Proportional setpoint weight (0=no setpoint, 1=full).
            setpoint_weight_c: Derivative setpoint weight (0=measurement only).
            name: Controller identifier.
        """
        super().__init__(
            name=name, setpoint=setpoint, output_limits=output_limits
        )
        self.gains = gains
        self.dt = dt
        self.anti_windup = anti_windup
        self.derivative_filter = derivative_filter
        self.setpoint_weight_b = setpoint_weight_b
        self.setpoint_weight_c = setpoint_weight_c

        # Internal state
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_measurement = None
        self._prev_derivative = 0.0
        self._prev_output = 0.0

    def reset(self) -> None:
        """Reset controller internal state."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_measurement = None
        self._prev_derivative = 0.0
        self._prev_output = 0.0

    def compute_control(
        self, state: np.ndarray, setpoint: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute PID control action.

        Args:
            state: Current measurement (scalar or array, uses first element).
            setpoint: Optional setpoint override.

        Returns:
            Control action as numpy array.
        """
        # Extract scalar measurement
        if isinstance(state, np.ndarray):
            measurement = float(state.flat[0])
        else:
            measurement = float(state)

        # Use provided setpoint or stored one
        sp = float(setpoint) if setpoint is not None else float(self.setpoint)

        # Compute error
        error = sp - measurement

        # Proportional term (with setpoint weighting)
        # P = Kp * (b*sp - y) where b is setpoint_weight_b
        p_error = self.setpoint_weight_b * sp - measurement
        p_term = self.gains.Kp * p_error

        # Integral term
        self._integral += self.gains.Ki * error * self.dt
        i_term = self._integral

        # Derivative term (on measurement to avoid derivative kick)
        # D = Kd * d(c*sp - y)/dt where c is setpoint_weight_c
        if self._prev_measurement is None:
            d_term = 0.0
        else:
            # Derivative on measurement (negative because we want d(-y)/dt)
            d_measurement = (measurement - self._prev_measurement) / self.dt
            # Derivative on setpoint (if weight > 0)
            d_setpoint = 0.0  # Assuming setpoint is constant
            derivative = self.setpoint_weight_c * d_setpoint - d_measurement

            # Optional derivative filtering (first-order low-pass)
            if self.derivative_filter > 0:
                alpha = self.dt / (self.derivative_filter + self.dt)
                derivative = alpha * derivative + (1 - alpha) * self._prev_derivative

            d_term = self.gains.Kd * derivative
            self._prev_derivative = derivative

        # Compute raw output
        output = p_term + i_term + d_term

        # Anti-windup: clamp output and back-calculate integrator
        if self.output_limits is not None and self.anti_windup:
            output_clamped = np.clip(
                output, self.output_limits[0], self.output_limits[1]
            )
            # Back-calculation: reduce integrator by the amount we were saturated
            if output != output_clamped:
                self._integral -= (output - output_clamped)
            output = output_clamped

        # Update state for next iteration
        self._prev_error = error
        self._prev_measurement = measurement
        self._prev_output = output

        return np.array([output], dtype=np.float32)

    def set_gains(self, Kp: float, Ki: float, Kd: float) -> None:
        """Update PID gains.

        Args:
            Kp: Proportional gain.
            Ki: Integral gain.
            Kd: Derivative gain.
        """
        self.gains = PIDGains(Kp=Kp, Ki=Ki, Kd=Kd)

    def set_setpoint(self, setpoint: float) -> None:
        """Update the setpoint.

        Args:
            setpoint: New target value.
        """
        self.setpoint = setpoint

    @classmethod
    def ziegler_nichols(
        cls,
        Ku: float,
        Tu: float,
        controller_type: str = "PID",
        dt: float = 1.0,
        **kwargs,
    ) -> "PIDController":
        """Create PID controller using Ziegler-Nichols tuning.

        Args:
            Ku: Ultimate gain (gain at which system oscillates).
            Tu: Ultimate period (oscillation period at Ku).
            controller_type: "P", "PI", or "PID".
            dt: Sampling time.
            **kwargs: Additional arguments passed to constructor.

        Returns:
            Tuned PIDController instance.
        """
        if controller_type == "P":
            Kp, Ki, Kd = 0.5 * Ku, 0.0, 0.0
        elif controller_type == "PI":
            Kp = 0.45 * Ku
            Ki = 0.54 * Ku / Tu
            Kd = 0.0
        elif controller_type == "PID":
            Kp = 0.6 * Ku
            Ki = 1.2 * Ku / Tu
            Kd = 0.075 * Ku * Tu
        else:
            raise ValueError(f"Unknown controller type: {controller_type}")

        gains = PIDGains(Kp=Kp, Ki=Ki, Kd=Kd)
        return cls(gains=gains, dt=dt, **kwargs)

    @classmethod
    def cohen_coon(
        cls,
        K: float,
        tau: float,
        theta: float,
        controller_type: str = "PID",
        dt: float = 1.0,
        **kwargs,
    ) -> "PIDController":
        """Create PID controller using Cohen-Coon tuning.

        Based on first-order plus dead-time (FOPDT) model:
            G(s) = K * exp(-theta*s) / (tau*s + 1)

        Args:
            K: Process gain.
            tau: Time constant.
            theta: Dead time (delay).
            controller_type: "P", "PI", or "PID".
            dt: Sampling time.
            **kwargs: Additional arguments passed to constructor.

        Returns:
            Tuned PIDController instance.
        """
        r = theta / tau  # Ratio of dead time to time constant

        if controller_type == "P":
            Kp = (1 / K) * (tau / theta) * (1 + r / 3)
            Ki, Kd = 0.0, 0.0
        elif controller_type == "PI":
            Kp = (1 / K) * (tau / theta) * (0.9 + r / 12)
            Ti = theta * (30 + 3 * r) / (9 + 20 * r)
            Ki = Kp / Ti
            Kd = 0.0
        elif controller_type == "PID":
            Kp = (1 / K) * (tau / theta) * (4 / 3 + r / 4)
            Ti = theta * (32 + 6 * r) / (13 + 8 * r)
            Td = theta * 4 / (11 + 2 * r)
            Ki = Kp / Ti
            Kd = Kp * Td
        else:
            raise ValueError(f"Unknown controller type: {controller_type}")

        gains = PIDGains(Kp=Kp, Ki=Ki, Kd=Kd)
        return cls(gains=gains, dt=dt, **kwargs)

    def get_config(self) -> Dict[str, Any]:
        """Get controller configuration.

        Returns:
            Configuration dictionary.
        """
        config = super().get_config()
        config.update({
            "Kp": self.gains.Kp,
            "Ki": self.gains.Ki,
            "Kd": self.gains.Kd,
            "dt": self.dt,
            "anti_windup": self.anti_windup,
            "derivative_filter": self.derivative_filter,
            "setpoint_weight_b": self.setpoint_weight_b,
            "setpoint_weight_c": self.setpoint_weight_c,
        })
        return config

    def get_terms(self) -> Dict[str, float]:
        """Get the current P, I, D terms for debugging.

        Returns:
            Dictionary with 'P', 'I', 'D' terms.
        """
        return {
            "P": self.gains.Kp * (self.setpoint_weight_b * self.setpoint -
                                  (self._prev_measurement or 0)),
            "I": self._integral,
            "D": self.gains.Kd * self._prev_derivative,
        }
