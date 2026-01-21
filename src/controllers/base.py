"""
Base controller interface for classical and optimal control methods.

This module provides a base class that adapts control-theoretic methods
to work within the RL agent interface, enabling fair comparisons between
learning-based and model-based approaches.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from ..agents.base import BaseAgent


class BaseController(BaseAgent):
    """Abstract base class for model-based controllers.

    Controllers differ from RL agents in that they:
    1. Do not learn from data (parameters are fixed or analytically computed)
    2. May require a model of the system dynamics
    3. Typically have no exploration/exploitation tradeoff

    This class implements the BaseAgent interface so controllers can be
    used interchangeably with RL agents in the training loop.

    Attributes:
        setpoint: Target value(s) for regulation (if applicable).
        output_limits: (min, max) bounds on control output.
    """

    def __init__(
        self,
        name: str = "BaseController",
        setpoint: Optional[Union[float, np.ndarray]] = None,
        output_limits: Optional[Tuple[float, float]] = None,
    ):
        """Initialize the controller.

        Args:
            name: Identifier for logging/debugging.
            setpoint: Target value for setpoint tracking controllers.
            output_limits: (min, max) bounds on control output.
        """
        super().__init__(name=name)
        self.setpoint = setpoint
        self.output_limits = output_limits
        self.training = False  # Controllers don't train

    @abstractmethod
    def compute_control(
        self, state: np.ndarray, setpoint: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute the control action given current state.

        Args:
            state: Current system state/observation.
            setpoint: Optional setpoint override.

        Returns:
            Control action (continuous).
        """
        pass

    def select_action(
        self, state: np.ndarray, explore: bool = True
    ) -> Union[int, np.ndarray]:
        """Select action using the control law.

        Args:
            state: Current observation from the environment.
            explore: Ignored for controllers (no exploration).

        Returns:
            The computed control action.
        """
        action = self.compute_control(state)
        if self.output_limits is not None:
            action = np.clip(action, self.output_limits[0], self.output_limits[1])
        return action

    def store(
        self,
        state: np.ndarray,
        action: Union[int, np.ndarray],
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """No-op for controllers (no learning from transitions)."""
        pass

    def update(self) -> Dict[str, float]:
        """No-op for controllers (no learning updates).

        Returns:
            Empty metrics dictionary.
        """
        return {}

    def ready_to_train(self) -> bool:
        """Controllers never need training updates.

        Returns:
            Always False.
        """
        return False

    def reset(self) -> None:
        """Reset controller internal state (e.g., integrator).

        Override in subclasses that maintain internal state.
        """
        pass

    def get_config(self) -> Dict[str, Any]:
        """Get controller configuration.

        Returns:
            Configuration dictionary.
        """
        config = super().get_config()
        config.update({
            "setpoint": self.setpoint,
            "output_limits": self.output_limits,
        })
        return config
