"""
Base environment protocol for the Ab Initio RL Simulation System.

This module defines the abstract interface that all simulation environments must
implement, along with data structures for state and action space definitions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class DiscreteSpace:
    """Discrete action/state space definition.

    Attributes:
        n: Number of discrete actions/states (0 to n-1).
    """

    n: int

    def sample(self, rng: Optional[np.random.RandomState] = None) -> int:
        """Sample a random action from the space."""
        if rng is None:
            return np.random.randint(0, self.n)
        return rng.randint(0, self.n)

    def contains(self, x: int) -> bool:
        """Check if x is a valid action in this space."""
        return isinstance(x, (int, np.integer)) and 0 <= x < self.n


@dataclass
class ContinuousSpace:
    """Continuous action/state space definition.

    Attributes:
        low: Lower bounds for each dimension.
        high: Upper bounds for each dimension.
        shape: Shape of the space (inferred from low/high).
    """

    low: np.ndarray
    high: np.ndarray

    def __post_init__(self):
        self.low = np.asarray(self.low, dtype=np.float32)
        self.high = np.asarray(self.high, dtype=np.float32)
        if self.low.shape != self.high.shape:
            raise ValueError("low and high must have the same shape")
        self.shape = self.low.shape

    def sample(self, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        """Sample a random action from the space."""
        if rng is None:
            return np.random.uniform(self.low, self.high).astype(np.float32)
        return rng.uniform(self.low, self.high).astype(np.float32)

    def contains(self, x: np.ndarray) -> bool:
        """Check if x is within bounds."""
        x = np.asarray(x)
        return x.shape == self.shape and np.all(x >= self.low) and np.all(x <= self.high)

    def clip(self, x: np.ndarray) -> np.ndarray:
        """Clip x to be within bounds."""
        return np.clip(x, self.low, self.high).astype(np.float32)


# Type alias for spaces
Space = Union[DiscreteSpace, ContinuousSpace]

# Type alias for state (can be array or dict)
State = Union[np.ndarray, Dict[str, np.ndarray]]

# Type alias for step return
StepResult = Tuple[State, float, bool, Dict[str, Any]]


class SimulationEnvironment(ABC):
    """Abstract base class for all simulation environments.

    This protocol enforces a rigid contract for environment implementations,
    handling both Discrete Time-Stepping (DT) and Discrete Event Simulation (DES).

    Attributes:
        observation_space: The observation/state space specification.
        action_space: The action space specification.
        rng: Per-environment random state for reproducibility.
    """

    observation_space: Space
    action_space: Space
    rng: np.random.RandomState

    def __init__(self, seed: Optional[int] = None):
        """Initialize the environment.

        Args:
            seed: Random seed for reproducibility.
        """
        self.rng = np.random.RandomState(seed)
        self._setup()

    @abstractmethod
    def _setup(self) -> None:
        """Internal setup called after RNG initialization.

        Subclasses should define observation_space and action_space here.
        """
        pass

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> State:
        """Reset the environment to an initial state.

        Args:
            seed: Optional seed to reset the RNG.

        Returns:
            The initial observation/state.
        """
        pass

    @abstractmethod
    def step(self, action: Union[int, np.ndarray]) -> StepResult:
        """Execute one timestep of the environment dynamics.

        Args:
            action: The action to take.

        Returns:
            Tuple of (next_state, reward, done, info).
            - next_state: The observation after action.
            - reward: The scalar reward signal.
            - done: Whether the episode has ended.
            - info: Additional diagnostic information.
        """
        pass

    @abstractmethod
    def get_legal_actions(self) -> List[int]:
        """Get list of legal actions in current state.

        Returns:
            List of valid action indices. For continuous spaces,
            this may return an empty list (all actions within bounds are legal).
        """
        pass

    def render(self) -> None:
        """Render the current state (text-based by default)."""
        print(f"State: {self._get_state_repr()}")

    def _get_state_repr(self) -> str:
        """Get a string representation of the current state."""
        return "Not implemented"

    def copy(self) -> "SimulationEnvironment":
        """Create a deep copy of the environment (for MCTS planning).

        Returns:
            A copy of this environment with identical state.
        """
        raise NotImplementedError(
            "copy() not implemented for this environment. "
            "Required for MCTS planning."
        )

    def seed(self, seed: int) -> None:
        """Reset the environment's random state.

        Args:
            seed: The seed value.
        """
        self.rng = np.random.RandomState(seed)

    @property
    def unwrapped(self) -> "SimulationEnvironment":
        """Return the base environment (for compatibility)."""
        return self
