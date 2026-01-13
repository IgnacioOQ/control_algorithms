"""
Base agent interface for the Ab Initio RL Simulation System.

This module defines the abstract interface that all agents must implement,
providing a unified API for training and inference across different algorithms.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import numpy as np


class BaseAgent(ABC):
    """Abstract base class for all RL agents.

    This interface supports both online (Bandit, MCTS) and batch (DQN, PPO)
    learning paradigms through a flexible store/update pattern.

    Attributes:
        name: Human-readable name of the agent.
        training: Whether the agent is in training mode.
    """

    def __init__(self, name: str = "BaseAgent"):
        """Initialize the agent.

        Args:
            name: Identifier for logging/debugging.
        """
        self.name = name
        self.training = True

    @abstractmethod
    def select_action(
        self, state: np.ndarray, explore: bool = True
    ) -> Union[int, np.ndarray]:
        """Select an action given the current state.

        Args:
            state: The current observation from the environment.
            explore: Whether to use exploration (e.g., epsilon-greedy, UCB).
                     Set to False for evaluation/inference.

        Returns:
            The selected action (int for discrete, array for continuous).
        """
        pass

    @abstractmethod
    def store(
        self,
        state: np.ndarray,
        action: Union[int, np.ndarray],
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition for learning.

        Args:
            state: The state before the action.
            action: The action taken.
            reward: The reward received.
            next_state: The state after the action.
            done: Whether the episode ended.
        """
        pass

    @abstractmethod
    def update(self) -> Dict[str, float]:
        """Perform a learning update.

        For Bandits: Update after each step.
        For DQN: Sample from replay buffer and do gradient descent.
        For PPO: Process trajectory buffer and do multiple epochs.
        For MCTS: This is a no-op (learning happens during search).

        Returns:
            Dictionary of training metrics (loss, etc.).
        """
        pass

    @abstractmethod
    def ready_to_train(self) -> bool:
        """Check if the agent has enough data to perform an update.

        For Bandits: Always True (update after each step).
        For DQN: True when replay buffer has enough samples.
        For PPO: True when trajectory buffer is full.
        For MCTS: Always False (no separate training phase).

        Returns:
            Whether update() should be called.
        """
        pass

    def train(self) -> None:
        """Set the agent to training mode."""
        self.training = True

    def eval(self) -> None:
        """Set the agent to evaluation mode (no exploration)."""
        self.training = False

    def save(self, path: str) -> None:
        """Save the agent's parameters to disk.

        Args:
            path: File path to save to.
        """
        raise NotImplementedError("save() not implemented for this agent.")

    def load(self, path: str) -> None:
        """Load the agent's parameters from disk.

        Args:
            path: File path to load from.
        """
        raise NotImplementedError("load() not implemented for this agent.")

    def get_config(self) -> Dict[str, Any]:
        """Get the agent's configuration as a dictionary.

        Returns:
            Configuration dictionary for reproducibility.
        """
        return {"name": self.name, "training": self.training}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
