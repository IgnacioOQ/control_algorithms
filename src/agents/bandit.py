"""
LinUCB Contextual Bandit Agent.

Implements the Disjoint LinUCB algorithm with per-arm ridge regression
and Sherman-Morrison O(d²) matrix inverse updates.

Suitable for:
- Smart Grid demand response (discrete action buckets)
- Server load balancing (routing decisions)
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np

from .base import BaseAgent
from ..utils.math_ops import sherman_morrison_update


class RandomAgent(BaseAgent):
    """Random action selection baseline agent.

    Selects actions uniformly at random, ignoring the state.
    Useful as a baseline for comparing against learned policies.

    Attributes:
        n_actions: Number of discrete actions available.
    """

    def __init__(self, n_actions: int, name: str = "Random"):
        """Initialize the Random agent.

        Args:
            n_actions: Number of discrete actions available.
            name: Agent identifier.
        """
        super().__init__(name=name)
        self.n_actions = n_actions

    def select_action(
        self, state: np.ndarray, explore: bool = True
    ) -> int:
        """Select a random action.

        Args:
            state: Current state (ignored).
            explore: Whether to explore (ignored - always random).

        Returns:
            Randomly selected action index.
        """
        return np.random.randint(0, self.n_actions)

    def store(
        self,
        state: np.ndarray,
        action: Union[int, np.ndarray],
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store transition (no-op for random agent)."""
        pass

    def update(self) -> Dict[str, float]:
        """Update the model (no-op for random agent)."""
        return {}

    def ready_to_train(self) -> bool:
        """Random agent never trains."""
        return False

    def get_config(self) -> Dict[str, Any]:
        """Get agent configuration."""
        return {
            **super().get_config(),
            "n_actions": self.n_actions,
        }


class LinUCBAgent(BaseAgent):
    """Disjoint LinUCB Contextual Bandit Agent.

    Maintains a separate ridge regression model for each arm (action).
    Uses Upper Confidence Bound (UCB) for exploration-exploitation balance.

    The expected reward for arm a given context x is modeled as:
        E[r | x, a] = x^T θ_a

    At each step, the agent:
    1. Computes UCB for each arm: p_a = θ̂_a^T x + α √(x^T A_a^{-1} x)
    2. Selects the arm with highest UCB
    3. Updates the model for the selected arm after receiving reward

    Attributes:
        n_arms: Number of discrete actions.
        context_dim: Dimension of context/state features.
        alpha: Exploration parameter (higher = more exploration).
        regularization: Ridge regression regularization (λ).
    """

    def __init__(
        self,
        n_arms: int,
        context_dim: int,
        alpha: float = 1.0,
        regularization: float = 1.0,
        name: str = "LinUCB",
    ):
        """Initialize the LinUCB agent.

        Args:
            n_arms: Number of discrete actions/arms.
            context_dim: Dimension of the context vector.
            alpha: UCB exploration parameter.
            regularization: Ridge regression regularization λ.
            name: Agent identifier.
        """
        super().__init__(name=name)
        
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.alpha = alpha
        self.regularization = regularization
        
        # Per-arm parameters
        # A_inv[a] = (A_a)^{-1} where A_a = X_a^T X_a + λI
        # b[a] = X_a^T r_a
        self.A_inv = np.array([
            np.eye(context_dim) / regularization for _ in range(n_arms)
        ])
        self.b = np.zeros((n_arms, context_dim))
        
        # Cached theta estimates
        self._theta = np.zeros((n_arms, context_dim))
        
        # Last action taken (for update)
        self._last_action: Optional[int] = None
        self._last_context: Optional[np.ndarray] = None
        
        # Statistics
        self.arm_counts = np.zeros(n_arms, dtype=np.int64)
        self.total_steps = 0

    def select_action(
        self, state: np.ndarray, explore: bool = True
    ) -> int:
        """Select an action using UCB criterion.

        Args:
            state: Context vector (1D array of shape (context_dim,)).
            explore: Whether to use UCB exploration. If False, uses greedy.

        Returns:
            Selected arm index.
        """
        x = np.asarray(state, dtype=np.float64).flatten()
        
        if len(x) != self.context_dim:
            raise ValueError(
                f"State dimension {len(x)} doesn't match context_dim {self.context_dim}"
            )
        
        # Compute UCB for each arm
        ucb_scores = np.zeros(self.n_arms)
        
        for a in range(self.n_arms):
            # θ̂_a = A_a^{-1} b_a
            theta_a = self.A_inv[a] @ self.b[a]
            self._theta[a] = theta_a
            
            # Mean estimate: θ̂_a^T x
            mean = np.dot(theta_a, x)
            
            if explore and self.training:
                # UCB bonus: α √(x^T A_a^{-1} x)
                x_A_inv_x = x @ self.A_inv[a] @ x
                ucb_bonus = self.alpha * np.sqrt(max(0.0, x_A_inv_x))
                ucb_scores[a] = mean + ucb_bonus
            else:
                ucb_scores[a] = mean
        
        # Select arm with highest UCB
        action = int(np.argmax(ucb_scores))
        
        # Store for update
        self._last_action = action
        self._last_context = x.copy()
        
        return action

    def store(
        self,
        state: np.ndarray,
        action: Union[int, np.ndarray],
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store transition (updates happen immediately in update()).

        For LinUCB, we store the context and reward for the selected arm
        to update the model in the update() call.
        """
        # Context and action already stored in select_action
        self._pending_reward = reward

    def update(self) -> Dict[str, float]:
        """Update the model for the last selected arm.

        Uses Sherman-Morrison formula for O(d²) inverse update.

        Returns:
            Training metrics (arm selected, reward received).
        """
        if self._last_action is None or self._last_context is None:
            return {}
        
        a = self._last_action
        x = self._last_context
        r = self._pending_reward
        
        # Update A_inv using Sherman-Morrison
        # A_new = A_old + x x^T
        # A_new^{-1} = A_old^{-1} - (A_old^{-1} x x^T A_old^{-1}) / (1 + x^T A_old^{-1} x)
        self.A_inv[a] = sherman_morrison_update(self.A_inv[a], x)
        
        # Update b: b_new = b_old + r * x
        self.b[a] += r * x
        
        # Update statistics
        self.arm_counts[a] += 1
        self.total_steps += 1
        
        # Clear pending
        self._last_action = None
        self._last_context = None
        
        return {
            "arm_selected": a,
            "reward": r,
            "arm_count": self.arm_counts[a],
        }

    def ready_to_train(self) -> bool:
        """LinUCB updates after every step."""
        return self._last_action is not None

    def get_arm_estimates(self, context: np.ndarray) -> np.ndarray:
        """Get estimated rewards for all arms given a context.

        Args:
            context: Context vector.

        Returns:
            Array of shape (n_arms,) with estimated rewards.
        """
        x = np.asarray(context, dtype=np.float64).flatten()
        estimates = np.zeros(self.n_arms)
        
        for a in range(self.n_arms):
            theta_a = self.A_inv[a] @ self.b[a]
            estimates[a] = np.dot(theta_a, x)
        
        return estimates

    def get_arm_uncertainties(self, context: np.ndarray) -> np.ndarray:
        """Get uncertainty estimates for all arms given a context.

        Args:
            context: Context vector.

        Returns:
            Array of shape (n_arms,) with uncertainty (√(x^T A^{-1} x)).
        """
        x = np.asarray(context, dtype=np.float64).flatten()
        uncertainties = np.zeros(self.n_arms)
        
        for a in range(self.n_arms):
            x_A_inv_x = x @ self.A_inv[a] @ x
            uncertainties[a] = np.sqrt(max(0.0, x_A_inv_x))
        
        return uncertainties

    def save(self, path: str) -> None:
        """Save agent parameters."""
        np.savez(
            path,
            A_inv=self.A_inv,
            b=self.b,
            arm_counts=self.arm_counts,
            total_steps=self.total_steps,
            alpha=self.alpha,
            regularization=self.regularization,
        )

    def load(self, path: str) -> None:
        """Load agent parameters."""
        data = np.load(path)
        self.A_inv = data["A_inv"]
        self.b = data["b"]
        self.arm_counts = data["arm_counts"]
        self.total_steps = int(data["total_steps"])
        self.alpha = float(data["alpha"])
        self.regularization = float(data["regularization"])

    def get_config(self) -> Dict[str, Any]:
        """Get agent configuration."""
        return {
            **super().get_config(),
            "n_arms": self.n_arms,
            "context_dim": self.context_dim,
            "alpha": self.alpha,
            "regularization": self.regularization,
        }

    def reset(self) -> None:
        """Reset the agent to initial state."""
        self.A_inv = np.array([
            np.eye(self.context_dim) / self.regularization 
            for _ in range(self.n_arms)
        ])
        self.b = np.zeros((self.n_arms, self.context_dim))
        self._theta = np.zeros((self.n_arms, self.context_dim))
        self.arm_counts = np.zeros(self.n_arms, dtype=np.int64)
        self.total_steps = 0
        self._last_action = None
        self._last_context = None
