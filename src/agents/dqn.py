"""
Deep Q-Network (DQN) Agent.

Implements DQN with:
- Pre-allocated NumPy replay buffer with O(1) insert/sample
- Target network with periodic updates
- Huber loss for robustness
- Double DQN variant for reduced overestimation bias

Suitable for:
- Server Load Balancing (discrete routing)
- Smart Grid (discretized actions)
"""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .base import BaseAgent


class ReplayBuffer:
    """Pre-allocated circular replay buffer for experience replay.

    Uses NumPy arrays for efficient memory allocation and sampling.
    Achieves O(1) insertion and O(batch_size) sampling.

    Attributes:
        capacity: Maximum buffer size.
        state_dim: Dimension of state vectors.
    """

    def __init__(self, capacity: int, state_dim: int):
        """Initialize the replay buffer.

        Args:
            capacity: Maximum number of transitions to store.
            state_dim: Dimension of state observations.
        """
        self.capacity = capacity
        self.state_dim = state_dim
        
        # Pre-allocate arrays
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        # Pointer and size tracking
        self.ptr = 0
        self.size = 0

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a transition to the buffer.

        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
            done: Whether episode ended.
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)
        
        # Advance pointer circularly
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones).
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def __len__(self) -> int:
        return self.size


if HAS_TORCH:
    class QNetwork(nn.Module):
        """Neural network for Q-value estimation.

        Architecture: FC -> ReLU -> FC -> ReLU -> FC
        """

        def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_dims: Tuple[int, ...] = (128, 128),
        ):
            """Initialize the Q-network.

            Args:
                state_dim: Dimension of state input.
                action_dim: Number of discrete actions.
                hidden_dims: Sizes of hidden layers.
            """
            super().__init__()
            
            layers = []
            prev_dim = state_dim
            
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                ])
                prev_dim = hidden_dim
            
            layers.append(nn.Linear(prev_dim, action_dim))
            
            self.network = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass.

            Args:
                x: State tensor of shape (batch, state_dim).

            Returns:
                Q-values of shape (batch, action_dim).
            """
            return self.network(x)


class DQNAgent(BaseAgent):
    """Deep Q-Network Agent.

    Uses experience replay and target networks for stable learning.
    Supports Double DQN for reduced overestimation bias.

    Attributes:
        state_dim: Dimension of state space.
        action_dim: Number of discrete actions.
        gamma: Discount factor.
        epsilon: Exploration rate.
        epsilon_decay: Rate of epsilon decay.
        epsilon_min: Minimum epsilon value.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (128, 128),
        gamma: float = 0.99,
        lr: float = 1e-3,
        buffer_size: int = 100000,
        batch_size: int = 64,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        target_update_freq: int = 100,
        min_buffer_size: int = 1000,
        double_dqn: bool = True,
        device: Optional[str] = None,
        name: str = "DQN",
    ):
        """Initialize the DQN agent.

        Args:
            state_dim: Dimension of state.
            action_dim: Number of actions.
            hidden_dims: Hidden layer sizes.
            gamma: Discount factor.
            lr: Learning rate.
            buffer_size: Replay buffer capacity.
            batch_size: Training batch size.
            epsilon_start: Initial exploration rate.
            epsilon_end: Final exploration rate.
            epsilon_decay: Epsilon decay multiplier per step.
            target_update_freq: Steps between target network updates.
            min_buffer_size: Minimum buffer size before training.
            double_dqn: Whether to use Double DQN.
            device: PyTorch device ('cuda' or 'cpu').
            name: Agent identifier.
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for DQN. Install with: pip install torch")
        
        super().__init__(name=name)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.min_buffer_size = min_buffer_size
        self.double_dqn = double_dqn
        
        # Device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        
        # Networks
        self.q_network = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size, state_dim)
        
        # Step counter
        self.step_count = 0
        self.update_count = 0

    def select_action(
        self, state: np.ndarray, explore: bool = True
    ) -> int:
        """Select action using epsilon-greedy policy.

        Args:
            state: Current state.
            explore: Whether to use epsilon-greedy exploration.

        Returns:
            Selected action index.
        """
        if explore and self.training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_t)
            return int(q_values.argmax(dim=1).item())

    def store(
        self,
        state: np.ndarray,
        action: Union[int, np.ndarray],
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store transition in replay buffer."""
        action = int(action) if isinstance(action, np.ndarray) else action
        self.buffer.add(state, action, reward, next_state, done)
        self.step_count += 1

    def update(self) -> Dict[str, float]:
        """Perform one gradient update on the Q-network.

        Uses the Bellman equation with target network:
            Y = r + γ max_a' Q_target(s', a') * (1 - done)

        For Double DQN:
            Y = r + γ Q_target(s', argmax_a Q(s', a)) * (1 - done)

        Returns:
            Training metrics.
        """
        if not self.ready_to_train():
            return {}
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: use online network to select action, target to evaluate
                next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
                next_q = self.target_network(next_states).gather(1, next_actions).squeeze(1)
            else:
                # Standard DQN
                next_q = self.target_network(next_states).max(dim=1)[0]
            
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Huber loss (Smooth L1)
        loss = F.smooth_l1_loss(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()
        
        # Update target network periodically
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return {
            "loss": loss.item(),
            "epsilon": self.epsilon,
            "buffer_size": len(self.buffer),
            "mean_q": current_q.mean().item(),
        }

    def ready_to_train(self) -> bool:
        """Check if buffer has enough samples."""
        return len(self.buffer) >= self.min_buffer_size

    def save(self, path: str) -> None:
        """Save agent state."""
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "step_count": self.step_count,
            "update_count": self.update_count,
        }, path)

    def load(self, path: str) -> None:
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.step_count = checkpoint["step_count"]
        self.update_count = checkpoint["update_count"]

    def get_config(self) -> Dict[str, Any]:
        """Get agent configuration."""
        return {
            **super().get_config(),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_end": self.epsilon_end,
            "double_dqn": self.double_dqn,
            "buffer_size": self.buffer.capacity,
            "batch_size": self.batch_size,
        }
