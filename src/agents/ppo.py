"""
Proximal Policy Optimization (PPO) Agent.

Implements PPO with:
- Actor-Critic network with shared or separate backbones
- Gaussian policy for continuous actions (learnable log_std)
- Generalized Advantage Estimation (GAE)
- Clipped surrogate objective
- Value function loss and entropy bonus

Suitable for:
- Homeostasis / Bergman model (continuous insulin control)
- Any continuous control task
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Normal
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .base import BaseAgent


if HAS_TORCH:
    class ActorCritic(nn.Module):
        """Actor-Critic network for PPO.

        Separate networks for policy (actor) and value (critic).
        Actor outputs mean and log_std for Gaussian policy.
        """

        def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_dims: Tuple[int, ...] = (64, 64),
            log_std_init: float = 0.0,
        ):
            """Initialize Actor-Critic network.

            Args:
                state_dim: Dimension of state input.
                action_dim: Dimension of action output.
                hidden_dims: Hidden layer sizes for both networks.
                log_std_init: Initial value for log standard deviation.
            """
            super().__init__()

            # Actor network (policy)
            actor_layers = []
            prev_dim = state_dim
            for hidden_dim in hidden_dims:
                actor_layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.Tanh(),
                ])
                prev_dim = hidden_dim
            
            self.actor_backbone = nn.Sequential(*actor_layers)
            self.actor_mean = nn.Linear(prev_dim, action_dim)
            
            # Learnable log standard deviation
            self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)

            # Critic network (value function)
            critic_layers = []
            prev_dim = state_dim
            for hidden_dim in hidden_dims:
                critic_layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.Tanh(),
                ])
                prev_dim = hidden_dim
            critic_layers.append(nn.Linear(prev_dim, 1))
            
            self.critic = nn.Sequential(*critic_layers)

        def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Forward pass.

            Args:
                state: State tensor of shape (batch, state_dim).

            Returns:
                Tuple of (action_mean, action_log_std, value).
            """
            # Actor
            actor_hidden = self.actor_backbone(state)
            action_mean = self.actor_mean(actor_hidden)
            action_log_std = self.log_std.expand_as(action_mean)

            # Critic
            value = self.critic(state).squeeze(-1)

            return action_mean, action_log_std, value

        def get_action_and_value(
            self, state: torch.Tensor, action: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """Sample action and compute log probability and value.

            Args:
                state: State tensor.
                action: Optional action (for computing log prob of given action).

            Returns:
                Tuple of (action, log_prob, entropy, value).
            """
            action_mean, action_log_std, value = self.forward(state)
            action_std = action_log_std.exp()
            
            dist = Normal(action_mean, action_std)
            
            if action is None:
                action = dist.sample()
            
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

            return action, log_prob, entropy, value


class TrajectoryBuffer:
    """Trajectory buffer for on-policy learning.

    Stores complete trajectories and computes GAE advantages.
    Cleared after each training epoch.
    """

    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        """Initialize trajectory buffer.

        Args:
            capacity: Maximum number of steps to store.
            state_dim: Dimension of state.
            action_dim: Dimension of action.
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Storage
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        # Computed after trajectory collection
        self.advantages = np.zeros(capacity, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)
        
        self.ptr = 0
        self.size = 0

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ) -> None:
        """Add a transition to the buffer."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = float(done)
        
        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)

    def compute_gae(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """Compute Generalized Advantage Estimation (GAE).

        GAE formula:
            δ_t = r_t + γ V(s_{t+1}) - V(s_t)
            A_t = δ_t + (γλ) A_{t+1}

        Args:
            last_value: Value estimate for the state after the last transition.
            gamma: Discount factor.
            gae_lambda: GAE lambda parameter (bias-variance tradeoff).
        """
        last_gae = 0.0
        
        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]
            
            # TD error
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            
            # GAE
            self.advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        
        # Returns = advantages + values
        self.returns[:self.ptr] = self.advantages[:self.ptr] + self.values[:self.ptr]

    def get_batches(
        self, batch_size: int, normalize_advantages: bool = True
    ) -> List[Tuple[np.ndarray, ...]]:
        """Get random mini-batches for training.

        Args:
            batch_size: Size of each mini-batch.
            normalize_advantages: Whether to normalize advantages to zero mean, unit std.

        Returns:
            List of (states, actions, old_log_probs, advantages, returns) tuples.
        """
        indices = np.random.permutation(self.ptr)
        
        # Normalize advantages
        if normalize_advantages:
            adv_mean = self.advantages[:self.ptr].mean()
            adv_std = self.advantages[:self.ptr].std() + 1e-8
            advantages = (self.advantages[:self.ptr] - adv_mean) / adv_std
        else:
            advantages = self.advantages[:self.ptr]
        
        batches = []
        for start in range(0, self.ptr, batch_size):
            end = min(start + batch_size, self.ptr)
            batch_indices = indices[start:end]
            
            batches.append((
                self.states[batch_indices],
                self.actions[batch_indices],
                self.log_probs[batch_indices],
                advantages[batch_indices],
                self.returns[batch_indices],
            ))
        
        return batches

    def clear(self) -> None:
        """Clear the buffer."""
        self.ptr = 0
        self.size = 0


class PPOAgent(BaseAgent):
    """Proximal Policy Optimization Agent for continuous control.

    Uses clipped surrogate objective for stable policy updates.
    Implements GAE for variance reduction in advantage estimates.

    Attributes:
        state_dim: Dimension of state space.
        action_dim: Dimension of action space.
        gamma: Discount factor.
        gae_lambda: GAE lambda parameter.
        clip_epsilon: PPO clipping parameter.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (64, 64),
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        lr: float = 3e-4,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
        buffer_size: int = 2048,
        action_low: float = -1.0,
        action_high: float = 1.0,
        device: Optional[str] = None,
        name: str = "PPO",
    ):
        """Initialize the PPO agent.

        Args:
            state_dim: Dimension of state.
            action_dim: Dimension of action.
            hidden_dims: Hidden layer sizes.
            gamma: Discount factor.
            gae_lambda: GAE lambda.
            clip_epsilon: PPO clipping epsilon.
            lr: Learning rate.
            value_coef: Value loss coefficient.
            entropy_coef: Entropy bonus coefficient.
            max_grad_norm: Maximum gradient norm for clipping.
            n_epochs: Number of epochs per update.
            batch_size: Mini-batch size.
            buffer_size: Trajectory buffer size.
            action_low: Lower bound for actions.
            action_high: Upper bound for actions.
            device: PyTorch device.
            name: Agent identifier.
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for PPO. Install with: pip install torch")
        
        super().__init__(name=name)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.action_low = action_low
        self.action_high = action_high
        
        # Device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        
        # Network
        self.ac_network = ActorCritic(state_dim, action_dim, hidden_dims).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.ac_network.parameters(), lr=lr)
        
        # Buffer
        self.buffer = TrajectoryBuffer(buffer_size, state_dim, action_dim)
        
        # Tracking
        self.step_count = 0
        self.update_count = 0
        
        # Last value for GAE computation
        self._last_value = 0.0

    def select_action(
        self, state: np.ndarray, explore: bool = True
    ) -> np.ndarray:
        """Select action from Gaussian policy.

        Args:
            state: Current state.
            explore: Whether to sample (True) or use mean (False).

        Returns:
            Action array.
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, _, value = self.ac_network.get_action_and_value(state_t)
        
        action = action.cpu().numpy().flatten()
        
        if not explore or not self.training:
            # Use mean action
            with torch.no_grad():
                mean, _, _ = self.ac_network(state_t)
            action = mean.cpu().numpy().flatten()
        
        # Clip to action bounds
        action = np.clip(action, self.action_low, self.action_high)
        
        # Store log_prob and value for buffer
        self._current_log_prob = log_prob.item()
        self._current_value = value.item()
        
        return action

    def store(
        self,
        state: np.ndarray,
        action: Union[int, np.ndarray],
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store transition in trajectory buffer."""
        action = np.asarray(action, dtype=np.float32).flatten()
        
        self.buffer.add(
            state=state,
            action=action,
            reward=reward,
            value=self._current_value,
            log_prob=self._current_log_prob,
            done=done,
        )
        
        self.step_count += 1
        
        # Compute last value for GAE if episode not done
        if not done:
            with torch.no_grad():
                next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                _, _, self._last_value = self.ac_network(next_state_t)
                self._last_value = self._last_value.item()
        else:
            self._last_value = 0.0

    def update(self) -> Dict[str, float]:
        """Perform PPO update on collected trajectory.

        Returns:
            Training metrics.
        """
        if not self.ready_to_train():
            return {}
        
        # Compute GAE
        self.buffer.compute_gae(self._last_value, self.gamma, self.gae_lambda)
        
        # Training metrics
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0
        
        # Multiple epochs over the data
        for _ in range(self.n_epochs):
            batches = self.buffer.get_batches(self.batch_size)
            
            for batch in batches:
                states, actions, old_log_probs, advantages, returns = batch
                
                # Convert to tensors
                states = torch.FloatTensor(states).to(self.device)
                actions = torch.FloatTensor(actions).to(self.device)
                old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
                advantages = torch.FloatTensor(advantages).to(self.device)
                returns = torch.FloatTensor(returns).to(self.device)
                
                # Get current policy outputs
                _, new_log_probs, entropy, values = self.ac_network.get_action_and_value(
                    states, actions
                )
                
                # Policy loss (clipped surrogate objective)
                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = 0.5 * ((values - returns) ** 2).mean()
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.ac_network.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                
                # Accumulate metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1
        
        # Clear buffer
        self.buffer.clear()
        self.update_count += 1
        
        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
            "n_epochs": self.n_epochs,
        }

    def ready_to_train(self) -> bool:
        """Check if buffer is full."""
        return self.buffer.ptr >= self.buffer_size

    def save(self, path: str) -> None:
        """Save agent state."""
        torch.save({
            "ac_network": self.ac_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step_count": self.step_count,
            "update_count": self.update_count,
        }, path)

    def load(self, path: str) -> None:
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.ac_network.load_state_dict(checkpoint["ac_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.step_count = checkpoint["step_count"]
        self.update_count = checkpoint["update_count"]

    def get_config(self) -> Dict[str, Any]:
        """Get agent configuration."""
        return {
            **super().get_config(),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_epsilon": self.clip_epsilon,
            "n_epochs": self.n_epochs,
            "buffer_size": self.buffer_size,
        }
