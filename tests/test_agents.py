"""
Unit tests for agents.

Tests:
- Agent interface compliance
- Action selection
- Buffer/storage operations
- Update mechanics
"""

import numpy as np
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.agents import LinUCBAgent

# DQN and PPO require PyTorch
try:
    import torch
    from src.agents import DQNAgent, PPOAgent, ReplayBuffer, TrajectoryBuffer
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestLinUCBAgent:
    """Tests for LinUCB contextual bandit agent."""

    @pytest.fixture
    def agent(self):
        """Create a test agent."""
        return LinUCBAgent(
            n_arms=4,
            context_dim=5,
            alpha=1.0,
            regularization=1.0,
        )

    def test_select_action_returns_valid_arm(self, agent):
        """Test that select_action returns a valid arm."""
        state = np.random.randn(5)
        action = agent.select_action(state)
        
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < 4

    def test_store_and_update(self, agent):
        """Test store and update cycle."""
        state = np.random.randn(5)
        action = agent.select_action(state)
        
        agent.store(state, action, reward=1.0, next_state=state, done=False)
        
        assert agent.ready_to_train()
        
        metrics = agent.update()
        assert "reward" in metrics
        assert metrics["reward"] == 1.0

    def test_ucb_exploration(self, agent):
        """Test that UCB explores unvisited arms."""
        np.random.seed(42)
        state = np.random.randn(5)
        
        # Select many actions
        arms_selected = set()
        for _ in range(100):
            action = agent.select_action(state, explore=True)
            arms_selected.add(action)
            agent.store(state, action, np.random.randn(), state, False)
            agent.update()
        
        # Should have explored all arms
        assert len(arms_selected) == 4

    def test_arm_estimates(self, agent):
        """Test get_arm_estimates method."""
        state = np.random.randn(5)
        estimates = agent.get_arm_estimates(state)
        
        assert estimates.shape == (4,)

    def test_reset(self, agent):
        """Test agent reset."""
        state = np.random.randn(5)
        action = agent.select_action(state)
        agent.store(state, action, 1.0, state, False)
        agent.update()
        
        assert agent.total_steps == 1
        
        agent.reset()
        
        assert agent.total_steps == 0
        assert np.all(agent.arm_counts == 0)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestReplayBuffer:
    """Tests for DQN replay buffer."""

    def test_add_and_sample(self):
        """Test adding and sampling from buffer."""
        buffer = ReplayBuffer(capacity=100, state_dim=4)
        
        # Add some transitions
        for i in range(50):
            buffer.add(
                state=np.random.randn(4),
                action=i % 3,
                reward=float(i),
                next_state=np.random.randn(4),
                done=False,
            )
        
        assert len(buffer) == 50
        
        # Sample
        states, actions, rewards, next_states, dones = buffer.sample(10)
        
        assert states.shape == (10, 4)
        assert actions.shape == (10,)
        assert rewards.shape == (10,)

    def test_circular_overwrite(self):
        """Test circular buffer overwrites old data."""
        buffer = ReplayBuffer(capacity=10, state_dim=2)
        
        # Add more than capacity
        for i in range(20):
            buffer.add(
                state=np.array([float(i), float(i)]),
                action=0,
                reward=float(i),
                next_state=np.zeros(2),
                done=False,
            )
        
        # Should still only have capacity items
        assert len(buffer) == 10
        
        # Should have most recent items (indices 10-19)
        assert buffer.rewards.min() >= 10


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestDQNAgent:
    """Tests for DQN agent."""

    @pytest.fixture
    def agent(self):
        """Create a test agent."""
        return DQNAgent(
            state_dim=4,
            action_dim=2,
            hidden_dims=(32, 32),
            buffer_size=1000,
            batch_size=32,
            min_buffer_size=100,
        )

    def test_select_action_returns_valid(self, agent):
        """Test action selection."""
        state = np.random.randn(4)
        action = agent.select_action(state)
        
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < 2

    def test_not_ready_initially(self, agent):
        """Test agent isn't ready to train initially."""
        assert not agent.ready_to_train()

    def test_ready_after_buffer_filled(self, agent):
        """Test agent is ready after buffer has enough samples."""
        for i in range(100):
            state = np.random.randn(4)
            action = agent.select_action(state)
            agent.store(state, action, 1.0, np.random.randn(4), False)
        
        assert agent.ready_to_train()

    def test_update_returns_metrics(self, agent):
        """Test that update returns training metrics."""
        # Fill buffer
        for i in range(100):
            agent.store(
                np.random.randn(4),
                np.random.randint(2),
                1.0,
                np.random.randn(4),
                False,
            )
        
        metrics = agent.update()
        
        assert "loss" in metrics
        assert "epsilon" in metrics


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestTrajectoryBuffer:
    """Tests for PPO trajectory buffer."""

    def test_add_and_get_batches(self):
        """Test adding and getting batches."""
        buffer = TrajectoryBuffer(capacity=100, state_dim=3, action_dim=1)
        
        for i in range(50):
            buffer.add(
                state=np.random.randn(3),
                action=np.array([0.5]),
                reward=1.0,
                value=0.9,
                log_prob=-0.5,
                done=False,
            )
        
        assert buffer.ptr == 50
        
        # Compute GAE
        buffer.compute_gae(last_value=0.0, gamma=0.99, gae_lambda=0.95)
        
        # Get batches
        batches = buffer.get_batches(batch_size=10)
        
        assert len(batches) == 5  # 50 / 10

    def test_gae_computation(self):
        """Test GAE computation."""
        buffer = TrajectoryBuffer(capacity=10, state_dim=2, action_dim=1)
        
        # Add simple trajectory
        for i in range(5):
            buffer.add(
                state=np.zeros(2),
                action=np.zeros(1),
                reward=1.0,
                value=0.5,
                log_prob=0.0,
                done=False,
            )
        
        buffer.compute_gae(last_value=0.5, gamma=0.99, gae_lambda=0.95)
        
        # Advantages should be computed
        assert not np.all(buffer.advantages[:5] == 0)

    def test_clear(self):
        """Test buffer clearing."""
        buffer = TrajectoryBuffer(capacity=100, state_dim=2, action_dim=1)
        
        for i in range(50):
            buffer.add(np.zeros(2), np.zeros(1), 0.0, 0.0, 0.0, False)
        
        assert buffer.ptr == 50
        
        buffer.clear()
        
        assert buffer.ptr == 0


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestPPOAgent:
    """Tests for PPO agent."""

    @pytest.fixture
    def agent(self):
        """Create a test agent."""
        return PPOAgent(
            state_dim=3,
            action_dim=1,
            hidden_dims=(32, 32),
            buffer_size=100,
            batch_size=20,
            n_epochs=2,
        )

    def test_select_action_returns_valid(self, agent):
        """Test action selection returns array."""
        state = np.random.randn(3)
        action = agent.select_action(state)
        
        assert isinstance(action, np.ndarray)
        assert action.shape == (1,)

    def test_action_clipping(self, agent):
        """Test that actions are clipped to bounds."""
        for _ in range(100):
            state = np.random.randn(3) * 10  # Large state to test extremes
            action = agent.select_action(state)
            
            assert np.all(action >= agent.action_low)
            assert np.all(action <= agent.action_high)


class TestAgentInterface:
    """Tests for common agent interface compliance."""

    @pytest.fixture(params=["linucb"])  # DQN/PPO tested separately if torch available
    def agent(self, request):
        """Parametrized fixture for agents."""
        return LinUCBAgent(n_arms=3, context_dim=4)

    def test_has_select_action(self, agent):
        """Test that agent has select_action method."""
        assert hasattr(agent, "select_action")

    def test_has_store(self, agent):
        """Test that agent has store method."""
        assert hasattr(agent, "store")

    def test_has_update(self, agent):
        """Test that agent has update method."""
        assert hasattr(agent, "update")

    def test_has_ready_to_train(self, agent):
        """Test that agent has ready_to_train method."""
        assert hasattr(agent, "ready_to_train")

    def test_train_eval_modes(self, agent):
        """Test train/eval mode switching."""
        agent.train()
        assert agent.training
        
        agent.eval()
        assert not agent.training

    def test_get_config(self, agent):
        """Test get_config returns dict."""
        config = agent.get_config()
        assert isinstance(config, dict)
        assert "name" in config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
