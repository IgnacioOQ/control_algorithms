"""
Unit tests for simulation environments.

Tests:
- Environment reset/step contracts
- State shape and bounds
- Reward ranges
- Seed reproducibility
"""

import numpy as np
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.envs import (
    ServerLoadEnv,
    ServerLoadConfig,
    SmartGridEnv,
    SmartGridConfig,
    HomeostasisEnv,
    HomeostasisConfig,
)


class TestServerLoadEnv:
    """Tests for Server Load Balancing environment."""

    @pytest.fixture
    def env(self):
        """Create a test environment."""
        config = ServerLoadConfig(
            num_servers=3,
            arrival_rate=5.0,
            service_rate=2.0,
            max_queue_size=20,
        )
        return ServerLoadEnv(config=config, seed=42)

    def test_reset_returns_valid_state(self, env):
        """Test that reset returns a valid observation."""
        state = env.reset()
        assert state is not None
        assert isinstance(state, np.ndarray)
        assert state.shape == (8,)  # 2*3 + 2 = 8

    def test_step_returns_valid_tuple(self, env):
        """Test that step returns (state, reward, done, info)."""
        env.reset()
        action = 0
        result = env.step(action)
        
        assert isinstance(result, tuple)
        assert len(result) == 4
        
        state, reward, done, info = result
        assert isinstance(state, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_action_space_bounds(self, env):
        """Test that action space is correctly defined."""
        env.reset()
        
        # Legal actions should be 0, 1, 2 (3 servers)
        legal = env.get_legal_actions()
        assert len(legal) == 3
        assert 0 in legal
        assert 2 in legal

    def test_invalid_action_raises(self, env):
        """Test that invalid actions raise an error."""
        env.reset()
        
        with pytest.raises(ValueError):
            env.step(5)  # Invalid server index

    def test_seed_reproducibility(self):
        """Test that same seed produces same trajectory."""
        config = ServerLoadConfig(num_servers=2)
        
        env1 = ServerLoadEnv(config=config, seed=123)
        env2 = ServerLoadEnv(config=config, seed=123)
        
        state1 = env1.reset()
        state2 = env2.reset()
        
        assert np.allclose(state1, state2)
        
        for _ in range(10):
            action = 0
            s1, r1, d1, _ = env1.step(action)
            s2, r2, d2, _ = env2.step(action)
            
            assert np.allclose(s1, s2)
            assert np.isclose(r1, r2)

    def test_copy_for_mcts(self, env):
        """Test that copy() creates independent copy."""
        env.reset()
        env.step(0)
        env.step(1)
        
        env_copy = env.copy()
        
        # Modify original
        env.step(0)
        
        # Copy should be unaffected
        assert env_copy.current_time != env.current_time


class TestSmartGridEnv:
    """Tests for Smart Grid environment."""

    @pytest.fixture
    def env(self):
        """Create a test environment."""
        config = SmartGridConfig(
            battery_capacity=50.0,
            max_power=10.0,
            episode_length=100,
        )
        return SmartGridEnv(config=config, seed=42)

    def test_reset_returns_valid_state(self, env):
        """Test reset returns valid observation."""
        state = env.reset()
        assert state is not None
        assert isinstance(state, np.ndarray)
        
        # Should be normalized to [0, 1]
        assert np.all(state >= 0)
        assert np.all(state <= 1)

    def test_soc_constraints(self, env):
        """Test that SoC stays within bounds."""
        env.reset()
        
        # Try to fully discharge
        for _ in range(50):
            env.step(np.array([-1.0]))  # Max discharge
        
        assert env.soc >= env.config.soc_min
        
        # Try to fully charge
        for _ in range(50):
            env.step(np.array([1.0]))  # Max charge
        
        assert env.soc <= env.config.soc_max

    def test_episode_terminates(self, env):
        """Test that episode terminates at correct length."""
        env.reset()
        done = False
        step_count = 0
        
        while not done:
            _, _, done, _ = env.step(np.array([0.0]))
            step_count += 1
        
        assert step_count == env.config.episode_length

    def test_action_clipping(self, env):
        """Test that actions are clipped to valid range."""
        env.reset()
        
        # Action outside bounds should be clipped
        state, reward, done, info = env.step(np.array([5.0]))  # > 1.0
        assert "power_applied" in info


class TestHomeostasisEnv:
    """Tests for Biological Homeostasis environment."""

    @pytest.fixture
    def env(self):
        """Create a test environment."""
        config = HomeostasisConfig(
            dt_control=5.0,
            episode_length=50,
            type1_diabetes=True,
        )
        return HomeostasisEnv(config=config, seed=42)

    def test_reset_returns_valid_state(self, env):
        """Test reset returns valid observation."""
        state = env.reset()
        assert state is not None
        assert isinstance(state, np.ndarray)
        assert len(state) == 3  # G, X, I

    def test_insulin_affects_glucose(self, env):
        """Test that insulin infusion lowers glucose."""
        # Reset with seed to ensure no random meal disturbance during test
        env.reset(seed=42)

        # Disable meals for this test to isolate insulin effect
        env.config.meal_probability = 0.0
        
        # Get initial glucose
        initial_G = env.state[0]
        
        # Apply max insulin for several steps
        for _ in range(10):
            env.step(np.array([1.0]))  # Max insulin
        
        final_G = env.state[0]
        
        # Glucose should decrease with max insulin (since we disabled meals)
        # Even with basal production, max insulin should drive it down or keep it stable
        assert final_G <= initial_G, f"Glucose rose from {initial_G} to {final_G} despite max insulin"

    def test_hypoglycemia_penalty(self, env):
        """Test that hypoglycemia incurs large penalty."""
        env.reset()
        
        # Force glucose very low (this shouldn't happen normally)
        env.state[0] = 60.0  # Below 70 threshold
        
        _, reward, _, info = env.step(np.array([0.0]))
        
        # Should have hypo event and penalty
        assert info["glucose"] < 70
        assert reward < -50  # Large penalty

    def test_glucose_stays_positive(self, env):
        """Test that glucose never goes negative."""
        env.reset()
        
        # Apply extreme insulin
        for _ in range(20):
            _, _, done, _ = env.step(np.array([1.0]))
            assert env.state[0] > 0, "Glucose went non-positive"
            if done:
                break


class TestEnvironmentInterface:
    """Tests for common environment interface compliance."""

    @pytest.fixture(params=["server_load", "smart_grid", "homeostasis"])
    def env(self, request):
        """Parametrized fixture for all environments."""
        if request.param == "server_load":
            return ServerLoadEnv(seed=42)
        elif request.param == "smart_grid":
            config = SmartGridConfig(episode_length=50)
            return SmartGridEnv(config=config, seed=42)
        else:
            config = HomeostasisConfig(episode_length=50)
            return HomeostasisEnv(config=config, seed=42)

    def test_has_observation_space(self, env):
        """Test that environment has observation_space."""
        assert hasattr(env, "observation_space")

    def test_has_action_space(self, env):
        """Test that environment has action_space."""
        assert hasattr(env, "action_space")

    def test_reset_seed(self, env):
        """Test that reset accepts seed."""
        state1 = env.reset(seed=100)
        state2 = env.reset(seed=100)
        assert np.allclose(state1, state2)

    def test_render_does_not_crash(self, env):
        """Test that render() doesn't crash."""
        env.reset()
        env.render()  # Should print without error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
