"""
Unit tests for math operations module.

Tests:
- RK4 integration accuracy
- Sherman-Morrison matrix inverse update
- OnlineNormalizer convergence
"""

import numpy as np
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.math_ops import (
    rk4_step,
    rk4_step_with_control,
    sherman_morrison_update,
    OnlineNormalizer,
)


class TestRK4Integration:
    """Tests for RK4 numerical integration."""

    def test_exponential_decay(self):
        """Test RK4 on simple exponential decay: dy/dt = -y."""
        decay_rate = 0.5
        
        def derivatives(y):
            return -decay_rate * y
        
        y0 = np.array([1.0])
        dt = 0.01
        
        # Integrate for 1 second
        y = y0.copy()
        for _ in range(100):
            y = rk4_step(y, dt, derivatives)
        
        # Analytical solution: y(1) = e^(-0.5)
        expected = np.exp(-decay_rate * 1.0)
        
        assert np.abs(y[0] - expected) < 1e-6, f"Expected {expected}, got {y[0]}"

    def test_harmonic_oscillator(self):
        """Test RK4 on harmonic oscillator: d²x/dt² = -ω²x."""
        omega = 2 * np.pi  # 1 Hz
        
        def derivatives(state):
            x, v = state
            return np.array([v, -omega**2 * x])
        
        # Initial conditions: x=1, v=0
        state = np.array([1.0, 0.0])
        dt = 0.001
        
        # Integrate for 1 period
        for _ in range(1000):
            state = rk4_step(state, dt, derivatives)
        
        # After 1 period, should return to initial position
        assert np.abs(state[0] - 1.0) < 0.01, f"Expected x≈1.0, got {state[0]}"
        assert np.abs(state[1]) < 0.1, f"Expected v≈0, got {state[1]}"

    def test_with_control_input(self):
        """Test RK4 with control input."""
        def derivatives(state, control, disturbance):
            return np.array([-0.1 * state[0] + control + disturbance])
        
        state = np.array([0.0])
        
        # Apply constant control
        # Need enough steps to converge (time constant = 10s, running for 100s = 1000 steps * 0.1)
        for _ in range(1000):
            state = rk4_step_with_control(state, 0.1, derivatives, control=1.0, disturbance=0.0)
        
        # Should converge to steady state: 0 = -0.1*x + 1 => x = 10
        assert np.abs(state[0] - 10.0) < 0.5


class TestShermanMorrison:
    """Tests for Sherman-Morrison matrix inverse update."""

    def test_correctness_against_direct_inverse(self):
        """Verify Sherman-Morrison gives same result as direct inversion."""
        d = 5
        np.random.seed(42)
        
        # Create random SPD matrix A
        X = np.random.randn(d, d)
        A = X @ X.T + np.eye(d)
        A_inv = np.linalg.inv(A)
        
        # Update vector
        x = np.random.randn(d)
        
        # Direct method: (A + x*x')^{-1}
        A_new = A + np.outer(x, x)
        A_new_inv_direct = np.linalg.inv(A_new)
        
        # Sherman-Morrison
        A_new_inv_sm = sherman_morrison_update(A_inv, x)
        
        # Compare
        error = np.linalg.norm(A_new_inv_direct - A_new_inv_sm)
        assert error < 1e-10, f"Sherman-Morrison error: {error}"

    def test_multiple_updates(self):
        """Test multiple sequential updates."""
        d = 4
        np.random.seed(123)
        
        # Start with identity
        A_inv = np.eye(d)
        A = np.eye(d)
        
        # Multiple updates
        for _ in range(10):
            x = np.random.randn(d) * 0.5
            A_inv = sherman_morrison_update(A_inv, x)
            A = A + np.outer(x, x)
        
        # Verify A_inv is still inverse of A
        product = A_inv @ A
        error = np.linalg.norm(product - np.eye(d))
        assert error < 1e-8, f"Accumulated error: {error}"


class TestOnlineNormalizer:
    """Tests for online mean/variance estimation."""

    def test_mean_convergence(self):
        """Test that running mean converges to true mean."""
        np.random.seed(42)
        dim = 3
        true_mean = np.array([1.0, 2.0, 3.0])
        
        normalizer = OnlineNormalizer(dim)
        
        # Add many samples
        for _ in range(10000):
            sample = true_mean + np.random.randn(dim)
            normalizer.update(sample)
        
        # Mean should be close to true mean
        error = np.linalg.norm(normalizer.mean - true_mean)
        assert error < 0.1, f"Mean error: {error}"

    def test_variance_convergence(self):
        """Test that running variance converges to true variance."""
        np.random.seed(42)
        dim = 3
        true_var = np.array([1.0, 4.0, 9.0])
        
        normalizer = OnlineNormalizer(dim)
        
        # Add many samples
        for _ in range(10000):
            sample = np.sqrt(true_var) * np.random.randn(dim)
            normalizer.update(sample)
        
        # Variance should be close to true variance
        error = np.linalg.norm(normalizer.var - true_var)
        assert error < 0.5, f"Variance error: {error}"

    def test_normalization(self):
        """Test that normalization produces zero mean, unit variance."""
        np.random.seed(42)
        dim = 2
        
        normalizer = OnlineNormalizer(dim)
        
        # Add samples
        samples = np.random.randn(1000, dim) * 2 + 5
        for sample in samples:
            normalizer.update(sample)
        
        # Normalize
        normalized = normalizer.normalize(samples)
        
        assert np.abs(normalized.mean()) < 0.1, "Mean not near zero"
        assert np.abs(normalized.std() - 1.0) < 0.2, "Std not near 1"

    def test_batch_update(self):
        """Test batch update produces same result as sequential."""
        np.random.seed(42)
        dim = 3
        
        normalizer1 = OnlineNormalizer(dim)
        normalizer2 = OnlineNormalizer(dim)
        
        samples = np.random.randn(100, dim)
        
        # Sequential update
        for sample in samples:
            normalizer1.update(sample)
        
        # Batch update
        normalizer2.update(samples)
        
        # Should be identical
        assert np.allclose(normalizer1.mean, normalizer2.mean)
        assert np.allclose(normalizer1.var, normalizer2.var)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
