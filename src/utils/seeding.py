"""
Random number generation and seeding utilities.

Provides deterministic reproducibility across all stochastic components:
- Global seeding for torch, numpy, and random
- Per-environment RNG management to prevent crosstalk
"""

import random
from typing import Optional

import numpy as np

# Try to import torch for seeding; if not available, skip torch seeding
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def set_global_seeds(seed: int) -> None:
    """Set random seeds for all libraries to ensure reproducibility.

    This sets seeds for:
    - Python's random module
    - NumPy's random module
    - PyTorch (if available)

    Args:
        seed: The seed value to use.

    Example:
        >>> set_global_seeds(42)
        >>> np.random.random()  # Will be deterministic
    """
    random.seed(seed)
    np.random.seed(seed)

    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # Deterministic operations (may reduce performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_rng(seed: Optional[int] = None) -> np.random.RandomState:
    """Create an independent random state for per-environment use.

    Each environment should maintain its own RandomState to prevent
    crosstalk during vectorized/parallel training.

    Args:
        seed: Optional seed for the RNG. If None, uses a random seed.

    Returns:
        An independent numpy RandomState object.

    Example:
        >>> rng1 = create_rng(42)
        >>> rng2 = create_rng(42)
        >>> rng1.random() == rng2.random()  # True - deterministic
    """
    return np.random.RandomState(seed)


def spawn_rngs(seed: int, n: int) -> list:
    """Create multiple independent RNGs from a single seed.

    Useful for vectorized environments where each needs its own RNG.
    Uses seed offsets to ensure independence.

    Args:
        seed: Base seed.
        n: Number of RNGs to create.

    Returns:
        List of n independent RandomState objects.
    """
    return [np.random.RandomState(seed + i) for i in range(n)]


def get_random_seed() -> int:
    """Generate a random seed value.

    Returns:
        A random integer suitable for seeding.
    """
    return np.random.randint(0, 2**31 - 1)
