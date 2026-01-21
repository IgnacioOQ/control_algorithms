# Controllers module - Classical and Optimal Control implementations
from .base import BaseController
from .pid import PIDController
from .lqr import LQRController, discretize_system, solve_dare
from .mpc import MPCController

__all__ = [
    "BaseController",
    "PIDController",
    "LQRController",
    "MPCController",
    "discretize_system",
    "solve_dare",
]
