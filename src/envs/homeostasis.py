"""
Biological Homeostasis Environment - Bergman Minimal Model.

This environment simulates the glucose-insulin feedback loop using the
Bergman Minimal Model. The agent acts as an "Artificial Pancreas" controller,
delivering insulin to maintain blood glucose within safe ranges.

Key challenges:
- Non-linear, continuous dynamics with significant delays
- Safety-critical: hypoglycemia can be fatal
- Stiff ODEs requiring careful numerical integration
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .base import ContinuousSpace, SimulationEnvironment, State, StepResult
from ..utils.math_ops import OnlineNormalizer, rk4_step_with_control


@dataclass
class HomeostasisConfig:
    """Configuration for the Bergman Minimal Model environment.

    Physiological parameters are based on literature values.

    Attributes:
        # Model parameters (Bergman Minimal Model)
        p1: Insulin-independent glucose uptake rate (1/min).
        p2: Rate of insulin clearance from remote compartment (1/min).
        p3: Sensitivity of remote compartment to plasma insulin (μU/mL/min^2).
        n: Insulin decay rate (1/min).
        gamma_pancreas: Endogenous pancreatic insulin secretion rate.
        h: Glucose threshold for insulin secretion (mg/dL).
        
        # Basal values (equilibrium)
        G_b: Basal glucose (mg/dL).
        I_b: Basal insulin (μU/mL).
        
        # Target and safety ranges
        G_target: Target glucose (mg/dL).
        G_hypo: Hypoglycemia threshold (mg/dL).
        G_hyper: Hyperglycemia threshold (mg/dL).
        
        # Agent/simulation parameters
        dt_sim: Internal simulation timestep (minutes).
        dt_control: Control timestep / agent decision interval (minutes).
        max_insulin: Maximum insulin infusion rate (μU/mL/min).
        
        # Reward weights
        lambda_hypo: Hypoglycemia penalty weight.
        lambda_hyper: Hyperglycemia penalty weight.
        
        # Meal disturbances
        meal_probability: Probability of meal at each control step.
        meal_size_mean: Mean meal glucose load (mg/dL contribution).
        meal_size_std: Meal size standard deviation.
        
        # Episode length
        episode_length: Maximum episode length in control steps.
        
        # Type 1 diabetes mode (no endogenous insulin)
        type1_diabetes: Whether to disable endogenous insulin secretion.
    """

    # Bergman model parameters (typical values)
    p1: float = 0.028  # 1/min
    p2: float = 0.025  # 1/min
    p3: float = 5.0e-6  # μU/mL/min^2
    n: float = 0.23  # 1/min
    gamma_pancreas: float = 0.01  # Pancreatic response
    h: float = 80.0  # mg/dL threshold

    # Basal values
    G_b: float = 90.0  # mg/dL
    I_b: float = 7.0  # μU/mL

    # Target and safety
    G_target: float = 100.0  # mg/dL
    G_hypo: float = 70.0  # mg/dL
    G_hyper: float = 180.0  # mg/dL

    # Simulation parameters
    dt_sim: float = 1.0  # 1 minute internal step
    dt_control: float = 5.0  # 5 minute control interval
    max_insulin: float = 5.0  # μU/mL/min

    # Reward weights
    lambda_hypo: float = 100.0  # Heavy penalty for hypoglycemia
    lambda_hyper: float = 1.0  # Lighter penalty for hyperglycemia

    # Meals
    meal_probability: float = 0.05  # ~3 meals per day at 5-min intervals
    meal_size_mean: float = 50.0  # mg/dL equivalent
    meal_size_std: float = 20.0

    # Episode
    episode_length: int = 288  # 24 hours at 5-min intervals

    # Type 1 diabetes (no endogenous insulin)
    type1_diabetes: bool = True


class HomeostasisEnv(SimulationEnvironment):
    """Bergman Minimal Model Environment for Glucose-Insulin Control.

    The agent controls insulin infusion to maintain blood glucose in a safe range.
    Uses RK4 numerical integration for the coupled ODE system.

    State space (continuous, normalized):
        - G: Plasma glucose concentration (mg/dL)
        - X: Remote insulin effect (μU/mL)
        - I: Plasma insulin concentration (μU/mL)

    Action space (continuous, 1D):
        - Insulin infusion rate: [0, 1] normalized to [0, max_insulin]

    Reward:
        R = -|G - G_target|² - λ_hypo * I(G < 70) - λ_hyper * I(G > 180)
    """

    def __init__(
        self,
        config: Optional[HomeostasisConfig] = None,
        seed: Optional[int] = None,
        normalize_obs: bool = True,
    ):
        """Initialize the environment.

        Args:
            config: Environment configuration.
            seed: Random seed.
            normalize_obs: Whether to use online observation normalization.
        """
        self.config = config or HomeostasisConfig()
        self.normalize_obs = normalize_obs
        super().__init__(seed=seed)

    def _setup(self) -> None:
        """Set up observation and action spaces."""
        # State: [G, X, I] = 3 dimensions
        # Raw ranges are very different, so we use normalization
        self.observation_space = ContinuousSpace(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([500.0, 0.1, 100.0], dtype=np.float32),
        )

        # Action: normalized insulin infusion [0, 1]
        self.action_space = ContinuousSpace(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
        )

        # Online normalizer for observations
        if self.normalize_obs:
            self.normalizer = OnlineNormalizer(dim=3)
        else:
            self.normalizer = None

        # Initialize internal state
        self._init_internal_state()

    def _init_internal_state(self) -> None:
        """Initialize internal simulation state."""
        cfg = self.config

        # State vector: [G, X, I]
        # Start near basal values with some noise
        self.state = np.array(
            [
                cfg.G_b + self.rng.randn() * 10.0,  # Glucose
                0.0,  # Remote insulin effect
                cfg.I_b,  # Plasma insulin
            ],
            dtype=np.float64,
        )

        # Ensure glucose is positive
        self.state[0] = max(50.0, self.state[0])

        # Time tracking
        self.current_step = 0
        self.current_time = 0.0  # Minutes

        # Current meal disturbance
        self.current_meal = 0.0

        # Statistics
        self.glucose_history: List[float] = []
        self.hypo_events = 0
        self.hyper_events = 0

    def reset(self, seed: Optional[int] = None) -> State:
        """Reset the environment."""
        if seed is not None:
            self.seed(seed)

        self._init_internal_state()
        return self._get_observation()

    def _derivatives(
        self, state: np.ndarray, insulin_infusion: float, meal_rate: float
    ) -> np.ndarray:
        """Compute derivatives for the Bergman model.

        Args:
            state: Current state [G, X, I].
            insulin_infusion: External insulin infusion rate (μU/mL/min).
            meal_rate: Glucose appearance rate from meal (mg/dL/min).

        Returns:
            Derivatives [dG/dt, dX/dt, dI/dt].
        """
        cfg = self.config
        G, X, I = state

        # Glucose dynamics
        # dG/dt = -(p1 + X)·G + p1·G_b + D(t)
        dG = -(cfg.p1 + X) * G + cfg.p1 * cfg.G_b + meal_rate

        # Remote insulin dynamics
        # dX/dt = -p2·X + p3·(I - I_b)
        dX = -cfg.p2 * X + cfg.p3 * (I - cfg.I_b)

        # Plasma insulin dynamics
        # dI/dt = -n·(I - I_b) + γ·[G - h]⁺ + u(t)
        if cfg.type1_diabetes:
            # Type 1: no endogenous insulin secretion
            pancreatic_response = 0.0
        else:
            # Type 2: some endogenous insulin
            pancreatic_response = cfg.gamma_pancreas * max(0.0, G - cfg.h)

        dI = -cfg.n * (I - cfg.I_b) + pancreatic_response + insulin_infusion

        return np.array([dG, dX, dI], dtype=np.float64)

    def step(self, action: np.ndarray) -> StepResult:
        """Execute one control timestep."""
        cfg = self.config

        # Extract insulin infusion from action
        action = np.asarray(action, dtype=np.float32).flatten()
        insulin_norm = float(np.clip(action[0], 0.0, 1.0))
        insulin_infusion = insulin_norm * cfg.max_insulin

        # Check for meal event
        if self.rng.random() < cfg.meal_probability:
            meal_size = max(0.0, self.rng.randn() * cfg.meal_size_std + cfg.meal_size_mean)
            # Spread meal absorption over control interval
            self.current_meal = meal_size / cfg.dt_control
        else:
            # Decay meal effect
            self.current_meal *= 0.5

        # Simulate internal dynamics using RK4
        n_sim_steps = int(cfg.dt_control / cfg.dt_sim)
        for _ in range(n_sim_steps):
            self.state = rk4_step_with_control(
                state=self.state,
                dt=cfg.dt_sim,
                derivatives_func=self._derivatives,
                control=insulin_infusion,
                disturbance=self.current_meal,
            )

            # Enforce non-negativity
            self.state = np.maximum(self.state, [1.0, 0.0, 0.0])

        # Extract glucose for reward computation
        G = self.state[0]
        self.glucose_history.append(G)

        # Compute reward
        tracking_error = (G - cfg.G_target) ** 2

        # Safety penalties
        hypo_penalty = 0.0
        hyper_penalty = 0.0

        if G < cfg.G_hypo:
            hypo_penalty = cfg.lambda_hypo
            self.hypo_events += 1
        if G > cfg.G_hyper:
            hyper_penalty = cfg.lambda_hyper
            self.hyper_events += 1

        reward = -(tracking_error / 1000.0 + hypo_penalty + hyper_penalty)

        # Advance time
        self.current_step += 1
        self.current_time += cfg.dt_control

        # Update normalizer
        if self.normalizer is not None:
            self.normalizer.update(self.state)

        # Check termination
        done = self.current_step >= cfg.episode_length

        # Terminate early if glucose is dangerously low
        if G < 40.0:
            done = True
            reward -= 500.0  # Severe penalty for dangerous hypoglycemia

        info = {
            "glucose": G,
            "insulin": self.state[2],
            "remote_insulin": self.state[1],
            "meal": self.current_meal,
            "hypo_events": self.hypo_events,
            "hyper_events": self.hyper_events,
            "in_range": cfg.G_hypo <= G <= cfg.G_hyper,
        }

        return self._get_observation(), reward, done, info

    def _get_observation(self) -> np.ndarray:
        """Get current state observation."""
        if self.normalizer is not None and self.normalizer.count > 1:
            return self.normalizer.normalize(self.state).astype(np.float32)
        else:
            # Simple normalization if no running stats yet
            return (self.state / np.array([100.0, 0.01, 20.0])).astype(np.float32)

    def get_legal_actions(self) -> List[int]:
        """Not applicable for continuous action space."""
        return []

    def render(self) -> None:
        """Render current state."""
        cfg = self.config
        G, X, I = self.state
        
        # Determine glucose status
        if G < cfg.G_hypo:
            status = "⚠️ HYPO"
        elif G > cfg.G_hyper:
            status = "⚠️ HYPER"
        else:
            status = "✓ Normal"
        
        hours = self.current_time / 60.0
        print(f"Time: {hours:.1f}h (Step {self.current_step})")
        print(f"  Glucose: {G:.1f} mg/dL [{status}]")
        print(f"  Insulin: {I:.2f} μU/mL")
        print(f"  Remote X: {X:.4f}")
        print(f"  Meal rate: {self.current_meal:.1f} mg/dL/min")
        print(f"  Events - Hypo: {self.hypo_events}, Hyper: {self.hyper_events}")

    def _get_state_repr(self) -> str:
        """Get string representation of state."""
        G, X, I = self.state
        return f"G={G:.1f}, I={I:.2f}, X={X:.4f}"

    def get_time_in_range(self) -> float:
        """Calculate percentage of time glucose was in target range.

        Returns:
            Fraction of time with 70 <= G <= 180.
        """
        if not self.glucose_history:
            return 0.0

        cfg = self.config
        in_range = sum(
            1 for g in self.glucose_history if cfg.G_hypo <= g <= cfg.G_hyper
        )
        return in_range / len(self.glucose_history)

    def get_glycemic_variability(self) -> float:
        """Calculate coefficient of variation for glucose.

        Returns:
            CV = std / mean of glucose values.
        """
        if len(self.glucose_history) < 2:
            return 0.0

        mean_g = np.mean(self.glucose_history)
        std_g = np.std(self.glucose_history)
        return std_g / mean_g if mean_g > 0 else 0.0
