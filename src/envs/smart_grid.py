"""
Smart Grid Environment - Battery Energy Storage System with Demand Response.

This environment simulates energy management with:
- Battery storage with efficiency losses and constraints
- Ornstein-Uhlenbeck price process for realistic market simulation
- Sinusoidal load profiles with stochastic noise
- Price arbitrage opportunities

The agent controls charge/discharge actions to maximize profit.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .base import ContinuousSpace, SimulationEnvironment, State, StepResult


@dataclass
class SmartGridConfig:
    """Configuration for the Smart Grid environment.

    Attributes:
        battery_capacity: Maximum battery capacity (kWh).
        soc_min: Minimum state of charge (fraction 0-1).
        soc_max: Maximum state of charge (fraction 0-1).
        charge_efficiency: Charging efficiency (η_ch).
        discharge_efficiency: Discharging efficiency (η_dis).
        max_power: Maximum charge/discharge power (kW).
        leakage_rate: Self-discharge rate per step (fraction).
        
        price_mean: Mean electricity price ($/kWh).
        price_theta: OU process mean reversion rate.
        price_sigma: OU process volatility.
        
        load_base: Base load demand (kW).
        load_amplitude: Daily load variation amplitude (kW).
        load_noise_std: Load noise standard deviation (kW).
        
        renewable_capacity: Peak renewable generation (kW).
        renewable_variability: Variability in renewable output.
        
        step_duration: Duration of each step (hours).
        horizon: Number of steps for price forecast in state.
        episode_length: Maximum episode length in steps.
    """

    # Battery parameters
    battery_capacity: float = 100.0  # kWh
    soc_min: float = 0.1  # 10% minimum
    soc_max: float = 0.9  # 90% maximum
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    max_power: float = 25.0  # kW
    leakage_rate: float = 0.001  # 0.1% per step

    # Price dynamics (Ornstein-Uhlenbeck)
    price_mean: float = 0.15  # $/kWh
    price_theta: float = 0.1  # Mean reversion rate
    price_sigma: float = 0.03  # Volatility

    # Load profile
    load_base: float = 50.0  # kW
    load_amplitude: float = 30.0  # kW variation
    load_noise_std: float = 5.0  # kW

    # Renewables
    renewable_capacity: float = 40.0  # kW peak
    renewable_variability: float = 0.3

    # Simulation
    step_duration: float = 0.25  # 15 minutes = 0.25 hours
    horizon: int = 8  # 2 hours lookahead for price forecast
    episode_length: int = 96 * 7  # One week at 15-min intervals


class SmartGridEnv(SimulationEnvironment):
    """Smart Grid / Demand Response Environment.

    The agent controls battery charging/discharging to:
    - Buy energy when prices are low
    - Sell/consume when prices are high
    - Meet load demands while minimizing costs

    State space (continuous):
        - SoC: Current battery state of charge (normalized)
        - L_t: Current load demand (normalized)
        - P_gen: Current renewable generation (normalized)
        - C_t: Current electricity price (normalized)
        - C_{t+1...t+H}: Future price forecast (normalized, H values)

    Action space (continuous, 1D):
        - Power command: [-1, 1] where -1 = max discharge, +1 = max charge

    Reward:
        Revenue from price arbitrage minus constraint violation penalties.
    """

    def __init__(self, config: Optional[SmartGridConfig] = None, seed: Optional[int] = None):
        """Initialize the environment."""
        self.config = config or SmartGridConfig()
        super().__init__(seed=seed)

    def _setup(self) -> None:
        """Set up observation and action spaces."""
        cfg = self.config
        
        # State: [SoC, Load, Generation, Price, FuturePrices...]
        # = 1 + 1 + 1 + 1 + horizon = 4 + horizon dimensions
        state_dim = 4 + cfg.horizon
        self.observation_space = ContinuousSpace(
            low=np.zeros(state_dim, dtype=np.float32),
            high=np.ones(state_dim, dtype=np.float32),
        )
        
        # Action: normalized power [-1, 1]
        self.action_space = ContinuousSpace(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
        )
        
        # Initialize internal state
        self._init_internal_state()

    def _init_internal_state(self) -> None:
        """Initialize internal simulation state."""
        cfg = self.config
        
        # Battery state of charge (start at 50%)
        self.soc = 0.5
        
        # Time tracking
        self.current_step = 0
        self.current_hour = 0.0  # Hour of day (0-24)
        
        # Price process (Ornstein-Uhlenbeck)
        self.current_price = cfg.price_mean
        
        # Generate price trajectory for lookahead
        self._generate_price_trajectory()
        
        # Current load and generation
        self.current_load = self._compute_load()
        self.current_generation = self._compute_generation()

    def _generate_price_trajectory(self, length: int = 1000) -> None:
        """Pre-generate price trajectory using OU process for lookahead."""
        cfg = self.config
        dt = cfg.step_duration
        
        self.price_trajectory = np.zeros(length)
        self.price_trajectory[0] = self.current_price
        
        for i in range(1, length):
            dW = self.rng.randn() * np.sqrt(dt)
            prev = self.price_trajectory[i - 1]
            self.price_trajectory[i] = (
                prev
                + cfg.price_theta * (cfg.price_mean - prev) * dt
                + cfg.price_sigma * dW
            )
            # Ensure price is positive
            self.price_trajectory[i] = max(0.01, self.price_trajectory[i])

    def _compute_load(self) -> float:
        """Compute current load demand based on time of day."""
        cfg = self.config
        
        # Sinusoidal daily pattern (peak at noon, trough at night)
        hour_angle = 2 * np.pi * self.current_hour / 24.0
        base_load = cfg.load_base + cfg.load_amplitude * np.sin(hour_angle - np.pi / 2)
        
        # Add noise
        noise = self.rng.randn() * cfg.load_noise_std
        
        return max(0.0, base_load + noise)

    def _compute_generation(self) -> float:
        """Compute current renewable generation (solar-like profile)."""
        cfg = self.config
        
        # Solar-like profile: generation only during day, peak at noon
        if 6 <= self.current_hour <= 18:
            hour_angle = np.pi * (self.current_hour - 6) / 12.0
            base_gen = cfg.renewable_capacity * np.sin(hour_angle)
        else:
            base_gen = 0.0
        
        # Add variability (clouds, etc.)
        variability = 1.0 + self.rng.randn() * cfg.renewable_variability
        variability = max(0.0, min(2.0, variability))
        
        return max(0.0, base_gen * variability)

    def reset(self, seed: Optional[int] = None) -> State:
        """Reset the environment."""
        if seed is not None:
            self.seed(seed)
        
        self._init_internal_state()
        return self._get_observation()

    def step(self, action: np.ndarray) -> StepResult:
        """Execute one timestep."""
        cfg = self.config
        
        # Extract power command from action
        action = np.asarray(action, dtype=np.float32).flatten()
        power_cmd = float(np.clip(action[0], -1.0, 1.0)) * cfg.max_power
        
        # Determine charging or discharging
        if power_cmd > 0:
            # Charging
            actual_power, energy_change = self._apply_charging(power_cmd)
        else:
            # Discharging
            actual_power, energy_change = self._apply_discharging(-power_cmd)
            actual_power = -actual_power
        
        # Update battery SoC
        old_soc = self.soc
        self.soc += energy_change / cfg.battery_capacity
        
        # Apply leakage
        self.soc *= (1.0 - cfg.leakage_rate)
        
        # Clamp SoC to valid range
        self.soc = np.clip(self.soc, cfg.soc_min, cfg.soc_max)
        
        # Calculate reward (arbitrage profit)
        # Positive power (charging) = buying = negative revenue
        # Negative power (discharging) = selling = positive revenue
        energy_cost = actual_power * cfg.step_duration * self.current_price
        reward = -energy_cost  # Profit from selling, cost from buying
        
        # Penalty for constraint violations (soft constraints)
        if old_soc <= cfg.soc_min and power_cmd < 0:
            reward -= 1.0  # Tried to discharge empty battery
        if old_soc >= cfg.soc_max and power_cmd > 0:
            reward -= 1.0  # Tried to charge full battery
        
        # Advance time
        self.current_step += 1
        self.current_hour = (self.current_hour + cfg.step_duration) % 24.0
        self.current_price = self.price_trajectory[
            min(self.current_step, len(self.price_trajectory) - 1)
        ]
        self.current_load = self._compute_load()
        self.current_generation = self._compute_generation()
        
        # Check termination
        done = self.current_step >= cfg.episode_length
        
        info = {
            "soc": self.soc,
            "price": self.current_price,
            "load": self.current_load,
            "generation": self.current_generation,
            "power_applied": actual_power,
        }
        
        return self._get_observation(), reward, done, info

    def _apply_charging(self, power: float) -> tuple:
        """Apply charging power. Returns (actual_power, energy_change)."""
        cfg = self.config
        
        # Limit by max power
        power = min(power, cfg.max_power)
        
        # Limit by available capacity
        available_capacity = (cfg.soc_max - self.soc) * cfg.battery_capacity
        max_energy = available_capacity / cfg.charge_efficiency
        max_power = max_energy / cfg.step_duration
        
        actual_power = min(power, max_power)
        energy_in = actual_power * cfg.step_duration * cfg.charge_efficiency
        
        return actual_power, energy_in

    def _apply_discharging(self, power: float) -> tuple:
        """Apply discharging power. Returns (actual_power, energy_change)."""
        cfg = self.config
        
        # Limit by max power
        power = min(power, cfg.max_power)
        
        # Limit by available energy
        available_energy = (self.soc - cfg.soc_min) * cfg.battery_capacity
        max_energy_out = available_energy * cfg.discharge_efficiency
        max_power = max_energy_out / cfg.step_duration
        
        actual_power = min(power, max_power)
        energy_out = actual_power * cfg.step_duration / cfg.discharge_efficiency
        
        return actual_power, -energy_out

    def _get_observation(self) -> np.ndarray:
        """Get current state observation (normalized)."""
        cfg = self.config
        
        # Normalize values to [0, 1]
        soc_norm = (self.soc - cfg.soc_min) / (cfg.soc_max - cfg.soc_min)
        load_norm = self.current_load / (cfg.load_base + cfg.load_amplitude + 3 * cfg.load_noise_std)
        gen_norm = self.current_generation / cfg.renewable_capacity
        
        # Price normalization (assume price ranges from 0 to 2x mean)
        price_norm = self.current_price / (2 * cfg.price_mean)
        
        # Future prices
        future_prices = []
        for i in range(1, cfg.horizon + 1):
            idx = min(self.current_step + i, len(self.price_trajectory) - 1)
            future_prices.append(self.price_trajectory[idx] / (2 * cfg.price_mean))
        
        # Clip all to [0, 1]
        obs = np.array(
            [soc_norm, load_norm, gen_norm, price_norm] + future_prices,
            dtype=np.float32,
        )
        return np.clip(obs, 0.0, 1.0)

    def get_legal_actions(self) -> List[int]:
        """Not applicable for continuous action space."""
        return []

    def render(self) -> None:
        """Render current state."""
        cfg = self.config
        print(f"Step: {self.current_step}, Hour: {self.current_hour:.1f}")
        print(f"  SoC: {self.soc * 100:.1f}% ({self.soc * cfg.battery_capacity:.1f} kWh)")
        print(f"  Price: ${self.current_price:.4f}/kWh")
        print(f"  Load: {self.current_load:.1f} kW")
        print(f"  Generation: {self.current_generation:.1f} kW")

    def _get_state_repr(self) -> str:
        """Get string representation of state."""
        return f"SoC: {self.soc * 100:.1f}%, Price: ${self.current_price:.3f}"

    def get_optimal_baseline(self) -> float:
        """Compute optimal profit with perfect foresight using simple heuristic.
        
        This provides a baseline for comparing agent performance.
        Returns the theoretical optimal profit assuming perfect price knowledge.
        """
        cfg = self.config
        
        # Simple heuristic: charge when price is below median, discharge when above
        prices = self.price_trajectory[:cfg.episode_length]
        median_price = np.median(prices)
        
        # Simulate with perfect strategy
        soc = 0.5
        total_profit = 0.0
        
        for step, price in enumerate(prices):
            if price < median_price * 0.9:  # Low price: charge
                power = cfg.max_power
                energy = power * cfg.step_duration * cfg.charge_efficiency
                new_soc = min(cfg.soc_max, soc + energy / cfg.battery_capacity)
                actual_energy = (new_soc - soc) * cfg.battery_capacity
                cost = actual_energy / cfg.charge_efficiency * price
                total_profit -= cost
                soc = new_soc
            elif price > median_price * 1.1:  # High price: discharge
                power = cfg.max_power
                energy_available = (soc - cfg.soc_min) * cfg.battery_capacity
                energy_out = min(power * cfg.step_duration, energy_available * cfg.discharge_efficiency)
                new_soc = soc - energy_out / cfg.discharge_efficiency / cfg.battery_capacity
                revenue = energy_out * price
                total_profit += revenue
                soc = new_soc
        
        return total_profit
