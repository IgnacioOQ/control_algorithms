"""
Stock Management Environment - Inventory Control with Perishable Goods.

This environment simulates a multi-item inventory management system where:
- Multiple items have different decay times (shelf life) and storage sizes
- Storage capacity is limited
- Demand is stochastic (Poisson process)
- Items spoil after their decay time, incurring waste costs
- The agent decides how many units of each item to purchase

Classic Operations Research problem: the newsvendor/multi-item inventory problem.
"""

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import ContinuousSpace, DiscreteSpace, SimulationEnvironment, State, StepResult


@dataclass
class ItemConfig:
    """Configuration for a single item type.

    Attributes:
        name: Human-readable item name.
        decay_time: Shelf life in timesteps (after which item spoils).
        size: Storage space per unit.
        purchase_cost: Cost to buy one unit.
        sell_price: Revenue from selling one unit.
        demand_mean: Expected demand per timestep (Poisson λ).
        demand_std: Demand variability (for forecasting noise).
    """

    name: str = "item"
    decay_time: int = 5
    size: float = 1.0
    purchase_cost: float = 1.0
    sell_price: float = 2.0
    demand_mean: float = 10.0
    demand_std: float = 2.0


@dataclass
class StockManagementConfig:
    """Configuration for the Stock Management environment.

    Attributes:
        items: List of item configurations.
        storage_capacity: Total storage capacity (in units of space).
        max_inventory: Maximum inventory per item (for state normalization).
        max_order: Maximum units that can be ordered per item per step.
        holding_cost_rate: Cost per unit of storage used per timestep.
        spoilage_penalty: Penalty multiplier for spoiled goods (times purchase cost).
        stockout_penalty: Penalty for unfulfilled demand per unit.
        lead_time: Steps between ordering and receiving goods (0 = immediate).
        episode_length: Steps per episode (None = infinite).
    """

    items: List[ItemConfig] = field(default_factory=lambda: [
        ItemConfig(name="fresh_produce", decay_time=3, size=2.0,
                   purchase_cost=2.0, sell_price=5.0, demand_mean=8.0),
        ItemConfig(name="dairy", decay_time=5, size=1.0,
                   purchase_cost=1.5, sell_price=3.0, demand_mean=15.0),
        ItemConfig(name="frozen", decay_time=10, size=1.5,
                   purchase_cost=3.0, sell_price=6.0, demand_mean=5.0),
    ])
    storage_capacity: float = 100.0
    max_inventory: int = 50
    max_order: int = 20
    holding_cost_rate: float = 0.05
    spoilage_penalty: float = 1.0
    stockout_penalty: float = 0.5
    lead_time: int = 0
    episode_length: Optional[int] = 100


@dataclass
class InventorySlot:
    """Represents units of an item with a specific age.

    Attributes:
        quantity: Number of units at this age.
        age: Current age in timesteps.
    """

    quantity: int
    age: int


class StockManagementEnv(SimulationEnvironment):
    """Multi-Item Inventory Management Environment with Perishable Goods.

    The agent manages inventory for multiple item types, each with different
    decay times, sizes, costs, and demand patterns. The goal is to maximize
    profit by balancing:
    - Sales revenue (selling items before they spoil)
    - Purchase costs (buying inventory)
    - Holding costs (storage costs)
    - Spoilage costs (items that decay before being sold)
    - Stockout costs (unfulfilled demand)

    State space (continuous, normalized):
        For each item i (n_items total):
        - Inventory level (normalized by max_inventory)
        - Average age (normalized by decay_time)
        - Demand estimate (normalized)
        Plus:
        - Storage utilization (fraction of capacity used)
        Total: 3 * n_items + 1 dimensions

    Action space (continuous):
        For each item i: order quantity in [0, max_order]
        Total: n_items dimensions

    Reward:
        R = sales_revenue - purchase_costs - holding_costs - spoilage_costs - stockout_costs
    """

    def __init__(
        self,
        config: Optional[StockManagementConfig] = None,
        seed: Optional[int] = None
    ):
        """Initialize the environment.

        Args:
            config: Environment configuration.
            seed: Random seed.
        """
        self.config = config or StockManagementConfig()
        super().__init__(seed=seed)

    def _setup(self) -> None:
        """Set up observation and action spaces."""
        n_items = len(self.config.items)

        # State: [inv_1, age_1, demand_1, ..., inv_n, age_n, demand_n, storage_util]
        state_dim = 3 * n_items + 1
        self.observation_space = ContinuousSpace(
            low=np.zeros(state_dim, dtype=np.float32),
            high=np.ones(state_dim, dtype=np.float32),
        )

        # Action: [order_1, ..., order_n] continuous in [0, max_order]
        self.action_space = ContinuousSpace(
            low=np.zeros(n_items, dtype=np.float32),
            high=np.full(n_items, self.config.max_order, dtype=np.float32),
        )

        # Initialize internal state
        self._init_internal_state()

    def _init_internal_state(self) -> None:
        """Initialize internal simulation state."""
        n_items = len(self.config.items)

        # Inventory: list of InventorySlot lists for each item
        # Each item has a list of (quantity, age) pairs
        self.inventory: List[List[InventorySlot]] = [[] for _ in range(n_items)]

        # Pending orders (for lead time > 0)
        self.pending_orders: List[List[int]] = [
            [0] * (self.config.lead_time + 1) for _ in range(n_items)
        ]

        # Demand estimation (exponential moving average)
        self.demand_estimates = np.array(
            [item.demand_mean for item in self.config.items],
            dtype=np.float32
        )
        self._demand_ema_alpha = 0.3

        # Statistics
        self.step_count = 0
        self.total_revenue = 0.0
        self.total_costs = 0.0
        self.total_spoilage = 0
        self.total_stockouts = 0

    def reset(self, seed: Optional[int] = None) -> State:
        """Reset the environment."""
        if seed is not None:
            self.seed(seed)

        self._init_internal_state()

        # Start with some initial inventory
        for i, item in enumerate(self.config.items):
            initial_qty = int(item.demand_mean * 2)
            if initial_qty > 0:
                self.inventory[i].append(InventorySlot(quantity=initial_qty, age=0))

        return self._get_observation()

    def _get_total_inventory(self, item_idx: int) -> int:
        """Get total inventory for an item across all ages."""
        return sum(slot.quantity for slot in self.inventory[item_idx])

    def _get_average_age(self, item_idx: int) -> float:
        """Get weighted average age of inventory for an item."""
        total_qty = 0
        weighted_age = 0.0
        for slot in self.inventory[item_idx]:
            total_qty += slot.quantity
            weighted_age += slot.quantity * slot.age
        if total_qty == 0:
            return 0.0
        return weighted_age / total_qty

    def _get_storage_used(self) -> float:
        """Get total storage space currently used."""
        total = 0.0
        for i, item in enumerate(self.config.items):
            total += self._get_total_inventory(i) * item.size
        return total

    def step(self, action: np.ndarray) -> StepResult:
        """Execute one timestep of the environment dynamics.

        Order of operations:
        1. Receive pending orders (if lead_time > 0)
        2. Age all inventory
        3. Remove spoiled items
        4. Fulfill demand (FIFO - oldest first)
        5. Place new orders
        6. Calculate reward

        Args:
            action: Order quantities for each item.

        Returns:
            Tuple of (next_state, reward, done, info).
        """
        action = np.asarray(action, dtype=np.float32)

        # Clip and round actions to valid order quantities
        order_quantities = np.clip(action, 0, self.config.max_order)
        order_quantities = np.round(order_quantities).astype(int)

        n_items = len(self.config.items)

        # Track step metrics
        step_revenue = 0.0
        step_purchase_cost = 0.0
        step_holding_cost = 0.0
        step_spoilage_cost = 0.0
        step_stockout_cost = 0.0
        step_sales = np.zeros(n_items, dtype=int)
        step_spoiled = np.zeros(n_items, dtype=int)
        step_stockouts = np.zeros(n_items, dtype=int)

        # 1. Receive pending orders (for lead_time > 0)
        for i in range(n_items):
            received = self.pending_orders[i].pop(0)
            if received > 0:
                self.inventory[i].append(InventorySlot(quantity=received, age=0))
            self.pending_orders[i].append(0)

        # 2. Age all inventory
        for i in range(n_items):
            for slot in self.inventory[i]:
                slot.age += 1

        # 3. Remove spoiled items (FIFO - oldest slots first, already sorted by age)
        for i, item in enumerate(self.config.items):
            # Sort by age descending (oldest first for removal check)
            self.inventory[i].sort(key=lambda s: s.age, reverse=True)

            new_inventory = []
            for slot in self.inventory[i]:
                if slot.age > item.decay_time:
                    # Spoiled
                    step_spoiled[i] += slot.quantity
                    step_spoilage_cost += slot.quantity * item.purchase_cost * self.config.spoilage_penalty
                else:
                    new_inventory.append(slot)
            self.inventory[i] = new_inventory

        # 4. Generate and fulfill demand (FIFO - oldest first)
        for i, item in enumerate(self.config.items):
            # Generate stochastic demand (Poisson)
            demand = self.rng.poisson(item.demand_mean)

            # Update demand estimate
            self.demand_estimates[i] = (
                self._demand_ema_alpha * demand +
                (1 - self._demand_ema_alpha) * self.demand_estimates[i]
            )

            # Fulfill demand (oldest first = highest age first)
            self.inventory[i].sort(key=lambda s: s.age, reverse=True)

            remaining_demand = demand
            new_inventory = []

            for slot in self.inventory[i]:
                if remaining_demand <= 0:
                    new_inventory.append(slot)
                elif slot.quantity <= remaining_demand:
                    # Sell entire slot
                    step_sales[i] += slot.quantity
                    step_revenue += slot.quantity * item.sell_price
                    remaining_demand -= slot.quantity
                else:
                    # Partial sale
                    step_sales[i] += remaining_demand
                    step_revenue += remaining_demand * item.sell_price
                    slot.quantity -= remaining_demand
                    new_inventory.append(slot)
                    remaining_demand = 0

            self.inventory[i] = new_inventory

            # Stockout penalty for unfulfilled demand
            if remaining_demand > 0:
                step_stockouts[i] = remaining_demand
                step_stockout_cost += remaining_demand * item.sell_price * self.config.stockout_penalty

        # 5. Place new orders (check storage capacity)
        storage_available = self.config.storage_capacity - self._get_storage_used()

        for i, item in enumerate(self.config.items):
            order_qty = order_quantities[i]

            # Limit order by available storage
            max_by_storage = int(storage_available / item.size)
            order_qty = min(order_qty, max_by_storage)

            if order_qty > 0:
                step_purchase_cost += order_qty * item.purchase_cost

                if self.config.lead_time == 0:
                    # Immediate delivery
                    self.inventory[i].append(InventorySlot(quantity=order_qty, age=0))
                else:
                    # Add to pending orders
                    self.pending_orders[i][self.config.lead_time - 1] += order_qty

                storage_available -= order_qty * item.size

        # 6. Calculate holding cost
        for i, item in enumerate(self.config.items):
            inventory_qty = self._get_total_inventory(i)
            step_holding_cost += inventory_qty * item.size * self.config.holding_cost_rate

        # Calculate total reward (profit)
        reward = step_revenue - step_purchase_cost - step_holding_cost - step_spoilage_cost - step_stockout_cost

        # Update statistics
        self.step_count += 1
        self.total_revenue += step_revenue
        self.total_costs += step_purchase_cost + step_holding_cost + step_spoilage_cost + step_stockout_cost
        self.total_spoilage += step_spoiled.sum()
        self.total_stockouts += step_stockouts.sum()

        # Check if episode is done
        done = False
        if self.config.episode_length is not None:
            done = self.step_count >= self.config.episode_length

        info = {
            "revenue": step_revenue,
            "purchase_cost": step_purchase_cost,
            "holding_cost": step_holding_cost,
            "spoilage_cost": step_spoilage_cost,
            "stockout_cost": step_stockout_cost,
            "sales": step_sales.tolist(),
            "spoiled": step_spoiled.tolist(),
            "stockouts": step_stockouts.tolist(),
            "inventory_levels": [self._get_total_inventory(i) for i in range(n_items)],
            "storage_utilization": self._get_storage_used() / self.config.storage_capacity,
            "step": self.step_count,
        }

        return self._get_observation(), reward, done, info

    def _get_observation(self) -> np.ndarray:
        """Get current state observation (normalized)."""
        n_items = len(self.config.items)
        obs = np.zeros(3 * n_items + 1, dtype=np.float32)

        for i, item in enumerate(self.config.items):
            # Inventory level (normalized by max_inventory)
            inv = self._get_total_inventory(i)
            obs[3 * i] = min(inv / self.config.max_inventory, 1.0)

            # Average age (normalized by decay_time)
            avg_age = self._get_average_age(i)
            obs[3 * i + 1] = min(avg_age / item.decay_time, 1.0)

            # Demand estimate (normalized by 2x mean demand)
            obs[3 * i + 2] = min(self.demand_estimates[i] / (2 * item.demand_mean), 1.0)

        # Storage utilization
        obs[-1] = self._get_storage_used() / self.config.storage_capacity

        return obs

    def get_legal_actions(self) -> List[int]:
        """Get legal actions (for continuous space, returns empty list)."""
        return []

    def render(self) -> None:
        """Render current state."""
        print(f"Step: {self.step_count}")
        print(f"Storage: {self._get_storage_used():.1f}/{self.config.storage_capacity:.1f}")
        for i, item in enumerate(self.config.items):
            inv = self._get_total_inventory(i)
            avg_age = self._get_average_age(i)
            print(f"  {item.name}: {inv} units, avg age {avg_age:.1f}/{item.decay_time}")
        print(f"Total revenue: ${self.total_revenue:.2f}")
        print(f"Total costs: ${self.total_costs:.2f}")
        print(f"Total spoilage: {self.total_spoilage} units")

    def _get_state_repr(self) -> str:
        """Get string representation of state."""
        levels = [self._get_total_inventory(i) for i in range(len(self.config.items))]
        return f"Inventory: {levels}, Storage: {self._get_storage_used():.1f}/{self.config.storage_capacity:.1f}"

    def copy(self) -> "StockManagementEnv":
        """Create a deep copy for planning."""
        new_env = StockManagementEnv(config=self.config, seed=None)
        new_env.rng = np.random.RandomState()
        new_env.rng.set_state(self.rng.get_state())
        new_env.inventory = copy.deepcopy(self.inventory)
        new_env.pending_orders = copy.deepcopy(self.pending_orders)
        new_env.demand_estimates = self.demand_estimates.copy()
        new_env.step_count = self.step_count
        new_env.total_revenue = self.total_revenue
        new_env.total_costs = self.total_costs
        new_env.total_spoilage = self.total_spoilage
        new_env.total_stockouts = self.total_stockouts
        return new_env

    # -------------------------------------------------------------------------
    # Methods for MPC integration
    # -------------------------------------------------------------------------

    def get_dynamics_function(self):
        """Get a simplified dynamics function for MPC planning.

        Returns a function f(state, action, dt) -> next_state that models
        the expected (deterministic) dynamics for planning purposes.
        """
        n_items = len(self.config.items)

        def dynamics(state: np.ndarray, action: np.ndarray, dt: float = 1.0) -> np.ndarray:
            """Simplified deterministic dynamics for MPC.

            State: [inv_1, age_1, demand_1, ..., inv_n, age_n, demand_n, storage_util]
            Action: [order_1, ..., order_n]
            """
            state = np.asarray(state, dtype=np.float64)
            action = np.asarray(action, dtype=np.float64)
            next_state = state.copy()

            storage_used = 0.0

            for i, item in enumerate(self.config.items):
                # Denormalize
                inv = state[3 * i] * self.config.max_inventory
                avg_age = state[3 * i + 1] * item.decay_time
                demand_est = state[3 * i + 2] * 2 * item.demand_mean

                # Expected demand (use estimate)
                expected_demand = demand_est

                # Expected spoilage (items near decay time)
                expected_spoilage = inv * max(0, (avg_age + 1 - item.decay_time) / item.decay_time)

                # Order quantity
                order = np.clip(action[i], 0, self.config.max_order)

                # Next inventory = current - expected_demand - expected_spoilage + orders
                next_inv = max(0, inv - expected_demand - expected_spoilage + order)
                next_inv = min(next_inv, self.config.max_inventory)

                # Next average age (simplified: blend of aged inventory and new orders)
                if next_inv > 0:
                    remaining_inv = max(0, inv - expected_demand - expected_spoilage)
                    next_avg_age = (remaining_inv * (avg_age + 1) + order * 0) / (remaining_inv + order + 1e-6)
                else:
                    next_avg_age = 0

                # Demand estimate (assume mean-reverting)
                next_demand_est = 0.7 * demand_est + 0.3 * item.demand_mean

                # Normalize
                next_state[3 * i] = min(next_inv / self.config.max_inventory, 1.0)
                next_state[3 * i + 1] = min(next_avg_age / item.decay_time, 1.0)
                next_state[3 * i + 2] = min(next_demand_est / (2 * item.demand_mean), 1.0)

                storage_used += next_inv * item.size

            # Storage utilization
            next_state[-1] = min(storage_used / self.config.storage_capacity, 1.0)

            return next_state.astype(np.float32)

        return dynamics

    def get_cost_function(self):
        """Get a cost function for MPC (negative reward).

        Returns a function cost(state, action, x_ref) -> float.
        """
        n_items = len(self.config.items)

        def cost(state: np.ndarray, action: np.ndarray, x_ref: np.ndarray) -> float:
            """Stage cost for MPC (penalize deviation from target + action costs)."""
            state = np.asarray(state, dtype=np.float64)
            action = np.asarray(action, dtype=np.float64)

            total_cost = 0.0

            for i, item in enumerate(self.config.items):
                # Denormalize state
                inv = state[3 * i] * self.config.max_inventory
                avg_age = state[3 * i + 1] * item.decay_time
                demand_est = state[3 * i + 2] * 2 * item.demand_mean

                order = np.clip(action[i], 0, self.config.max_order)

                # Purchase cost
                total_cost += order * item.purchase_cost

                # Holding cost
                total_cost += inv * item.size * self.config.holding_cost_rate

                # Expected spoilage cost (items likely to spoil)
                if avg_age > item.decay_time * 0.7:
                    spoilage_risk = inv * (avg_age / item.decay_time)
                    total_cost += spoilage_risk * item.purchase_cost * self.config.spoilage_penalty * 0.5

                # Expected stockout cost - penalize more heavily
                if inv + order < demand_est:
                    expected_stockout = demand_est - inv - order
                    # Lost profit = sell_price - purchase_cost (margin we miss out on)
                    lost_margin = item.sell_price - item.purchase_cost
                    total_cost += expected_stockout * lost_margin * 2.0

                # Target inventory penalty (want to keep ~1.5x demand in stock)
                target_inv = demand_est * 1.5
                inv_after_order = inv + order
                inv_deviation = abs(inv_after_order - target_inv)
                total_cost += inv_deviation * 0.1

                # Expected revenue (negative cost) - only for items we expect to sell
                expected_sales = min(inv + order, demand_est)
                profit_margin = item.sell_price - item.purchase_cost
                total_cost -= expected_sales * profit_margin * 0.8  # Slight discount for uncertainty

            return total_cost

        return cost
