# Control Algorithms Skill
- status: active
- type: how-to
- description: Implementation instructions for control-theoretic methods in the RL Simulation System — PID, LQR, MPC (done) and remaining work: threshold controllers, hybrid RL-control methods (Phase 3).
- label: [agent, skill]
- injection: procedural
- volatility: evolving
- last_checked: 2026-05-01
<!-- content -->

**Role:** You are a specialist in classical and modern control theory implementation.

**Goal:** Implement and integrate control-theoretic methods into the Ab Initio RL Simulation System. Phases 1 (PID) and 2 (LQR, MPC) are complete. This document covers remaining work: threshold controllers (Phase 1.2) and hybrid RL-control methods (Phase 3).

For algorithm background, design rationale, and real-world domain guidance, read [CONTROL_ALGORITHMS_EXPLANATION.md](CONTROL_ALGORITHMS_EXPLANATION.md) first.

## Core Constraints

1. **Interface compliance:** All controllers must implement `BaseAgent` from `src/agents/base.py`.
2. **No modification of existing agents:** Do not modify files in `src/agents/` except to add new files.
3. **Classical controllers in `src/controllers/`:** Threshold controllers go in `src/controllers/threshold.py`.
4. **Hybrid agents in `src/agents/`:** Residual RL, CBF, and gain-scheduled agents go in `src/agents/`.
5. **Testing required:** Every new controller must have tests in `tests/test_controllers.py` (classical) or `tests/test_agents.py` (hybrid).
6. **Log completions:** Add a WORKLOG.md entry after significant implementations.

## Phase 1.2: Threshold / Heuristic Controllers

**File:** `src/controllers/threshold.py` (create new)  
**Target environment:** Server Load (`src/envs/server_load.py`)

```python
class ShortestQueueAgent(BaseAgent):
    """Route to the server with the minimum current queue length."""

    def select_action(self, state: np.ndarray) -> int:
        # state contains queue lengths for each server
        # return index of server with min queue
        pass

    def update(self, *args, **kwargs) -> dict:
        return {}


class PowerOfTwoChoices(BaseAgent):
    """
    Sample 2 servers at random, route to the shorter queue.
    Achieves O(log log n) max queue vs O(log n) for pure random.
    """

    def __init__(self, n_servers: int, seed: int = 0):
        pass

    def select_action(self, state: np.ndarray) -> int:
        pass

    def update(self, *args, **kwargs) -> dict:
        return {}


class ThresholdAgent(BaseAgent):
    """
    Threshold routing with hysteresis to prevent oscillation.
    Route to server k if queue_k < threshold, else round-robin.
    """

    def __init__(self, threshold: int, n_servers: int):
        pass

    def select_action(self, state: np.ndarray) -> int:
        pass

    def update(self, *args, **kwargs) -> dict:
        return {}
```

**Implementation notes:**
- `state` layout in `ServerLoadEnv`: `[q_0, q_1, ..., q_{k-1}, busy_0, ..., busy_{k-1}, lambda_obs, mean_latency]`. Queue lengths are the first `n_servers` elements.
- `ShortestQueueAgent` and `PowerOfTwoChoices` require no tuning — implement as pure heuristics with no learnable parameters.
- `ThresholdAgent`: expose `threshold` as a constructor parameter so it can be swept in benchmarks.

## Phase 3: Hybrid Control-RL Methods

### 3.1 Residual Policy Agent

**File:** `src/agents/residual_rl.py` (create new)

```python
class ResidualPolicyAgent(BaseAgent):
    """
    Combine a base controller with a learned residual.

    action = clip(base_controller(state) + residual_scale * residual_agent(state), bounds)

    The RL agent learns to correct the base controller's errors.
    Limit residual_scale to keep exploration safe during training.
    """

    def __init__(
        self,
        base_controller: BaseAgent,
        residual_agent: BaseAgent,    # typically PPO for continuous, DQN for discrete
        residual_scale: float = 0.1,
        action_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        pass

    def select_action(self, state: np.ndarray) -> np.ndarray:
        base = self.base_controller.select_action(state)
        residual = self.residual_agent.select_action(state)
        action = base + self.residual_scale * residual
        if self.action_bounds is not None:
            action = np.clip(action, self.action_bounds[0], self.action_bounds[1])
        return action

    def update(self, *args, **kwargs) -> dict:
        # Delegate to the residual agent's update; base controller is frozen
        return self.residual_agent.update(*args, **kwargs)

    def store(self, *args, **kwargs):
        self.residual_agent.store(*args, **kwargs)
```

### 3.2 Control Barrier Function Safe Agent

**File:** `src/agents/cbf_rl.py` (create new)  
**Priority use case:** Homeostasis (prevent hypoglycemia: G < 50 mg/dL)

```python
class CBFSafeAgent(BaseAgent):
    """
    Wrap an RL agent with a CBF safety filter.

    The filter solves a small QP at each step to find the nearest safe action:
        min  ||u - u_rl||²
        s.t. dh/dx · f(x, u) + alpha * h(x) ≥ 0
    where h(x) ≥ 0 defines the safe set.
    """

    def __init__(
        self,
        rl_agent: BaseAgent,
        barrier_fn: Callable[[np.ndarray], float],   # h(x), ≥ 0 = safe
        barrier_grad_fn: Callable[[np.ndarray], np.ndarray],  # ∂h/∂x
        dynamics_fn: Callable,                        # f(x, u)
        alpha: float = 1.0,
        action_bounds: Optional[Tuple] = None,
    ):
        pass

    def select_action(self, state: np.ndarray) -> np.ndarray:
        u_rl = self.rl_agent.select_action(state)
        # project u_rl onto safe set using QP
        return self._project_safe(state, u_rl)

    def _project_safe(self, x: np.ndarray, u_nominal: np.ndarray) -> np.ndarray:
        """Solve safety QP: min ||u - u_nominal||² s.t. CBF constraint."""
        pass

    def update(self, *args, **kwargs) -> dict:
        return self.rl_agent.update(*args, **kwargs)
```

**Homeostasis barrier function:**
```python
def glucose_barrier(state: np.ndarray, G_min: float = 50.0) -> float:
    """h(x) = G - G_min.  Safe set: G ≥ G_min."""
    G = state[0]  # glucose is first state element in HomeostasisEnv
    return G - G_min
```

### 3.3 Gain-Scheduled Agent

**File:** `src/agents/gain_scheduled.py` (create new)

```python
class GainScheduledAgent(BaseAgent):
    """
    Switch between multiple controllers based on operating region.

    regions: list of (condition_fn, controller) pairs.
    The first condition_fn(state) that returns True selects its controller.
    Falls back to default_controller if no condition matches.
    """

    def __init__(
        self,
        regions: List[Tuple[Callable[[np.ndarray], bool], BaseAgent]],
        default_controller: BaseAgent,
    ):
        pass

    def select_action(self, state: np.ndarray) -> np.ndarray:
        for condition, controller in self.regions:
            if condition(state):
                return controller.select_action(state)
        return self.default_controller.select_action(state)

    def update(self, *args, **kwargs) -> dict:
        return {}
```

## Testing Strategy

### Phase 1.2 tests (add to `tests/test_controllers.py`)

```python
class TestThresholdControllers:
    def test_shortest_queue_selects_min(self):
        """ShortestQueueAgent always returns index of minimum queue."""
        agent = ShortestQueueAgent(n_servers=3)
        state = np.array([5, 2, 8, 1, 1, 1, 1.0, 0.1])
        assert agent.select_action(state) == 1  # queue index 1 has length 2

    def test_power_of_two_valid_action(self):
        """PowerOfTwoChoices always returns a valid server index."""
        agent = PowerOfTwoChoices(n_servers=4, seed=42)
        state = np.zeros(10)
        action = agent.select_action(state)
        assert 0 <= action < 4

    def test_threshold_below_threshold(self):
        """ThresholdAgent routes to first server below threshold."""
        agent = ThresholdAgent(threshold=5, n_servers=3)
        state = np.array([10, 3, 10, 0, 1, 0, 1.0, 0.1])
        assert agent.select_action(state) == 1
```

### Phase 3 tests (add to `tests/test_agents.py`)

```python
class TestResidualPolicyAgent:
    def test_action_is_base_plus_residual(self):
        """Output equals base + residual_scale * residual (before clipping)."""
        pass

    def test_clipping_respects_bounds(self):
        """Output is always within action_bounds."""
        pass

class TestCBFSafeAgent:
    def test_safe_action_unchanged(self):
        """If RL action is already safe, CBF should not modify it."""
        pass

    def test_unsafe_action_projected(self):
        """CBF must project unsafe action to the safe set boundary."""
        pass

    def test_glucose_never_below_minimum(self):
        """Run on HomeostasisEnv; glucose must stay >= G_min throughout."""
        pass
```

## Implementation Checklist

### Phase 1: Classical Controllers
- [x] `PIDController` with anti-windup (`src/controllers/pid.py`)
- [x] Tests for PID — `tests/test_controllers.py` (30 tests, passing)
- [ ] `ShortestQueueAgent`, `PowerOfTwoChoices`, `ThresholdAgent` (`src/controllers/threshold.py`)
- [ ] Tests for threshold controllers

### Phase 2: Optimal Control
- [x] `LQRController` with DARE solver (`src/controllers/lqr.py`)
- [x] `MPCController` with scipy SLSQP + `LinearMPC` QP (`src/controllers/mpc.py`)
- [x] Controllability / stabilizability checks in `src/controllers/lqr.py`
- [ ] Integration test: LQR on linearized Smart Grid
- [ ] Integration test: MPC on Homeostasis with glucose safety constraint

### Phase 3: Hybrid Methods
- [ ] `ResidualPolicyAgent` (`src/agents/residual_rl.py`)
- [ ] `CBFSafeAgent` for Homeostasis (`src/agents/cbf_rl.py`)
- [ ] `GainScheduledAgent` (`src/agents/gain_scheduled.py`)
- [ ] Benchmark: ResidualPolicyAgent(PID, PPO) vs. pure PPO on Homeostasis
- [ ] Benchmark: CBFSafeAgent vs. unconstrained PPO — hypoglycemia rate comparison
