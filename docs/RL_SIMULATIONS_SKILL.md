# RL Simulations Skill
- status: active
- type: how-to
- description: Implementation cookbook for building simulation environments from scratch — SimulationEnvironment interface, dynamics patterns (DES, ODE, time-step), parameterization from data, and testing strategy.
- label: [agent, skill]
- injection: procedural
- volatility: stable
- last_checked: 2026-05-01
<!-- content -->

**Role:** You are implementing a simulation environment for an optimization problem.

**Goal:** Build a `SimulationEnvironment`-compliant class that faithfully models the problem dynamics, supports reproducible seeding, and can be plugged directly into any agent in this repo (RL or classical controller).

For domain patterns and state/reward templates, read [RL_SIMULATIONS_REF.md](RL_SIMULATIONS_REF.md) first to identify which simulation pattern fits your problem.

## Core Constraints

1. **No high-level RL libraries.** No Gymnasium, Stable Baselines, or RLLib. Every transition, stochastic process, and integration step is explicit.
2. **Implement `SimulationEnvironment` protocol** from `src/envs/base.py`. The four required methods are `reset`, `step`, `get_legal_actions`, `render`.
3. **Seeding is mandatory.** Every stochastic component uses a `np.random.RandomState` initialized from a seed. No global `np.random` calls.
4. **New env file in `src/envs/`.** Add an import to `src/envs/__init__.py` and register it in `src/config.py`.
5. **Tests required** in `tests/test_envs.py`: reset/step contract, state shapes, reward bounds, seeding reproducibility.
6. **Log the implementation** in `WORKLOG.md`.

---

## Step 1: Define the Interface

All environments implement this protocol (see `src/envs/base.py`):

```python
class SimulationEnvironment(Protocol):
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Re-initialize stochastic state, return initial observation."""
        ...

    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        """Apply action, advance dynamics, return (obs, reward, done, info)."""
        ...

    def get_legal_actions(self) -> List:
        """Return valid actions from the current state (for MCTS/bandits)."""
        ...

    def render(self) -> None:
        """Print or plot current state (text-based is fine)."""
        ...
```

Start with a skeleton and fill in each method:

```python
class MyEnv:
    def __init__(self, config: MyConfig):
        self.config = config
        self.rng = np.random.RandomState()
        self._state = None

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        # initialize state variables
        self._state = self._initial_state()
        self._step_count = 0
        return self._observe()

    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        # 1. validate / clip action
        # 2. apply dynamics
        # 3. compute reward
        # 4. check termination
        self._step_count += 1
        done = self._step_count >= self.config.max_steps
        return self._observe(), reward, done, {}

    def _observe(self) -> np.ndarray:
        """Build the observation vector from internal state."""
        return np.array([...], dtype=np.float32)
```

---

## Step 2: Implement Transition Dynamics

Choose the dynamics pattern that matches your problem (see [RL_SIMULATIONS_REF.md](RL_SIMULATIONS_REF.md)):

### Pattern A: Discrete time-step (inventory, storage, market)

The simplest pattern. Each `step()` call advances by one fixed time unit Δt.

```python
def step(self, action):
    action = np.clip(action, self.config.action_low, self.config.action_high)

    # deterministic dynamics
    self._stock = self._stock + action - self._demand

    # stochastic element — use self.rng, not np.random
    self._demand = self.rng.poisson(self.config.mean_demand)
    self._price  = self._price + self.config.theta * (self.config.mu - self._price) \
                   + self.config.sigma * self.rng.randn()

    # clip physical constraints
    self._stock = np.clip(self._stock, 0, self.config.capacity)

    reward = self._compute_reward(action)
    done = self._step_count >= self.config.max_steps
    return self._observe(), reward, done, {}
```

### Pattern B: Discrete Event Simulation (queueing, routing)

The agent's `step()` spans a wall-clock interval Δt; inside it, the environment processes micro-events.

```python
def step(self, action: int):
    # action = server index to route next arriving job
    self._route_job(action)

    elapsed = 0.0
    n_dropped = 0
    total_latency = 0.0

    while elapsed < self.config.dt_step:
        # time to next arrival
        t_arr = -np.log(self.rng.uniform()) / self.config.lam

        # time to next departure on each busy server
        t_deps = [
            -np.log(self.rng.uniform()) / self.config.mu
            if self._busy[i] else np.inf
            for i in range(self.config.n_servers)
        ]

        t_next = min(t_arr, *t_deps)

        if elapsed + t_next > self.config.dt_step:
            elapsed = self.config.dt_step
            break

        elapsed += t_next

        if t_next == t_arr:
            # arrival event
            if all(q >= self.config.buffer_size for q in self._queues):
                n_dropped += 1
            else:
                self._queues[self._select_server()] += 1
        else:
            # departure event
            i = np.argmin(t_deps)
            self._queues[i] = max(0, self._queues[i] - 1)
            self._busy[i] = self._queues[i] > 0
            total_latency += elapsed

    reward = self._compute_reward(n_dropped, total_latency)
    return self._observe(), reward, False, {"n_dropped": n_dropped}
```

### Pattern C: ODE system with RK4 (process control, biological)

The environment integrates differential equations over each time step. Use RK4 for stiff systems.

```python
def step(self, action: float):
    action = np.clip(action, 0, self.config.u_max)

    # integrate ODEs from t to t + dt_control
    self._state = rk4_step(
        state=self._state,
        u=action,
        disturbance=self._sample_disturbance(),
        dt=self.config.dt_control,
        derivatives_fn=self._derivatives,
    )

    reward = self._compute_reward()
    done = (self._step_count >= self.config.max_steps
            or self._unsafe_termination())
    return self._observe(), reward, done, {}

def _derivatives(self, state: np.ndarray, u: float, d: float) -> np.ndarray:
    """Return dx/dt for each state variable."""
    x1, x2, x3 = state
    dx1 = ...   # f1(x1, x2, x3, u, d)
    dx2 = ...   # f2(x1, x2, x3, u, d)
    dx3 = ...   # f3(x1, x2, x3, u, d)
    return np.array([dx1, dx2, dx3])
```

The `rk4_step` utility is in `src/utils/math_ops.py`:

```python
from src.utils.math_ops import rk4_step
```

---

## Step 3: Design the Reward Function

**Checklist for reward design:**

- [ ] Is the reward dense (signal every step) or sparse (signal at episode end)? Dense is almost always better for learning speed.
- [ ] Are there safety constraints? Add a large penalty term — at least 10× the magnitude of the tracking reward.
- [ ] Are cost components on different scales? Normalize them (divide by typical magnitude) so no single term dominates.
- [ ] Is the goal economic (profit/cost) or regulatory (stay near a setpoint)? Economic rewards need forecasting in the state.

```python
def _compute_reward(self) -> float:
    tracking_error  = -np.sum((self._state - self.config.target) ** 2)
    safety_penalty  = -self.config.lambda_safe * float(self._is_unsafe())
    control_cost    = -self.config.lambda_u * float(self._last_action ** 2)
    return tracking_error + safety_penalty + control_cost
```

---

## Step 4: Observation Normalization

Neural network agents (PPO, DQN) are sensitive to input scale. If state variables span different orders of magnitude (e.g., glucose ≈ 100, remote insulin ≈ 0.01), normalize online:

```python
from src.utils.math_ops import WelfordNormalizer

class MyEnv:
    def __init__(self, config):
        ...
        self.obs_normalizer = WelfordNormalizer(dim=self.obs_dim)

    def _observe(self) -> np.ndarray:
        raw = np.array([self._x1, self._x2, ...], dtype=np.float32)
        return self.obs_normalizer.normalize(raw)   # updates running mean/var
```

`WelfordNormalizer` is in `src/utils/math_ops.py`.

---

## Step 5: Parameterize from Real Data

| Dynamics type | What to fit | How |
|---------------|-------------|-----|
| Poisson arrival rate λ | Inter-arrival times from logs | MLE: λ̂ = n / total_time |
| OU process (θ, μ, σ) | Price / signal time series | MLE via discretized OU likelihood, or method of moments |
| ODE parameters | State trajectories + known inputs | Nonlinear least squares (`scipy.optimize.curve_fit`) or MCMC |
| Demand distribution | Sales history | Fit Poisson or Negative Binomial; test overdispersion |
| Lead time L | Order-to-receipt records | Empirical CDF; use mean for deterministic, fit distribution for stochastic |

**Validation:** After fitting, run the simulation forward and compare simulated trajectories against held-out historical data. Key metrics: mean absolute error, coverage of 90% prediction interval, qualitative pattern match (seasonality, burst behavior).

---

## Step 6: Write Tests

Add to `tests/test_envs.py`:

```python
class TestMyEnv:
    def setup_method(self):
        self.env = MyEnv(MyConfig())

    def test_reset_returns_correct_shape(self):
        obs = self.env.reset(seed=42)
        assert obs.shape == (EXPECTED_OBS_DIM,)
        assert obs.dtype == np.float32

    def test_step_returns_correct_types(self):
        self.env.reset(seed=0)
        obs, reward, done, info = self.env.step(VALID_ACTION)
        assert isinstance(reward, float)
        assert isinstance(done, bool)

    def test_seeding_reproducibility(self):
        obs1 = self.env.reset(seed=7)
        _, r1, _, _ = self.env.step(VALID_ACTION)
        obs2 = self.env.reset(seed=7)
        _, r2, _, _ = self.env.step(VALID_ACTION)
        np.testing.assert_array_equal(obs1, obs2)
        assert r1 == r2

    def test_reward_bounds(self):
        """Reward should stay within expected range over a full episode."""
        self.env.reset(seed=0)
        rewards = []
        done = False
        while not done:
            _, r, done, _ = self.env.step(self.env.get_legal_actions()[0])
            rewards.append(r)
        assert all(R_MIN <= r <= R_MAX for r in rewards)

    def test_physical_constraints_never_violated(self):
        """State variables must stay within physical bounds."""
        self.env.reset(seed=0)
        for _ in range(200):
            obs, _, done, _ = self.env.step(RANDOM_ACTION)
            assert LOWER_BOUND <= obs[STATE_IDX] <= UPPER_BOUND
            if done:
                self.env.reset()
```

---

## Step 7: Register the Environment

1. Add import to `src/envs/__init__.py`:
```python
from src.envs.my_env import MyEnv
```

2. Add preset to `src/config.py`:
```python
PRESETS["my_env_ppo"] = {
    "env": "my_env",
    "agent": "ppo",
    "env_config": MyConfig(),
    "agent_config": PPOConfig(state_dim=OBS_DIM, action_dim=ACTION_DIM, ...),
    "episodes": 500,
}
```

3. Run the full test suite to confirm no regressions:
```bash
python -m pytest tests/ -v
```
