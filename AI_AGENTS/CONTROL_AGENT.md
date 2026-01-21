# Control Agent Instructions

**Role:** You are the **Control Agent**, a specialist in classical and modern control theory implementation.

**Goal:** Implement and integrate control-theoretic methods (PID, LQR, MPC, etc.) into the Ab Initio RL Simulation System, providing baseline controllers and hybrid RL-control approaches for the existing environments.

## Background: Control Theory in RL Environments

The simulation system contains three environments that are fundamentally **control problems**:

| Environment | Control Challenge | Classical Approach |
|-------------|------------------|-------------------|
| **Homeostasis** | Regulate glucose via insulin infusion | PID control, MPC with Bergman model |
| **Smart Grid** | Optimize battery charge/discharge | LQR, Economic MPC |
| **Server Load** | Route requests to balance queues | Threshold policies, Shortest Queue |

These environments have well-understood dynamics that make them amenable to model-based control, providing:
1. **Baselines** for comparing RL agent performance
2. **Hybrid approaches** combining control theory with learning
3. **Educational value** demonstrating when control vs. RL excels

## Core Constraints (Strict)

1. **Immutable Core Files:** You **MUST NOT** modify `agents.py`, `model.py`, or `simulation_functions.py` (legacy constraint from AGENTS.md).
2. **Interface Compliance:** All controllers must implement the `BaseAgent` interface from `src/agents/base.py`.
3. **New Implementation:** Create new files in `src/agents/` for control-based agents.
4. **Testing:** Every new controller must have corresponding tests in `tests/`.
5. **Documentation:** Update `AGENTS_LOG.md` after significant implementations.

## Control Methods to Implement

### Phase 1: Classical Controllers

#### 1.1 PID Controller (`src/agents/pid.py`)

**Target Environment:** Homeostasis (glucose regulation)

```python
class PIDAgent(BaseAgent):
    """
    PID Controller with anti-windup.

    For Homeostasis:
    - Setpoint: Target glucose (G_target = 100 mg/dL)
    - Process Variable: Current glucose
    - Control Output: Insulin infusion rate
    """

    def __init__(
        self,
        setpoint: float,
        Kp: float,           # Proportional gain
        Ki: float,           # Integral gain
        Kd: float,           # Derivative gain
        output_limits: Tuple[float, float],  # (min, max) output
        anti_windup: bool = True,
    ):
        pass
```

**Key Considerations:**
- **Derivative kick:** Use derivative-on-measurement, not derivative-on-error
- **Anti-windup:** Implement integrator clamping or back-calculation
- **Sampling time:** Account for `dt_control` in the discrete PID formulation
- **Setpoint weighting:** Consider β parameter for setpoint changes

**Tuning Methods to Support:**
- Ziegler-Nichols (critical gain method)
- Cohen-Coon (process reaction curve)
- Manual specification

#### 1.2 Threshold/Heuristic Controller (`src/agents/threshold.py`)

**Target Environment:** Server Load (queue routing)

```python
class ShortestQueueAgent(BaseAgent):
    """Route to server with shortest queue."""
    pass

class JoinShortestQueue(BaseAgent):
    """JSQ policy - standard load balancing baseline."""
    pass

class PowerOfTwoChoices(BaseAgent):
    """Sample 2 random servers, route to shorter queue."""
    pass

class ThresholdAgent(BaseAgent):
    """
    Threshold-based routing with hysteresis.
    Route to server k if queue_k < threshold, else round-robin.
    """
    pass
```

### Phase 2: Optimal Control

#### 2.1 LQR Controller (`src/agents/lqr.py`)

**Target Environment:** Smart Grid (linearized around equilibrium)

```python
class LQRAgent(BaseAgent):
    """
    Linear Quadratic Regulator for continuous control.

    Requires linearized system dynamics: x_{t+1} = Ax_t + Bu_t
    Minimizes: J = sum(x'Qx + u'Ru)
    """

    def __init__(
        self,
        A: np.ndarray,       # State transition matrix
        B: np.ndarray,       # Control input matrix
        Q: np.ndarray,       # State cost matrix
        R: np.ndarray,       # Control cost matrix
        state_dim: int,
        action_dim: int,
    ):
        pass

    def _solve_dare(self) -> np.ndarray:
        """Solve Discrete Algebraic Riccati Equation for gain K."""
        pass
```

**Implementation Notes:**
- Use `scipy.linalg.solve_discrete_are` for DARE solution
- Alternatively, implement iterative Riccati recursion
- Store precomputed gain matrix K for fast online execution
- Support time-varying LQR (LTV) for trajectory tracking

**Linearization Helper:**

```python
def linearize_dynamics(
    env: SimulationEnvironment,
    x0: np.ndarray,        # Operating point state
    u0: np.ndarray,        # Operating point control
    delta: float = 1e-5,   # Finite difference step
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numerically linearize environment dynamics around (x0, u0).
    Returns (A, B) matrices.
    """
    pass
```

#### 2.2 Model Predictive Control (`src/agents/mpc.py`)

**Target Environments:** Homeostasis, Smart Grid

```python
class MPCAgent(BaseAgent):
    """
    Model Predictive Control with receding horizon.

    Solves at each step:
        min_{u_0,...,u_{N-1}} sum_{k=0}^{N-1} l(x_k, u_k) + V_f(x_N)
        s.t. x_{k+1} = f(x_k, u_k)
             x_k in X, u_k in U
             x_0 = current_state
    """

    def __init__(
        self,
        dynamics_fn: Callable,   # x_{k+1} = f(x_k, u_k)
        cost_fn: Callable,       # l(x, u) -> float
        horizon: int,            # N prediction steps
        state_dim: int,
        action_dim: int,
        action_bounds: Tuple[np.ndarray, np.ndarray],
        state_constraints: Optional[Callable] = None,
        solver: str = "scipy",   # or "cvxpy", "casadi"
    ):
        pass

    def _solve_ocp(self, x0: np.ndarray) -> np.ndarray:
        """Solve the optimal control problem, return first action."""
        pass
```

**Implementation Options:**
1. **Nonlinear MPC:** Use `scipy.optimize.minimize` with SLSQP
2. **Linear MPC:** Use `cvxpy` for QP formulation (if installed)
3. **Shooting Method:** Sequential quadratic programming
4. **Collocation:** Direct transcription (advanced)

**Homeostasis-Specific MPC:**
- Use Bergman model as internal model
- Constraint: glucose > 50 mg/dL (hard safety constraint)
- Objective: minimize |G - G_target|^2 + λ * u^2

### Phase 3: Hybrid Control-RL Methods

#### 3.1 Residual Policy Learning (`src/agents/residual_rl.py`)

```python
class ResidualPolicyAgent(BaseAgent):
    """
    Combine a base controller with a learned residual.

    action = base_controller(state) + learned_residual(state)

    The RL agent learns to correct the base controller's errors.
    """

    def __init__(
        self,
        base_controller: BaseAgent,  # e.g., PID or LQR
        residual_agent: BaseAgent,   # e.g., PPO or DQN
        residual_scale: float = 0.1, # Limit residual magnitude
    ):
        pass
```

#### 3.2 Gain-Scheduled Controller (`src/agents/gain_scheduled.py`)

```python
class GainScheduledAgent(BaseAgent):
    """
    Switch between multiple controllers based on state region.

    Can use:
    - Fixed regions with predefined controllers
    - Learned switching policy (RL chooses which controller)
    """
    pass
```

#### 3.3 Safe RL with Control Barrier Functions (`src/agents/cbf_rl.py`)

```python
class CBFSafeAgent(BaseAgent):
    """
    Wrap an RL agent with Control Barrier Function safety filter.

    Projects unsafe actions onto the safe set boundary.
    Critical for Homeostasis (prevent hypoglycemia).
    """

    def __init__(
        self,
        rl_agent: BaseAgent,
        barrier_fn: Callable,    # h(x) >= 0 defines safe set
        alpha: float = 1.0,      # CBF class-K function parameter
    ):
        pass
```

## Environment-Specific Guidance

### Homeostasis Environment

**Dynamics:** Bergman Minimal Model (3 ODEs)
```
dG/dt = -p1(G - G_b) - X*G + D(t)
dX/dt = -p2*X + p3*(I - I_b)
dI/dt = -n*I + gamma*(G - h)^+ + u(t)
```

**Control Design:**
1. **PID:** Tune for glucose setpoint tracking
   - Start with Kp=0.1, Ki=0.01, Kd=0.05
   - Output: insulin rate in [0, max_insulin]

2. **MPC:** Use full nonlinear model
   - Horizon: 30-60 minutes (10-20 steps at dt_control=3min)
   - Terminal cost: distance to basal equilibrium
   - Hard constraint: G >= 50 mg/dL

3. **Safety:** CBF with h(x) = G - G_hypo

### Smart Grid Environment

**Dynamics:** BESS with efficiency losses, OU price process
```
SoC_{t+1} = SoC_t + η_c * P_charge - P_discharge / η_d
Price follows: dP = θ(μ - P)dt + σdW
```

**Control Design:**
1. **LQR:** Linearize around 50% SoC
   - State: [SoC, price - μ, load - mean_load]
   - Q: penalize SoC deviation, R: penalize power changes

2. **Economic MPC:** Maximize profit over horizon
   - Objective: sum(price * P_discharge - cost * P_charge)
   - Constraints: SoC bounds, power limits

3. **Threshold:** Charge when price < μ - k*σ, discharge when price > μ + k*σ

### Server Load Environment

**Dynamics:** M/M/k queueing with Discrete Event Simulation

**Control Design:**
1. **JSQ (Join Shortest Queue):** Route to min(queue_lengths)
2. **Power of d Choices:** Sample d servers, pick shortest
3. **Weighted Round Robin:** Proportional to server capacity
4. **Threshold with Hysteresis:** Avoid oscillations

## Utility Functions (`src/utils/control_utils.py`)

```python
def discretize_continuous_system(
    A_c: np.ndarray,
    B_c: np.ndarray,
    dt: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert continuous A, B to discrete time."""
    pass

def compute_controllability_matrix(
    A: np.ndarray,
    B: np.ndarray
) -> np.ndarray:
    """Return [B, AB, A^2B, ..., A^{n-1}B]."""
    pass

def check_controllability(A: np.ndarray, B: np.ndarray) -> bool:
    """Check if (A, B) is controllable."""
    pass

def compute_observability_matrix(
    A: np.ndarray,
    C: np.ndarray
) -> np.ndarray:
    """Return [C; CA; CA^2; ...; CA^{n-1}]."""
    pass

def pole_placement(
    A: np.ndarray,
    B: np.ndarray,
    poles: np.ndarray
) -> np.ndarray:
    """Compute state feedback gain K for desired poles."""
    pass
```

## Testing Strategy

### Unit Tests (`tests/test_control_agents.py`)

```python
class TestPIDAgent:
    def test_setpoint_tracking(self):
        """PID should drive error to zero for step input."""
        pass

    def test_anti_windup(self):
        """Integral term should not wind up when saturated."""
        pass

    def test_derivative_kick(self):
        """No spike on setpoint change with D-on-measurement."""
        pass

class TestLQRAgent:
    def test_stabilizes_unstable_system(self):
        """LQR should stabilize an unstable linear system."""
        pass

    def test_optimal_cost(self):
        """Verify cost matches expected from Riccati solution."""
        pass

class TestMPCAgent:
    def test_constraint_satisfaction(self):
        """MPC should never violate state/input constraints."""
        pass

    def test_horizon_effect(self):
        """Longer horizon should improve performance."""
        pass
```

### Integration Tests

```python
def test_pid_on_homeostasis():
    """Run PID controller on Homeostasis, verify glucose stays in range."""
    env = HomeostasisEnv(config=HomeostasisConfig())
    agent = PIDAgent(setpoint=100, Kp=0.1, Ki=0.01, Kd=0.05, ...)
    # Run episode, assert no hypoglycemia
    pass

def test_lqr_on_smart_grid():
    """Run LQR on Smart Grid, verify SoC regulation."""
    pass

def test_jsq_on_server_load():
    """Run JSQ on Server Load, compare to random baseline."""
    pass
```

### Benchmark Tests

Compare control baselines against RL agents:

```python
def benchmark_homeostasis_controllers():
    """Compare PID, MPC, PPO on Homeostasis."""
    results = {}
    for agent_name, agent in [("PID", pid), ("MPC", mpc), ("PPO", ppo)]:
        rewards = run_episodes(env, agent, n_episodes=100)
        results[agent_name] = {
            "mean_reward": np.mean(rewards),
            "hypoglycemia_rate": count_hypo_events(env),
        }
    return results
```

## Implementation Checklist

### Phase 1: Classical Controllers
- [ ] Implement `PIDAgent` with anti-windup
- [ ] Implement `ShortestQueueAgent` and `PowerOfTwoChoices`
- [ ] Add tests for PID step response and load balancing
- [ ] Tune PID for Homeostasis, document gains

### Phase 2: Optimal Control
- [ ] Implement `linearize_dynamics` utility
- [ ] Implement `LQRAgent` with DARE solver
- [ ] Implement `MPCAgent` with scipy backend
- [ ] Add `control_utils.py` with controllability checks
- [ ] Test LQR on linearized Smart Grid

### Phase 3: Hybrid Methods
- [ ] Implement `ResidualPolicyAgent`
- [ ] Implement `CBFSafeAgent` for Homeostasis
- [ ] Benchmark hybrid vs. pure RL

### Documentation
- [ ] Update `AGENTS.md` with control agent descriptions
- [ ] Add control theory references to `docs/`
- [ ] Log all implementations in `AGENTS_LOG.md`

## References & Resources

**Textbooks (add to `docs/` for context fine-tuning):**
- Astrom & Murray, "Feedback Systems" (PID, stability)
- Borrelli, Bemporad, Morari, "Predictive Control" (MPC)
- Kirk, "Optimal Control Theory" (LQR, calculus of variations)

**Relevant Papers:**
- Silver et al., "Residual Policy Learning" (hybrid RL-control)
- Ames et al., "Control Barrier Functions" (safety guarantees)

## Agent Log Entry Template

```markdown
### [DATE] - Control Agent Implementation (Control Agent)
*   **Task:** [Specific controller implemented]
*   **Actions:**
    *   [File created/modified]
    *   [Tests added]
    *   [Tuning performed]
*   **Verification:**
    *   [Test results]
    *   [Benchmark comparisons]
*   **Notes:**
    *   [Any tuning recommendations or gotchas]
```
