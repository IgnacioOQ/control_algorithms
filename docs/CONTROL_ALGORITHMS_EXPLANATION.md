# Control Algorithms: Explanation and Application Guide
- status: active
- type: explanation
- description: Explains PID, LQR, and MPC control algorithms implemented in this repo — what they do, when to use them vs. RL, and how to apply them to real-world automation domains like logistics and inventory control.
- label: [source-material]
- injection: background
- volatility: stable
- last_checked: 2026-05-01
<!-- content -->

This document is the background reference for the classical and optimal control algorithms implemented in `src/controllers/`. It covers the mathematical intuition, practical tradeoffs, environment-specific application patterns, and guidance for mapping these methods to real-world automation problems.

For the full mathematical treatment of the simulation environments (Bergman model ODEs, BESS dynamics, M/M/k queueing), see [RL_SIMULATION_EXPLANATION.md](RL_SIMULATION_EXPLANATION.md).

For implementation instructions and remaining work (Phase 3 hybrid methods), see [CONTROL_ALGORITHMS_SKILL.md](CONTROL_ALGORITHMS_SKILL.md).

---

## 1. Classical Control vs. Reinforcement Learning

The central question when approaching an automation problem is: **should you use a classical controller or train an RL agent?**

| Dimension | Classical Control (PID/LQR/MPC) | Reinforcement Learning |
|-----------|--------------------------------|------------------------|
| **Model requirement** | Requires known or estimated dynamics | Model-free; learns from interaction |
| **Sample efficiency** | No data needed — runs from first step | Needs thousands to millions of episodes |
| **Interpretability** | Fully interpretable (gain matrices, prediction horizon) | Black-box policy |
| **Optimality** | LQR is globally optimal for linear systems | Locally optimal; depends on exploration |
| **Constraint handling** | MPC handles hard constraints explicitly | Soft constraints via reward shaping |
| **Adaptation** | Fixed (unless gains are re-tuned) | Adapts to distribution shift |
| **Deployment risk** | Predictable behavior; certifiable | Difficult to certify safety |

**Rule of thumb:** Use classical control when the dynamics are known and roughly linear; use RL when the system is poorly modeled, highly nonlinear, or requires learning from sparse signals. Hybrid approaches (residual RL, gain scheduling) cover the middle ground.

---

## 2. PID Control

**File:** [src/controllers/pid.py](../src/controllers/pid.py)  
**Class:** `PIDController`

### What it does

PID is a feedback controller that computes a control signal from three terms:

```
u(t) = Kp * e(t)  +  Ki * ∫e(τ)dτ  +  Kd * de/dt
```

- **Proportional (Kp):** React to current error. High Kp → fast response but overshoot.
- **Integral (Ki):** Eliminate steady-state offset by accumulating error over time. Risk: integrator windup when the actuator saturates.
- **Derivative (Kd):** Anticipate error by reacting to its rate of change. Applied to the measurement (not the error) to avoid derivative kick on setpoint changes.

The implementation uses **anti-windup** (integrator clamping) and **derivative-on-measurement** by default.

### Tuning

Two auto-tuning methods are supported:
- **Ziegler-Nichols:** Drive the system to sustained oscillation; record critical gain K_u and period T_u.
- **Cohen-Coon:** Use the process reaction curve (open-loop step response).

### When to use PID

- Single-input single-output (SISO) setpoint tracking.
- The system has one dominant time constant and minimal nonlinearity.
- Fast deployment is required with minimal modeling effort.
- **Avoid** when: the system has significant time delay, strong nonlinearities, or multiple interacting outputs (MIMO).

### Environment application: Homeostasis

The glucose-insulin loop is a SISO control problem: regulate plasma glucose G toward a target (100 mg/dL) by modulating insulin infusion rate u(t).

```
Setpoint: G_target = 100 mg/dL
Process variable: G(t) (plasma glucose)
Control output: insulin infusion rate u(t) ∈ [0, u_max]
Suggested starting gains: Kp=0.1, Ki=0.01, Kd=0.05
```

Limitation: PID does not see the meal disturbance D(t) coming. MPC with a meal predictor outperforms PID in meal-challenge scenarios.

---

## 3. Linear Quadratic Regulator (LQR)

**File:** [src/controllers/lqr.py](../src/controllers/lqr.py)  
**Classes:** `LQRController`, `FiniteHorizonLQR`  
**Utilities:** `discretize_system`, `check_controllability`, `check_stabilizability`

### What it does

LQR computes a state-feedback gain matrix K that minimizes the infinite-horizon quadratic cost:

```
J = Σ (xᵀQx + uᵀRu)
```

subject to linear dynamics `x_{t+1} = Ax_t + Bu_t`. The optimal gain is:

```
u* = -Kx,  where K = (R + BᵀPB)⁻¹ BᵀPA
```

P is the solution to the **Discrete Algebraic Riccati Equation (DARE)**, solved once offline. Online execution is then a single matrix multiply — O(n²) per step.

**Q** penalizes state deviation (how much you care about staying near the setpoint). **R** penalizes control effort (how much you care about actuator usage). The ratio Q/R is the primary design parameter.

### Prerequisites

LQR requires the pair (A, B) to be **controllable** (or at minimum stabilizable). The `check_controllability` utility verifies this before solving the DARE.

For nonlinear systems, linearize around an operating point (x₀, u₀) using finite differences:
```
A ≈ ∂f/∂x|_(x₀,u₀),   B ≈ ∂f/∂u|_(x₀,u₀)
```
The resulting controller is valid in a neighborhood of (x₀, u₀); it degrades as the system moves far from the linearization point.

### When to use LQR

- The system is linear (or can be linearized around a stable operating point).
- You have a clear cost trade-off between state regulation and control effort.
- You need a provably stable controller with a performance guarantee.
- **Avoid** when: the system has hard actuator constraints (LQR ignores them), or the operating point changes significantly over time.

### Environment application: Smart Grid

The BESS dynamics are approximately linear around 50% SoC:

```
State:   x = [SoC - 0.5,  price - μ_price,  load - μ_load]
Control: u = net_power (positive = charge, negative = discharge)
Q: penalize SoC deviation from 50%
R: penalize large power swings (battery degradation proxy)
```

LQR regulates the battery around the equilibrium SoC while reacting to price/load deviations — but cannot directly maximize profit (that requires MPC with an economic objective).

---

## 4. Model Predictive Control (MPC)

**File:** [src/controllers/mpc.py](../src/controllers/mpc.py)  
**Classes:** `MPCController`, `LinearMPC`

### What it does

MPC solves a finite-horizon optimal control problem at every step, applies only the first action, then re-solves (receding horizon):

```
min_{u_0,...,u_{N-1}}  Σ_{k=0}^{N-1} l(x_k, u_k) + V_f(x_N)
subject to:
    x_{k+1} = f(x_k, u_k)       (dynamics model)
    x_k ∈ X, u_k ∈ U             (state and input constraints)
    x_0 = current state
```

The implementation uses `scipy.optimize.minimize` with SLSQP for nonlinear problems, and a dedicated QP formulation (`LinearMPC`) for linear systems. **Warm-starting** from the previous solution is used to reduce computation time.

### Key design parameters

| Parameter | Effect |
|-----------|--------|
| Horizon N | Longer = better performance, higher compute. Typical: 10–30 steps. |
| Stage cost l(x,u) | Shape to encode the objective (tracking, economic, safety) |
| Terminal cost V_f(x_N) | Ensures stability beyond the horizon |
| Constraint sets X, U | Hard constraint satisfaction — MPC's key advantage over PID/LQR |

### When to use MPC

- The system has hard constraints that must never be violated (e.g., glucose > 50 mg/dL, SoC ∈ [0.1, 0.9]).
- The objective is economic (maximize profit, minimize cost) rather than purely regulatory.
- You have a reasonably accurate model and can afford online computation.
- **Avoid** when: computation budget is tight (each step solves an optimization), or the model is highly inaccurate.

### Environment applications

**Homeostasis:** Nonlinear MPC using the full Bergman model as the internal model.
```
Objective: min |G - G_target|² + λ·u²
Constraint: G_k ≥ 50 mg/dL (hard hypoglycemia prevention)
Horizon: 10–20 steps at dt=3 min (30–60 minutes of prediction)
```
This directly encodes the asymmetric safety requirement that PID cannot.

**Smart Grid:** Economic MPC maximizing profit over the prediction horizon.
```
Objective: max Σ price_k · P_discharge_k - cost_k · P_charge_k
Constraints: SoC ∈ [0.1, 0.9], P ∈ [-P_max, P_max]
Horizon: 24 steps (24-hour lookahead with price forecast)
```

**Stock Management:** MPC on inventory dynamics (orders as control inputs).
```
Objective: min holding_cost + spoilage_cost + stockout_cost
Constraints: inventory ≥ 0, order ≤ supplier_capacity
Horizon: 7–14 days
```

---

## 5. Algorithm Selection Guide

### By problem structure

| Problem type | Recommended algorithm |
|-------------|----------------------|
| Single variable setpoint tracking, well-understood dynamics | PID |
| Multi-variable regulation around a fixed operating point | LQR |
| Constrained optimization, economic objective, or safety-critical | MPC |
| Poorly modeled, highly nonlinear, or requires adaptation | RL (PPO/DQN) |
| Unknown dynamics + hard safety constraints | CBF-safe RL (Phase 3) |
| Model available but imperfect | Residual RL on top of PID/LQR (Phase 3) |

### By environment

| Environment | Best classical approach | Why |
|-------------|------------------------|-----|
| Homeostasis (glucose) | MPC | Hard safety constraint; nonlinear model available |
| Smart Grid (BESS) | Economic MPC | Economic objective; price forecast available |
| Server Load | Join Shortest Queue (JSQ) | Near-optimal heuristic; no model needed |
| Stock Management | MPC | Multi-item constraints; demand forecasts available |

---

## 6. Real-World Domain Mapping

These algorithms generalize directly to common automation problems. When building a strategy agent for a new domain, use this table to select a starting point.

### Inventory and logistics

| Problem | State | Control | Algorithm |
|---------|-------|---------|-----------|
| Reorder point optimization (single SKU) | current stock, demand rate | order quantity | PID (track target stock level) |
| Multi-SKU inventory with shelf life | stock levels, age profiles | order quantities per SKU | MPC (spoilage constraints, demand forecasts) |
| Warehouse routing / load balancing | queue lengths per station | job assignment | JSQ / Power-of-2-Choices |
| Fleet dispatch | vehicle positions, demand | dispatch decisions | MPC or DQN |

### Energy and utilities

| Problem | State | Control | Algorithm |
|---------|-------|---------|-----------|
| HVAC setpoint control | room temperature | heating/cooling power | PID |
| Building energy storage arbitrage | SoC, price forecast | charge/discharge | Economic MPC |
| Microgrid frequency regulation | frequency deviation | generation set-point | LQR |
| Demand response with renewables | SoC, generation, price | load scheduling | MPC |

### Industrial processes

| Problem | State | Control | Algorithm |
|---------|-------|---------|-----------|
| Chemical reactor temperature | temperature, concentration | coolant flow | PID (if linear) or MPC (if nonlinear) |
| Production line throughput | WIP buffers | machine speeds | LQR (linearized) |
| Drug infusion (insulin pump, anesthesia) | blood concentration | infusion rate | MPC with safety constraints |

### Decision heuristics

For **queueing and routing** problems where the system is a service network (servers, warehouses, call centers), threshold policies and JSQ-type heuristics often match or beat RL:
- **Join Shortest Queue (JSQ):** Route to the server/node with the minimum current queue. Optimal for symmetric homogeneous queues.
- **Power of d Choices:** Sample d nodes randomly, route to the shortest. Achieves O(log log n) max queue vs. O(log n) for random routing — a major improvement with d=2.
- **Threshold with hysteresis:** Activate/deactivate servers based on queue length thresholds. Prevents oscillation from rapid on/off cycling.

---

## 7. Hybrid Control-RL Methods (Phase 3)

When neither pure classical control nor pure RL is sufficient, hybrid approaches combine the strengths of both:

### Residual Policy Learning

```
u_total = u_base(state) + u_residual(state)
```

A PID or LQR handles the nominal task; an RL agent learns a small correction for the cases the base controller handles poorly. The RL agent's action space is limited (e.g., ±10% of nominal), making training stable and safe. Best when: a good base controller exists but performance is capped by model mismatch.

### Control Barrier Functions (CBF) + RL

A safety filter projects any unsafe RL action onto the boundary of the safe set:

```
h(x) ≥ 0  defines the safe set  (e.g., h(x) = G - G_hypo for glucose)
```

The RL agent can explore freely; the CBF layer guarantees constraint satisfaction. Best when: safety constraints are hard and well-defined, but the optimal behavior within the safe set is complex.

### Gain-Scheduled Control

Multiple controllers are designed for different operating regions; a supervisor (fixed rule or learned RL policy) selects which controller is active. Best when: the system has distinct operating modes (e.g., battery nearly full vs. nearly empty) where a single linear controller performs poorly across all modes.

---

## 8. Implementation Status in This Repository

| Algorithm | File | Status |
|-----------|------|--------|
| PID (anti-windup, ZN/CC tuning) | `src/controllers/pid.py` | ✅ Implemented, 30 tests passing |
| LQR (DARE, FiniteHorizon, controllability) | `src/controllers/lqr.py` | ✅ Implemented, tested |
| MPC (nonlinear SLSQP, LinearMPC QP) | `src/controllers/mpc.py` | ✅ Implemented, tested |
| JSQ / Threshold controllers | `src/controllers/threshold.py` | ⏳ Not yet implemented |
| ResidualPolicyAgent | `src/agents/residual_rl.py` | ⏳ Not yet implemented |
| CBFSafeAgent | `src/agents/cbf_rl.py` | ⏳ Not yet implemented |
| GainScheduledAgent | `src/agents/gain_scheduled.py` | ⏳ Not yet implemented |

For implementation instructions and class stubs, see [CONTROL_ALGORITHMS_SKILL.md](CONTROL_ALGORITHMS_SKILL.md).

---

## 9. References

- Åström & Murray, *Feedback Systems: An Introduction for Scientists and Engineers* (PID, stability, frequency domain)
- Borrelli, Bemporad & Morari, *Predictive Control for Linear and Hybrid Systems* (MPC theory)
- Kirk, *Optimal Control Theory: An Introduction* (LQR, calculus of variations)
- Silver et al., "Residual Policy Learning", 2019 (hybrid RL-control)
- Ames et al., "Control Barrier Functions: Theory and Applications", 2019 (CBF safety)
- Mitzenmacher, "The Power of Two Choices in Randomized Load Balancing", 2001 (d-choices heuristic)
