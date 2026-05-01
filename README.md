# Ab Initio RL Simulation System

A from-scratch reinforcement learning ecosystem implementing simulation environments, RL agents, and classical control baselines — built without high-level RL libraries (no Gymnasium, Stable Baselines). Every state transition, gradient update, and stochastic process is explicitly defined for maximum transparency.

**Philosophy:** Explicit State-Space Orchestration — every mathematical bound, transition dynamic, and reward manifold is explicitly coded.

**Tech Stack:** Python · NumPy · PyTorch · SciPy

---

## Quick Start

```bash
# Install dependencies
pip install numpy torch scipy pytest

# Run all tests
python -m pytest tests/ -v

# Train an agent
python -m src.main --env server_load --agent dqn --episodes 100
python -m src.main --env homeostasis --agent ppo --episodes 1000

# Available presets
python -m src.main --preset server_load_dqn
python -m src.main --preset smart_grid_linucb
python -m src.main --preset homeostasis_ppo
python -m src.main --preset server_load_mcts
```

---

## Environments

| Environment | File | Dynamics | Action Space |
|-------------|------|----------|--------------|
| **Server Load** | `src/envs/server_load.py` | M/M/k queueing, Discrete Event Simulation | Discrete (route to server k) |
| **Smart Grid** | `src/envs/smart_grid.py` | BESS with efficiency losses, OU price process | Continuous (charge/discharge power) |
| **Homeostasis** | `src/envs/homeostasis.py` | Bergman minimal model (3 ODEs), RK4 integration | Continuous (insulin infusion rate) |
| **Stock Management** | `src/envs/stock_management.py` | Multi-item inventory, FIFO spoilage, Poisson demand | Continuous (order quantities) |

## RL Agents

| Agent | File | Algorithm | Action Space |
|-------|------|-----------|--------------|
| **LinUCB** | `src/agents/bandit.py` | Contextual Bandit, Sherman-Morrison O(d²) updates | Discrete |
| **DQN** | `src/agents/dqn.py` | Deep Q-Network, Replay Buffer, Double DQN | Discrete |
| **MCTS** | `src/agents/mcts.py` | Monte Carlo Tree Search, PUCT selection | Discrete |
| **PPO** | `src/agents/ppo.py` | Proximal Policy Optimization, Actor-Critic, GAE | Continuous |

## Controllers (Classical Baselines)

All controllers implement the `BaseAgent` interface for direct comparison with RL agents.

| Controller | File | Algorithm |
|------------|------|-----------|
| **PID** | `src/controllers/pid.py` | Proportional-Integral-Derivative, anti-windup, ZN/CC tuning |
| **LQR** | `src/controllers/lqr.py` | Linear Quadratic Regulator, pure-NumPy DARE solver |
| **MPC** | `src/controllers/mpc.py` | Model Predictive Control, scipy SLSQP, warm-starting |

## Simulations

| Simulation | File | Description |
|------------|------|-------------|
| **Stock Management Comparison** | `src/simulations/stock_management_sim.py` | Head-to-head MPC vs. PPO on inventory management |

---

## Project Structure

```
control_algorithms/
├── src/
│   ├── envs/               # Simulation environments
│   │   ├── base.py         # SimulationEnvironment protocol
│   │   ├── server_load.py
│   │   ├── smart_grid.py
│   │   ├── homeostasis.py
│   │   └── stock_management.py
│   ├── agents/             # RL agents
│   │   ├── base.py         # BaseAgent interface
│   │   ├── bandit.py       # LinUCB
│   │   ├── dqn.py
│   │   ├── mcts.py
│   │   └── ppo.py
│   ├── controllers/        # Classical/optimal control
│   │   ├── base.py         # BaseController (extends BaseAgent)
│   │   ├── pid.py
│   │   ├── lqr.py
│   │   └── mpc.py
│   ├── simulations/        # Comparison frameworks
│   │   └── stock_management_sim.py
│   ├── utils/
│   │   ├── math_ops.py     # RK4, Sherman-Morrison, Welford normalizer
│   │   ├── seeding.py      # Reproducibility utilities
│   │   └── logger.py       # CSV/TensorBoard logging
│   ├── config.py           # Hyperparameter presets
│   └── main.py             # Training orchestrator + CLI
├── tests/
│   ├── test_math_ops.py
│   ├── test_envs.py
│   ├── test_agents.py
│   └── test_controllers.py
├── docs/                   # Documentation and agent instruction files
│   ├── CONTROL_ALGORITHMS_EXPLANATION.md  # PID/LQR/MPC background + domain guide
│   ├── CONTROL_ALGORITHMS_SKILL.md        # Controller implementation spec
│   ├── RL_SIMULATIONS_REF.md              # Simulation pattern catalog by domain
│   ├── RL_SIMULATIONS_SKILL.md            # Environment implementation cookbook
│   ├── Project_Creation_Walkthrough.md
│   └── REPOSITORY_MD_CLEANUP.md
├── notebooks/
├── logs/
├── HOUSEKEEPING.md         # Codebase health protocol
└── WORKLOG.md              # Session log
```

---

## Tests

```bash
python -m pytest tests/ -v
```

| Suite | File | Covers |
|-------|------|--------|
| Math Operations | `tests/test_math_ops.py` | RK4, Sherman-Morrison, Welford normalizer |
| Environments | `tests/test_envs.py` | reset/step contracts, state shapes, reward bounds, seeding |
| Agents | `tests/test_agents.py` | Interface compliance, action selection, buffer ops, updates |
| Controllers | `tests/test_controllers.py` | PID response, LQR stability, MPC constraints, DARE solver |

---

## For AI Agents

Read `HOUSEKEEPING.md` before starting any session — it contains the dependency network, the housekeeping protocol, and the latest codebase health report. Detailed implementation instructions for specific subsystems live in `docs/`:

| Document | Use when |
|----------|----------|
| `docs/RL_SIMULATIONS_REF.md` | Mapping a new optimization problem to a simulation pattern |
| `docs/RL_SIMULATIONS_SKILL.md` | Implementing a new environment from scratch |
| `docs/CONTROL_ALGORITHMS_EXPLANATION.md` | Deciding which controller (PID/LQR/MPC) fits a problem |
| `docs/CONTROL_ALGORITHMS_SKILL.md` | Implementing remaining controllers (Phase 1.2, Phase 3) |
