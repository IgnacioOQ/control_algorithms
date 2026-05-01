# Repository Markdown Cleanup Plan
- status: done
- type: plan
- description: Multi-session plan to bring all repository markdown files into MD_CONVENTIONS compliance and fix stale content.
- label: [agent, planning]
- injection: informational
- volatility: evolving
- last_checked: 2026-05-01
<!-- content -->

This plan covers two layers of work for every `.md` file in the repository:

1. **MD_CONVENTIONS compliance** — every file is missing the required metadata block and `<!-- content -->` separator (files were written before the conventions existed).
2. **Content staleness** — several files have outdated information (missing modules, wrong file paths, unchecked completed tasks).

Execute tasks in order. After completing all file edits, run the post-cleanup verification step.

**Open decisions — RESOLVED 2026-05-01:**
- [x] Delete `Guideline_Project.md` → **YES — delete it.** The clean version `RL_SIMULATION_EXPLANATION.md` is the canonical source.
- [x] Rename `Reinforcement Learning Project Guideline.md` → **YES — rename to `RL_SIMULATION_EXPLANATION.md`.**
- [x] Keep `LINEARIZE_AGENT.md` and `MC_AGENT.md` → **YES — keep with legacy note.** These files mirror KB entries `content/how-to/LINEARIZE_SKILL.md` and `content/how-to/MC_SKILL.md` (last_checked: 2026-03-14). The project files are the editable local copies; the KB holds the versioned canonical copies. Add a cross-reference in each file.

---

## AGENTS.md
- status: done
- type: task
<!-- content -->

**File path:** `AGENTS.md`

**MD_CONVENTIONS schema to add** (insert after the `# AGENTS.md` title line):
```
- status: active
- type: reference
- description: Rules, workflow, constraints, and full architecture overview for AI agents working on the Ab Initio RL Simulation System (environments, agents, controllers, utilities).
- label: [agent, normative]
- injection: directive
- volatility: evolving
- last_checked: <date of edit>
<!-- content -->
```

**Content changes:**

1. **Environments table** (under `#### 1. Environments`): Add a 4th row for **Stock Management**:
   ```
   | **Stock Management** (`stock_management.py`) | Multi-item inventory, FIFO spoilage, Poisson demand | Inventory levels, avg ages, demand estimates, storage utilization | Continuous (order quantities) |
   ```

2. **Add Section 4: Simulations** (currently jumps from Section 3 Controllers to "Section 5" Utilities — the 4 was never added):
   ```
   #### 4. Simulations (`src/simulations/`)
   | Simulation | File | Description |
   |------------|------|-------------|
   | **Stock Management Comparison** | `stock_management_sim.py` | Head-to-head MPC planner vs PPO agent on inventory management |
   ```

3. **Rename "Section 5" → "Section 5"** — already correct number, just ensure it says 5 not the old erroneous label.

4. **Directory structure block**: Add `stock_management.py` under `src/envs/` and add a `src/simulations/` subtree:
   ```
   │   ├── envs/
   │   │   ├── stock_management.py     ← ADD
   ...
   ├── simulations/                    ← ADD ENTIRE SUBTREE
   │   ├── __init__.py
   │   └── stock_management_sim.py
   ```

5. **File dependencies block**: Add `src/simulations/stock_management_sim.py` → `src/envs/stock_management.py`, `src/agents/ppo.py`, `src/controllers/mpc.py`.

---

## AGENTS_LOG.md
- status: done
- type: task
<!-- content -->

**File path:** `AGENTS_LOG.md`

**MD_CONVENTIONS schema to add** (insert after the `# AGENTS_LOG` title line):
```
- status: active
- type: log
- description: Chronological record of all agent interventions, implementations, and housekeeping runs for the control_algorithms project.
- label: [agent]
- injection: informational
- volatility: evolving
- last_checked: <date of edit>
<!-- content -->
```

**Content changes:** None. Log is current through 2026-01-23 (stock management implementation). Individual log entries do NOT get per-node metadata — they are prose records, not tracked tasks.

---

## HOUSEKEEPING.md
- status: done
- type: task
<!-- content -->

**File path:** `HOUSEKEEPING.md`

**MD_CONVENTIONS schema to add** (insert after the `# Housekeeping Protocol` title line):
```
- status: active
- type: how-to
- description: Housekeeping protocol for verifying codebase health: dependency network check, unit tests, and latest report. Run this before any major session.
- label: [agent]
- injection: procedural
- volatility: stable
- last_checked: <date of edit>
<!-- content -->
```

**Content changes:**

1. **Dependency tree** — add `stock_management.py` and `simulations/`:
   ```
   - `src/envs/stock_management.py`          ← ADD (standalone, no cross-deps beyond base)
   - `src/simulations/__init__.py`            ← ADD
   - `src/simulations/stock_management_sim.py`← ADD
     - `src/envs/stock_management.py`
     - `src/agents/ppo.py`
     - `src/controllers/mpc.py`
   ```

2. **Latest Report section**: Mark as stale with a note — it reflects the codebase as of 2026-01-21, before `stock_management.py` and `src/simulations/` were added. Update file counts (currently says 23 source files; actual count is higher after those additions).

---

## docs/CONTROL_AGENT.md
- status: done
- type: task
<!-- content -->

**File path:** `docs/CONTROL_AGENT.md`

**MD_CONVENTIONS schema to add** (insert after the `# Control Agent Instructions` title line):
```
- status: active
- type: how-to
- description: Instructions for the Control Agent: implementing and integrating PID, LQR, and MPC controllers as classical baselines for the RL simulation environments.
- label: [agent, skill]
- injection: procedural
- volatility: evolving
- last_checked: <date of edit>
<!-- content -->
```

**Content changes:**

1. **File path correction throughout**: The original spec placed controllers in `src/agents/` (e.g., `src/agents/pid.py`). The actual implementation lives in `src/controllers/`. Update every file path reference in the document:
   - `src/agents/pid.py` → `src/controllers/pid.py`
   - `src/agents/lqr.py` → `src/controllers/lqr.py`
   - `src/agents/mpc.py` → `src/controllers/mpc.py`
   - `tests/test_control_agents.py` → `tests/test_controllers.py`

2. **Phase 1 checklist** — mark all as done:
   ```
   - [x] Implement PIDController with anti-windup (src/controllers/pid.py)
   - [x] Implement LQRController with DARE solver (src/controllers/lqr.py)
   - [x] Implement MPCController with scipy backend (src/controllers/mpc.py)
   - [x] Add tests/test_controllers.py (30 unit tests, all passing)
   ```

3. **Phase 2 checklist** — mark all as done (LQR and MPC were implemented as part of the same pass):
   ```
   - [x] Implement linearize_dynamics utility (in src/controllers/lqr.py)
   - [x] Implement LQRController with DARE solver
   - [x] Implement MPCController with scipy backend
   - [x] Add controllability checks (check_controllability, discretize_system in lqr.py)
   ```

4. **Phase 3 checklist** — these are NOT yet implemented; keep as `[ ]`. Add a note: "Phase 3 (ResidualPolicyAgent, CBFSafeAgent, GainScheduledAgent) is not yet implemented."

5. **Section 1.2 Threshold/Heuristic Controller**: These were not implemented (only PID, LQR, MPC). Add a note: "Not yet implemented — threshold/queue-routing heuristics remain a future task."

---

## docs/LINEARIZE_AGENT.md
- status: done
- type: task
<!-- content -->

**File path:** `docs/LINEARIZE_AGENT.md`

**KB mirror:** `content/how-to/LINEARIZE_SKILL.md` (last_checked: 2026-03-14). The KB holds the versioned canonical copy; this project file is the local working copy. Keep both in sync when making content changes.

**MD_CONVENTIONS schema to add** (insert after the `# Linearize Agent Instructions` title line):
```
- status: active
- type: how-to
- description: Linearize Agent instructions for vectorizing the network epistemology simulation by replacing Python agent loops with efficient NumPy matrix operations.
- label: [agent, skill]
- injection: procedural
- volatility: stable
- last_checked: 2026-05-01
<!-- content -->
```

**Content changes:**

Add a **cross-reference block at the top** (immediately after `<!-- content -->`):

```
> **Network Epistemology phase — legacy document.** This agent instruction was written for the Network Epistemology project. It references files (`model.py`, `agents.py`, `simulation_functions.py`, `VectorizedModel`) that do not exist in the current RL Simulation System codebase. Canonical KB copy: `content/how-to/LINEARIZE_SKILL.md`.
```

All checklist items are already `[x]` — no further content changes needed.

---

## docs/MC_AGENT.md
- status: done
- type: task
<!-- content -->

**File path:** `docs/MC_AGENT.md`

**KB mirror:** `content/how-to/MC_SKILL.md` (last_checked: 2026-03-14). The KB holds the versioned canonical copy; this project file is the local working copy. Keep both in sync when making content changes.

**MD_CONVENTIONS schema to add** (insert after the `# Markov Chain Analysis Agent Instructions` title line):
```
- status: active
- type: how-to
- description: Markov Chain Agent instructions for tracking stochastic properties (absorbing states, convergence, information flow) of the network epistemology simulation.
- label: [agent, skill]
- injection: procedural
- volatility: evolving
- last_checked: 2026-05-01
<!-- content -->
```

**Content changes:**

Add a **cross-reference block at the top** (immediately after `<!-- content -->`):

```
> **Network Epistemology phase — legacy document.** This agent instruction was written for the Network Epistemology project. It references classes (`VectorizedModel`, `net_epistemology/`, `credences`, `alphas_betas`) that do not exist in the current RL Simulation System codebase. Canonical KB copy: `content/how-to/MC_SKILL.md`.
```

---

## Guideline_Project.md
- status: done
- type: task
<!-- content -->

**File path:** `Guideline_Project.md`

**Decision: DELETE (Option A).** User confirmed 2026-05-01. The clean canonical version is `RL_SIMULATION_EXPLANATION.md` (renamed from `Reinforcement Learning Project Guideline.md`).

---

## RL_SIMULATION_EXPLANATION.md
- status: done
- type: task
<!-- content -->

**File path:** `RL_SIMULATION_EXPLANATION.md` (renamed from `Reinforcement Learning Project Guideline.md` — user confirmed 2026-05-01).

**MD_CONVENTIONS schema to add** (insert after the title `#` line):
```
- status: active
- type: explanation
- description: Source material explaining the full mathematical architecture, environments (Server Load, Smart Grid, Homeostasis), and agent designs (LinUCB, DQN, MCTS, PPO) of the ab initio RL simulation system.
- label: [source-material]
- injection: background
- volatility: stable
- last_checked: 2026-05-01
<!-- content -->
```

**Content changes:** None — document content is clean and complete. The rename itself is the only structural change.

---

## Project_Creation_Walkthrough.md
- status: done
- type: task
<!-- content -->

**File path:** `Project_Creation_Walkthrough.md`

**MD_CONVENTIONS schema to add** (insert after the title `#` line):
```
- status: active
- type: log
- description: Implementation log of the initial project build: what was created, directory structure, key components, and how to run the system.
- label: [agent]
- injection: informational
- volatility: stable
- last_checked: <date of edit>
<!-- content -->
```

**Content changes:**

Fix all absolute `file://` links — they reference a machine-specific path (`/Users/ignacio/Documents/VS Code/GitHub Repositories/control_algorithms/`) that is non-portable. Replace with relative paths:

| Old (absolute) | New (relative) |
|---|---|
| `file:///Users/ignacio/.../src/envs/server_load.py` | `src/envs/server_load.py` |
| `file:///Users/ignacio/.../src/envs/smart_grid.py` | `src/envs/smart_grid.py` |
| `file:///Users/ignacio/.../src/envs/homeostasis.py` | `src/envs/homeostasis.py` |
| `file:///Users/ignacio/.../src/agents/bandit.py` | `src/agents/bandit.py` |
| `file:///Users/ignacio/.../src/agents/dqn.py` | `src/agents/dqn.py` |
| `file:///Users/ignacio/.../src/agents/mcts.py` | `src/agents/mcts.py` |
| `file:///Users/ignacio/.../src/agents/ppo.py` | `src/agents/ppo.py` |
| `file:///Users/ignacio/.../AGENTS.md` | `AGENTS.md` |

---

## Create WORKLOG.md
- status: done
- type: task
<!-- content -->

**File path:** `WORKLOG.md` (new file, root of repository)

The coding workflow (`CODING_AGENT_MAIN_WORKFLOW.md`) requires a `WORKLOG.md` at the working repository root. It does not currently exist.

Create with this initial structure and first entry:

```markdown
# WORKLOG
- status: active
- type: log
- description: Session-by-session log of significant work performed in this repository.
- label: [agent]
- injection: informational
- volatility: evolving
- last_checked: 2026-05-01
<!-- content -->

## 2026-05-01 — Repository markdown cleanup and MD_CONVENTIONS compliance
- **Task:** Audit all `.md` files for MD_CONVENTIONS compliance and content staleness. Write a multi-session cleanup plan.
- **Outcome:** Created `REPOSITORY_MD_CLEANUP.md` with a full per-file plan. No file edits made yet — pending user decisions on three open questions (duplicate file, filename with spaces, legacy agent files).
- **Key decisions:** Used `injection: excluded` for LINEARIZE_AGENT.md and MC_AGENT.md to prevent them from being injected into agent context (they reference a prior codebase).
- **KB changes:** None.
- **Follow-up:** Execute tasks in REPOSITORY_MD_CLEANUP.md across subsequent sessions. Resolve three open decisions first.
```

---

## Post-Cleanup Verification
- status: done
- type: task
- blocked_by: [all prior tasks]
<!-- content -->

After all file edits are complete:

1. **Run full test suite** — confirm no regressions: `python -m pytest tests/ -v`
2. **Update HOUSEKEEPING.md** Latest Report section with current date, test results, and accurate file counts.
3. **Update AGENTS_LOG.md** with a new entry recording the markdown cleanup session.
4. **Update this file** (`REPOSITORY_MD_CLEANUP.md`) — set root `status: done` and each task to `status: done`.
5. **Record KB performance** — call `knowledge_base_record_performance` with a session summary.
