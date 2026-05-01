# Housekeeping Protocol
- status: active
- type: workflow
- description: Recurring codebase health check — dependency network verification, unit tests, bug identification, and markdown compliance. Run at the start of any major session.
- label: [agent]
- injection: procedural
- volatility: stable
- scope: project-specific
- last_checked: 2026-05-01
<!-- content -->

This workflow defines the recurring health check for the `control_algorithms` codebase. Run it at the start of any major session to confirm the dependency graph is intact, all tests pass, no regressions or bugs have accumulated, markdowns are schema-compliant, and the codebase is in a known good state before making changes.

The output is a structured report written into the **Latest Report** section below and summarized in `WORKLOG.md`.

## Phase 1 — Review Project Structure

1. Read `README.md` to confirm the current source layout and available components.
2. Cross-check the **Dependency Network** section below against the actual files on disk — verify all listed modules exist and no new files are missing from the map. Update the Dependency Network section if anything has changed.

## Phase 2 — Unit Tests

3. Run the full test suite from the project root:
   ```bash
   python -m pytest tests/ -v
   ```
4. Record the total number of tests, pass/fail counts, and any errors or warnings.
5. For each failing test: identify the root cause (import error, logic bug, missing fixture, etc.) and attempt a fix before proceeding. If the fix is non-trivial, note it in the report and flag for follow-up.

## Phase 3 — Code Verification

6. Trace the dependency tree from root scripts to leaves — confirm all imports resolve correctly and no circular dependencies exist.
7. Spot-check each module for obvious issues: missing return values, unreachable code, hardcoded paths, or uncaught exceptions at module boundaries.
8. Verify environment dependencies are importable:
   ```bash
   python -c "import numpy, torch, scipy; print('OK')"
   ```

## Phase 4 — Markdown Compliance

9. Identify all Markdown files in the repository root and in `docs/` (if any):
   ```bash
   find . -maxdepth 2 -name "*.md" | sort
   ```
10. For each Markdown file that uses the MD_CONVENTIONS schema (i.e., contains a metadata block under its `#` header), verify:
    - `status`, `type`, and `<!-- content -->` separator are present.
    - `type` is one of the eight valid values.
    - No metadata blocks appear on `##` or deeper headers (content/workflow documents).
    - `description` and `scope` fields are present on content documents.
    - Any `label` values are drawn from the registered label set.
11. Note any non-compliant files in the report. Minor issues (missing `scope`, missing `description`) may be fixed in place; structural violations should be flagged for a dedicated fix session.

## Phase 5 — Report

12. Compile all findings into a structured report (see Latest Report format below).
13. Overwrite the **Latest Report** section with the new report.
14. Append a brief summary entry to `WORKLOG.md`.

---

## Dependency Network

**Status: VERIFIED**
The dependency network is mapped and all imports are valid.

**Dependency Tree:**
- `src/main.py`
  - `src/config.py`
  - `src/envs/__init__.py`
    - `src/envs/base.py`
    - `src/envs/server_load.py`
    - `src/envs/smart_grid.py`
    - `src/envs/homeostasis.py`
    - `src/envs/stock_management.py`
  - `src/agents/__init__.py`
    - `src/agents/base.py`
    - `src/agents/bandit.py`
    - `src/agents/dqn.py`
    - `src/agents/mcts.py`
    - `src/agents/ppo.py`
  - `src/utils/logger.py`
  - `src/utils/seeding.py`
- `src/controllers/__init__.py`
  - `src/controllers/base.py` -> `src/agents/base.py`
  - `src/controllers/pid.py`
  - `src/controllers/lqr.py`
  - `src/controllers/mpc.py`
- `src/simulations/__init__.py`
  - `src/simulations/stock_management_sim.py`
    - `src/envs/stock_management.py`
    - `src/agents/ppo.py`
    - `src/controllers/mpc.py`
- `src/envs/homeostasis.py` -> `src/utils/math_ops.py`
- `src/agents/bandit.py` -> `src/utils/math_ops.py`

## Latest Report

**Author:** Claude (Claude Code CLI — Sonnet 4.6)
**Execution Date:** 2026-05-01

**Test Results:**
- `python -m pytest tests/ -v`: **76 passed, 11 skipped, 0 failed** (1.23s)
- Skipped: all 11 are PyTorch-dependent (DQN, PPO, ReplayBuffer, TrajectoryBuffer). Root cause: broken torch install — `libtorch_cpu.dylib` missing. Code handles this gracefully via try/except in `seeding.py` and pytest skip markers. Not a code defect.

**Code Verification:**
- **File Existence:** All 26 source files verified present.
  - `src/`: 26 files (envs: 6, agents: 6, controllers: 5, utils: 4, simulations: 2, root: 3)
  - `tests/`: 5 files (test_agents, test_envs, test_math_ops, test_controllers, __init__)
- **Imports:** All non-torch imports resolve correctly. `src.config` exports `PRESETS` (4 presets: server_load_dqn, smart_grid_linucb, homeostasis_ppo, server_load_mcts). `src.utils.seeding` exports `set_global_seeds`, `create_rng`, `spawn_rngs`.
- **Dependency Network:** Verified — all listed modules present on disk, no structural changes since last map.
- **Environment:**
  - `numpy`: 1.23.5
  - `scipy`: 1.11.4
  - `pytest`: 7.1.2
  - `torch`: BROKEN — `libtorch_cpu.dylib` missing (pre-existing environment issue)

**Markdown Compliance:**
- `HOUSEKEEPING.md`: Compliant ✓ (updated this session: `type` → `workflow`, added `scope`, added preamble, restructured into phases)
- `README.md`: Plain project README — no schema metadata expected ✓
- `WORKLOG.md`: Compliant ✓ (added `scope` field this session)

**Issues Found:**
1. **torch broken (environment):** `libtorch_cpu.dylib` missing from the conda environment. 11 PyTorch-dependent tests are skipped. All affected code (seeding, DQN, PPO) handles absence gracefully — no code changes needed. Fix: reinstall PyTorch in the conda env.

**Summary:**
Codebase is **HEALTHY** (with environment caveat). 76/87 tests pass; 11 skipped due to a broken PyTorch installation (not a code issue). All non-torch modules import and function correctly. Dependency network matches files on disk. No logic bugs found.
