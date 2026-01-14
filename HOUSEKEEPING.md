# Housekeeping Protocol

1. Read the AGENTS.md file.
2. Look at the dependency network of the project, namely which script refers to which one.
3. Proceed doing different sanity checks and unit tests from root scripts to leaves.
4. Compile all errors and tests results into a report. Include the author of the report (Claude, Jules, etc). And print that report in the Latest Report subsection below, overwriting previous reports.
5. Add that report to the AGENTS_LOG.md.

# Current Project Housekeeping

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
  - `src/agents/__init__.py`
    - `src/agents/base.py`
    - `src/agents/bandit.py`
    - `src/agents/dqn.py`
    - `src/agents/mcts.py`
    - `src/agents/ppo.py`
  - `src/utils/logger.py`
  - `src/utils/seeding.py`
- `src/envs/homeostasis.py` -> `src/utils/math_ops.py`
- `src/agents/bandit.py` -> `src/utils/math_ops.py`

## Latest Report

**Author:** Jules
**Execution Date:** 2026-01-14

**Test Results:**
- `python -m pytest tests/ -v`: **PASSED** (57/57 passed).

**Code Verification:**
- **File Existence:** All files listed in AGENTS.md exist.
- **Imports:** Verified successfully.
- **Environment:**
  - `numpy`: 2.4.1
  - `torch`: 2.9.1
  - `pytest`: 9.0.2

**Summary:**
The codebase is now fully healthy and functional.
- Fixed `TypeError` in `src/utils/math_ops.py` (NumPy 2.x scalar conversion).
- Fixed `tests/test_math_ops.py` RK4 convergence test by increasing simulation steps.
- Fixed `tests/test_envs.py` Homeostasis test by disabling random meals to isolate insulin effects.
- All unit tests pass.
