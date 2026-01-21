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

**Author:** Claude (Claude Code CLI - Opus 4.5)
**Execution Date:** 2026-01-21

**Test Results:**
- `python -m pytest tests/ -v`: **PASSED** (57/57 passed in 5.33s)

**Code Verification:**
- **File Existence:** All 18 source files verified present (17 in src/ + 1 tests/__init__.py).
- **Imports:** All dependencies properly structured.
- **Dependency Network:** Verified - all imports resolve correctly.
  - `src/main.py` → `config.py`, `envs/`, `agents/`, `utils/`
  - `src/envs/homeostasis.py` → `utils/math_ops.py`
  - `src/agents/bandit.py` → `utils/math_ops.py`
- **Environment:**
  - `numpy`: 1.23.5
  - `torch`: 2.2.2
  - `pytest`: 7.1.2

**Summary:**
Codebase is **HEALTHY**. All 57 unit tests pass. Dependency network verified.
