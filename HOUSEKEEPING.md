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
- `python -m pytest tests/ -v`: **PARTIALLY PASSED** (50 passed, 7 failed).

**Failures:**
1.  **`src.utils.math_ops.sherman_morrison_update` (TypeError)**
    - **Error:** `TypeError: only 0-dimensional arrays can be converted to Python scalars`
    - **Location:** `src/utils/math_ops.py:102`
    - **Affected Tests:**
        - `tests/test_agents.py::TestLinUCBAgent::test_store_and_update`
        - `tests/test_agents.py::TestLinUCBAgent::test_ucb_exploration`
        - `tests/test_agents.py::TestLinUCBAgent::test_reset`
        - `tests/test_math_ops.py::TestShermanMorrison::test_correctness_against_direct_inverse`
        - `tests/test_math_ops.py::TestShermanMorrison::test_multiple_updates`
    - **Analysis:** The expression `float(x.T @ A_inv_x)` fails because the matrix multiplication returns a 1x1 numpy array, which `float()` rejects in newer numpy versions (2.0+).

2.  **`tests/test_envs.py::TestHomeostasisEnv::test_insulin_affects_glucose`**
    - **Error:** `AssertionError: assert 133.37 <= (88.6 + 20)`
    - **Analysis:** The final glucose level after insulin injection is higher than expected. This suggests either the insulin dynamic in `HomeostasisEnv` is not potent enough, the delay is too long, or the test expectation is unrealistic for the current parameters.

3.  **`tests/test_math_ops.py::TestRK4Integration::test_with_control_input`**
    - **Error:** `AssertionError: assert 3.67 < 0.5`
    - **Analysis:** The test expects the system to reach steady state (10.0) within 10 seconds. However, for `dx/dt = -0.1x + 1`, the time constant is 10s. After 10s (1 time constant), it reaches only ~63% of steady state (6.32). The test logic assumes faster convergence or longer simulation time.

**Code Verification:**
- **File Existence:** All files listed in AGENTS.md exist.
- **Imports:** `verify_imports.py` passed successfully (after installing dependencies).
- **Environment:** Installed `numpy` (2.4.1), `torch` (2.9.1), `pytest` (9.0.2).

**Summary:**
The codebase is structurally sound with valid imports. However, there are significant functional issues:
1.  A breaking `TypeError` in core math operations (`sherman_morrison_update`) affects the LinUCB agent.
2.  Numerical integration tests (`test_with_control_input`) have incorrect expectations regarding convergence speed.
3.  The Homeostasis environment logic or its corresponding test needs adjustment.
