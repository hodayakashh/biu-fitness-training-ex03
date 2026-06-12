# TODO — Exercise 03: Fitness RL
**Version:** 1.00  
**Last Updated:** 2026-06-12

Legend: `[ ]` pending · `[~]` in progress · `[x]` done

---

## Phase 0 — Documentation (MUST complete before any code)

| # | Task | Priority | DoD |
|---|------|----------|-----|
| 0.1 | Write `docs/PRD.md` | 🔴 Critical | File exists, covers all 6 parts |
| 0.2 | Write `docs/PLAN.md` | 🔴 Critical | State/action/reward defined, module map complete |
| 0.3 | Write `docs/TODO.md` | 🔴 Critical | This file |
| 0.4 | Write `docs/PRD_LSTM.md` | 🔴 Critical | I/O, architecture, training protocol defined |
| 0.5 | Write `docs/PRD_REINFORCE.md` | 🔴 Critical | Policy formulation, update rule, episode structure |
| 0.6 | Write `docs/PRD_A2C.md` | 🔴 Critical | Actor/Critic roles, TD-error, comparison plan |
| 0.7 | **Get approval** from user on all docs | 🔴 Critical | User confirms → Phase 1 unlocked |

**Status:** `[x]` done (docs approved, downstream phases implemented)

---

## Phase 1 — Project Skeleton

| # | Task | Priority | DoD |
|---|------|----------|-----|
| 1.1 | Create directory tree | 🔴 Critical | All dirs exist |
| 1.2 | Create `pyproject.toml` (uv) | 🔴 Critical | `uv sync` succeeds |
| 1.3 | Create `config/setup.json` v1.00 | 🔴 Critical | All hyperparams present |
| 1.4 | Create `config/rate_limits.json` v1.00 | 🟡 High | Gatekeeper can load it |
| 1.5 | Create `.env-example` | 🟡 High | KAGGLE_* keys present as dummies |
| 1.6 | Create `.gitignore` | 🟡 High | `.env`, `data/`, `__pycache__` excluded |
| 1.7 | Create `src/fitness_rl/__init__.py` skeleton | 🔴 Critical | Exports SDK, defines `__version__` |
| 1.8 | Create `tests/conftest.py` | 🟡 High | Shared fixtures for all tests |

**Status:** `[x]` done

---

## Phase 2 — Data Pipeline (Part B)

| # | Task | Priority | DoD |
|---|------|----------|-----|
| 2.1 | Implement `DataService.load_raw()` | 🔴 Critical | Returns DataFrame from CSV |
| 2.2 | Implement `DataService.compute_daily_summaries()` | 🔴 Critical | Columns: total_volume, muscle_dist, duration, day_in_cycle |
| 2.3 | Implement `DataService.cluster_actions()` | 🔴 Critical | 6 clusters, labels added to daily df |
| 2.4 | Implement `DataService.normalize_states()` | 🟡 High | Min-max scaler fit on train split |
| 2.5 | Implement `DataService.build_windows()` | 🔴 Critical | Sliding windows of seq_len=7 |
| 2.6 | Write unit tests for DataService | 🔴 Critical | ≥ 85% branch coverage |

**Status:** `[x]` done (all DataService sub-modules implemented + tested)

---

## Phase 3 — LSTM Transition Model (Part C)

| # | Task | Priority | DoD |
|---|------|----------|-----|
| 3.1 | Implement `LSTMModel` (forward pass) | 🔴 Critical | Output shape = (batch, state_dim) |
| 3.2 | Implement `LSTMTrainer.train()` | 🔴 Critical | Train/val loss logged per epoch |
| 3.3 | Implement `LSTMTrainer.predict()` | 🔴 Critical | Predicts next state given window |
| 3.4 | Save model weights to `results/lstm_weights.pt` | 🟡 High | File exists after training |
| 3.5 | Plot train/val loss curves | 🔴 Critical | Figure saved to `results/plots/lstm_loss.png` |
| 3.6 | Write unit tests for LSTMModel | 🔴 Critical | Shape tests, deterministic output |

**Status:** `[x]` done

---

## Phase 4 — RL Environment

| # | Task | Priority | DoD |
|---|------|----------|-----|
| 4.1 | Implement `RLEnvironment.reset()` | 🔴 Critical | Returns initial state s_0 |
| 4.2 | Implement `RLEnvironment.step(action)` | 🔴 Critical | Returns (s_next, reward, done) |
| 4.3 | Implement `RLEnvironment.compute_reward()` | 🔴 Critical | Formula matches PRD |
| 4.4 | Write unit tests for RLEnvironment | 🔴 Critical | Reward bounds, episode length |

**Status:** `[x]` done

---

## Phase 5 — REINFORCE (Part D)

| # | Task | Priority | DoD |
|---|------|----------|-----|
| 5.1 | Implement `PolicyNetwork` (actor only) | 🔴 Critical | Output is valid prob distribution |
| 5.2 | Implement `REINFORCETrainer.run_episode()` | 🔴 Critical | Returns (log_probs, rewards) for 28 steps |
| 5.3 | Implement `REINFORCETrainer.update()` | 🔴 Critical | Correct PG formula with baseline |
| 5.4 | Implement `REINFORCETrainer.train()` loop | 🔴 Critical | 500 episodes, logs avg return |
| 5.5 | Plot REINFORCE training curve | 🔴 Critical | Saved to `results/plots/reinforce_return.png` |
| 5.6 | Plot reward variance diagnostic | 🟡 High | Saved to `results/plots/reinforce_variance.png` |
| 5.7 | Write unit tests for REINFORCETrainer | 🔴 Critical | Update direction is correct |

**Status:** `[x]` done

---

## Phase 6 — A2C (Part E)

| # | Task | Priority | DoD |
|---|------|----------|-----|
| 6.1 | Add `CriticNetwork` to `policy_network.py` | 🔴 Critical | Output scalar |
| 6.2 | Implement `A2CTrainer.run_episode()` | 🔴 Critical | Collects (states, actions, rewards) |
| 6.3 | Implement `A2CTrainer.update()` | 🔴 Critical | Actor + critic updates |
| 6.4 | Implement `A2CTrainer.train()` loop | 🔴 Critical | 500 episodes, logs avg return |
| 6.5 | Plot A2C training curve | 🔴 Critical | Saved to `results/plots/a2c_return.png` |
| 6.6 | Plot REINFORCE vs A2C comparison | 🔴 Critical | Saved to `results/plots/comparison.png` |
| 6.7 | Write unit tests for A2CTrainer | 🔴 Critical | Advantage sign is correct |

**Status:** `[x]` done

---

## Phase 7 — Analysis & Submission (Part F)

| # | Task | Priority | DoD |
|---|------|----------|-----|
| 7.1 | Plot state-level analysis (weekly load or muscle balance) | 🔴 Critical | Saved to `results/plots/state_analysis.png` |
| 7.2 | Assemble `notebooks/ex03_fitness_rl.ipynb` | 🔴 Critical | Runs end-to-end in Colab |
| 7.3 | Write `README.md` (user manual) | 🔴 Critical | Covers all 4 required sections |
| 7.4 | Write `docs/report.pdf` | 🔴 Critical | All 5 analysis questions answered |
| 7.5 | Run `ruff check` → 0 errors | 🔴 Critical | CI passes |
| 7.6 | Run `pytest --cov` → ≥ 85% | 🔴 Critical | Coverage report green |

**Status:** `[~]` in progress (notebook, README, plots, lint, tests done; `docs/report.pdf` (7.4) still outstanding)
