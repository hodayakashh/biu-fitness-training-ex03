# PRD — Exercise 03: LSTM + REINFORCE + A2C for Personal Fitness Planning
**Version:** 1.01  
**Course:** BIU Deep Reinforcement Learning — Dr. Yoram Segal  
**Author:** Hodaya Kashkash  
**Date:** 2026-06-13

---

## 1. Problem Statement

Personal fitness planning is reframed from a *prediction problem* into a *sequential decision-making problem*.

Instead of asking "what will happen tomorrow?", the system asks:
> **How should a learning agent choose the next daily workout so that long-term training quality improves over 4–6 weeks?**

A workout that seems beneficial today may be harmful two or three days later if it creates excessive cumulative load or muscle-group imbalance. Therefore a *policy* over time is needed, not a one-step predictor.

---

## 2. Scope

| Part | Deliverable | Status |
|------|-------------|--------|
| A | Introduction & motivation | ✅ This PRD |
| B | Kaggle dataset download & preprocessing | ✅ `data_*` services |
| C | LSTM transition model | ✅ `lstm_model` + `lstm_trainer` |
| D | REINFORCE policy gradient | ✅ `reinforce_trainer` |
| E | A2C (Advantage Actor-Critic) | ✅ `a2c_trainer` |
| F | Analysis, plots, conclusions | ✅ `plotter` + notebook analysis cells |

---

## 3. Users

Single user: BIU DRL student submitting to Dr. Segal.  
Grader: Dr. Segal or course TA running the notebook end-to-end.

---

## 4. Functional Requirements

### 4.1 Data Pipeline (Part B)
- **FR-01** Load a Kaggle workout dataset (CSV) into a Pandas DataFrame.
- **FR-02** Compute daily workout summaries from raw exercise-level rows:
  - `total_volume_t` = Σ(sets × reps × weight) per day (fallback: sets × reps if weight unavailable)
  - `muscle_distribution_t` = normalized volume per muscle group (vector, sums to 1)
  - `session_duration_t` = total training time per day (minutes)
  - `day_in_cycle_t` = day index within the 28-day training cycle
- **FR-03** Cluster daily summaries into 6 workout-type classes (action labels) using K-Means.
- **FR-03a** Derive each cluster's empirical muscle-group profile (`action_muscle_profiles`) and a
  data-driven human-readable label (dominant muscle + load tier). Cluster IDs from K-Means are
  arbitrary, so labels MUST NOT be hardcoded against fixed IDs (review finding #3).

### 4.2 LSTM Transition Model (Part C)
- **FR-04** Accept sequences of length `seq_len=7` days as input (state + action embedding per step).
- **FR-05** Predict the next-day state vector `s_{t+1}`.
- **FR-06** Train via supervised learning (MSE loss) on consecutive windows.
- **FR-07** Report train/validation loss curves.

### 4.3 REINFORCE Policy (Part D)
- **FR-08** Policy network π_θ(a|s): MLP → categorical distribution over 6 actions.
- **FR-09** Generate 28-day episodes using the LSTM as the environment model.
- **FR-10** Compute reward r_t = gain_t − λ₁·overload_penalty_t − λ₂·imbalance_penalty_t − λ₃·repetition_penalty_t.
- **FR-10a** The `imbalance_penalty` MUST be action-aware: during rollout the environment governs
  `muscle_balance_score` from the chosen actions' muscle profiles (hybrid world model, ADR-001), so the
  penalty responds to the agent's real choices through the environment state, not only the LSTM prediction.
- **FR-11** Update policy with episodic policy-gradient (Monte Carlo returns G_t).
- **FR-12** Log average episodic return per training iteration; plot reward variance.

### 4.4 A2C Policy (Part E)
- **FR-13** Actor network: same architecture as REINFORCE policy.
- **FR-14** Critic network V_ψ(s): MLP → scalar value estimate.
- **FR-15** Update actor with TD advantage δ_t = r_t + γ·V(s_{t+1}) − V(s_t).
- **FR-16** Update critic with MSE value loss.
- **FR-17** Produce comparison plots: REINFORCE vs A2C on convergence, stability, final return.

### 4.5 Analysis (Part F)
- **FR-18** State-level analysis plot (rolling weekly load, muscle balance, or action distribution).
- **FR-19** Structured discussion addressing all 5 required analysis questions.

---

## 5. Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| NFR-01 | All code passes `ruff check` with zero errors |
| NFR-02 | Test coverage ≥ 85% (unit + integration) |
| NFR-03 | No file exceeds 150 lines (excluding blank/comment lines) |
| NFR-04 | No hardcoded values — all config via `config/setup.json` |
| NFR-05 | All dependencies managed via `uv`; `pyproject.toml` is single source of truth |
| NFR-06 | No secrets in source code; `.env-example` committed with dummy values |
| NFR-07 | Version tracking at `"1.01"` in code and all config files |
| NFR-08 | Notebook runs end-to-end in Google Colab without errors |

---

## 6. Out of Scope

- Real-time data collection from wearables or fitness APIs
- Multi-user or production deployment
- GUI or web interface
- Any algorithm beyond LSTM, REINFORCE, A2C

---

## 7. Submission Artefacts

1. `notebooks/ex03_fitness_rl.ipynb` — runnable Colab notebook (end-to-end)
2. `docs/report.pdf` — short structured report
3. `results/plots/` — all required figures (≥5)
4. `README.md` — user manual explaining dataset, state/action/reward, and run instructions

---

## 8. Approval Checkpoint

> **⚠ Do not write any code until this PRD is approved.**

Mechanism-specific PRDs:
- [`docs/PRD_LSTM.md`](PRD_LSTM.md)
- [`docs/PRD_REINFORCE.md`](PRD_REINFORCE.md)
- [`docs/PRD_A2C.md`](PRD_A2C.md)
