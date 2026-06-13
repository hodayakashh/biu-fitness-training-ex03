# PLAN — Architecture & Design
**Version:** 1.01
**Project:** BIU DRL Ex03 — Fitness RL
**Date:** 2026-06-13

---

## 1. High-Level Architecture

```
External (Colab / CLI)
        │
        ▼
  ┌─────────────┐
  │     SDK     │  src/fitness_rl/sdk/sdk.py
  │  FitnessRL  │  — single entry point for all logic
  └──────┬──────┘
         │
   ┌─────┴──────────────────────────────┐
   │                                    │
   ▼                                    ▼
┌──────────────────┐          ┌────────────────────┐
│  DataService     │          │  ModelService      │
│  (preprocessing) │          │  (LSTM / Policy)   │
└──────────────────┘          └────────────────────┘
         │                              │
         ▼                              ▼
┌──────────────────┐          ┌────────────────────┐
│  RLEnvironment   │          │  TrainingService   │
│  (step/reset)    │◄─────────│  (REINFORCE / A2C) │
└──────────────────┘          └────────────────────┘
         │
         ▼
   ConfigManager / ApiGatekeeper / VersionTracker
```

---

## 2. Module Map

| Module | File | Responsibility | Max lines |
|--------|------|---------------|-----------|
| SDK | `sdk/sdk.py` | Single public entry-point | 150 |
| DataService | `services/data_service.py` | Pipeline orchestrator: ties the four data sub-modules together | 150 |
| DataPreprocessor | `services/data_preprocessor.py` | Daily summaries + K-Means action clusters + per-cluster muscle profiles & data-driven labels | 150 |
| DataFeatures | `services/data_features.py` | Rolling load/duration, muscle-balance entropy, sin/cos day encoding | 80 |
| DataNormalizer | `services/data_normalizer.py` | MinMaxScaler fitted on train split only | 100 |
| DataWindows | `services/data_windows.py` | Sliding-window (state, action, next-state) tensors | 80 |
| LSTMModel | `services/lstm_model.py` | LSTM architecture + train/predict | 150 |
| LSTMTrainer | `services/lstm_trainer.py` | Supervised training loop, loss curves | 100 |
| RLEnvironment | `services/rl_env.py` | step(), reset() — hybrid world model: LSTM + action-conditioned muscle balance (ADR-001) | 150 |
| RewardFunction | `services/reward.py` | gain + overload/imbalance/repetition penalties (injected building block) | 100 |
| PolicyNetwork | `services/policy_network.py` | Actor MLP + Critic MLP | 100 |
| REINFORCETrainer | `services/reinforce_trainer.py` | Episode generation + PG update + entropy bonus | 150 |
| A2CTrainer | `services/a2c_trainer.py` | TD-error, actor+critic updates | 130 |
| Plotter | `services/plotter.py` | All matplotlib figures | 120 |
| ConfigManager | `shared/config.py` | Load config/setup.json | 80 |
| ApiGatekeeper | `shared/gatekeeper.py` | Centralized external call manager | 100 |
| Seeding | `shared/seeding.py` | Global RNG seed for reproducible, fair comparison | 30 |
| VersionTracker | `shared/version.py` | Version string "1.01" | 20 |
| Constants | `constants.py` | Enums, immutable constants | 60 |

---

## 3. State, Action, Reward Design

### 3.1 State Vector s_t (dim = 5)

| Index | Feature | Description | Encoding |
|-------|---------|-------------|----------|
| 0 | `rolling_7day_load` | Sum of `total_volume` over last 7 days | MinMaxScaler → [0, 1] |
| 1 | `muscle_balance_score` | Entropy of muscle-group distribution (higher = more balanced). **During RL rollout this dimension is governed by the chosen action's muscle profile — see ADR-001.** | MinMaxScaler → [0, 1] |
| 2 | `session_duration_avg` | Rolling 7-day average session duration | MinMaxScaler → [0, 1] |
| 3 | `day_sin` | sin(2π · t / 28) | Sinusoidal — preserves cyclicality |
| 4 | `day_cos` | cos(2π · t / 28) | Sinusoidal — preserves cyclicality |

**Why sinusoidal for day_in_cycle:**
A linear fraction t/27 breaks cyclicality — day 27 and day 0 are numerically far apart but
conceptually adjacent. sin/cos encode that day 0 ≡ day 28, exactly as a circular feature should.

**Scaler fit rule (data leakage prevention):**
MinMaxScaler is fitted **only on the training split** (first 80% of daily summaries),
then applied (transform only, no re-fit) on the validation split.
This prevents future statistics from leaking into the LSTM training signal.

### 3.2 Action Space a_t (discrete, N_ACTIONS = 6)

K-Means clusters (k=6) on daily summaries yield 6 workout archetypes. **Cluster
IDs from K-Means are arbitrary**, so human-readable names are NOT hardcoded against
fixed IDs. Instead each cluster is labelled *from its own data* — by its dominant
muscle group (argmax of the cluster's mean `mg_*` profile) plus a load tier derived
from the cluster's mean `total_volume` tercile (see `DataPreprocessor.describe_clusters`).
A representative labelling on the Hevy dataset (seed 42):

| Action (cluster id) | Dominant muscle | Mean volume | Data-driven label |
|--------|-------|----------------|-------------------|
| 0 | cardio / abdominals | ~280 | Core/Cardio (low) |
| 1 | chest | ~5.5k | Chest (moderate) |
| 2 | quadriceps | ~13.9k | Quadriceps (high) |
| 3 | cardio (pure) | ~0 | Cardio (low) |
| 4 | chest | ~3.2k | Chest (moderate) |
| 5 | shoulders / lower-back | ~8.8k | Shoulders (high) |

Each cluster's mean muscle-group distribution is also exported as
`action_muscle_profiles` (shape `n_actions × n_muscle_groups`) and consumed by the
RL environment to drive the muscle-balance dynamics (ADR-001).

### 3.3 Reward Function

```
r_t = gain_t − λ₁ · overload_penalty_t − λ₂ · imbalance_penalty_t − λ₃ · repetition_penalty_t
```

- `gain_t = max(0, 1 − |rolling_load − optimal_load_norm| / optimal_load_norm)` — bell-shaped, peaks at `optimal_load_norm = 0.5` so the agent is not rewarded for unbounded volume
- `overload_penalty_t = max(0, rolling_load − overload_threshold_norm) / (1 − overload_threshold_norm)` — all in normalised [0, 1] state space (`overload_threshold_norm = 0.8`)
- `imbalance_penalty_t = 1 − muscle_balance_score`. **As of v1.01 this is the principled, action-aware variety driver** (ADR-001): during rollout the environment overwrites `muscle_balance_score` with the normalised Shannon entropy of the *mean muscle-group profile of the agent's last-`variety_window` chosen actions*. Hammering one archetype → concentrated muscle exposure → low entropy → high imbalance penalty; alternating archetypes that target different muscle groups → high entropy → low penalty. This directly answers the assignment's "did the policy avoid overloading one muscle group?" through the **environment state**, not an external hack.
- `repetition_penalty_t = 1 − H(recent_actions) / log(N_ACTIONS)` — computed from the entropy of the agent's OWN recent actions. **Empirical finding:** the action-aware imbalance penalty alone does *not* fully prevent greedy-rollout collapse, so this term is weighted **co-equally** with the imbalance penalty — both are needed to restore genuine workout variety (a co-dependence documented in `docs/report.pdf §6`).
- λ₁ = 0.4, λ₂ = 1.5, λ₃ = 1.5, `variety_window = 7` (all from config; code defaults kept in sync with `setup.json` to avoid silent divergence).
- The reward is implemented as an injected `RewardFunction` building block (`services/reward.py`) — single responsibility, independently testable.

---

## 3.4 ADR-001 — Hybrid world model: action-conditioned muscle-balance dynamics

**Status:** Accepted (v1.01) · **Date:** 2026-06-13

**Context.** The original design used the LSTM as the *sole* transition model for all
five state dimensions, including `muscle_balance_score`. Empirically the LSTM "barely
differentiates between actions": the K-Means `action_label` is a near-deterministic
function of the same daily features that form the state, so action and state are
collinear in the training windows and the LSTM learns to predict the next state from
state history while largely ignoring the action embedding. Two consequences:

1. **Degenerate MDP (review finding #1):** the agent's choice barely changed the
   predicted next state, so "decision-making" had little causal grip on the environment.
2. **Inert imbalance penalty (review finding #2):** `λ₂·imbalance` read a state
   dimension the LSTM could not make action-dependent, so it never reacted to the
   agent's real choices. A `repetition_penalty` computed *outside* the world model was
   bolted on as the actual variety driver — a hack that optimises a function of the
   agent's own action sequence rather than anything mediated by the environment.

**Decision.** Adopt a **hybrid world model**:
- The LSTM remains the learned transition model for the genuinely history-dependent,
  fatigue-related dimensions (`rolling_7day_load`, `session_duration_avg`) and the
  deterministic temporal encoding (`day_sin`, `day_cos`).
- The `muscle_balance_score` dimension is governed by a **known, action-conditioned
  kinematic**: each K-Means cluster carries an empirical muscle-group profile
  (`action_muscle_profiles`), the environment tracks a rolling window of the profiles
  of the agent's chosen actions, and `muscle_balance_score` becomes the normalised
  Shannon entropy of that window's mean profile. The environment overwrites the LSTM's
  prediction for this single dimension at each step.

**Consequences.**
- **+** Actions now have a real, persistent causal effect on the state, and the
  imbalance penalty is action-aware *through the environment* — fixing #1 and #2 while
  staying faithful to the assignment's reward formula.
- **+** This is the "principled fix" the prior report's Limitations section promised.
- **−** The LSTM is no longer the *whole* world model. This is a deliberate, documented
  decomposition ("known + learned dynamics"), standard in model-based RL, and is honest
  about which dynamics are learned vs. analytic.
- **Alternative considered & rejected:** expand the state with per-muscle-group rolling
  exposure (one dim per muscle group) and let the LSTM learn the action→muscle mapping.
  Rejected because action/state collinearity in the training data means the LSTM still
  could not reliably recover the mapping, and it bloats `state_dim` from 5 to ~20.

---

## 4. LSTM Architecture

```
Input: [seq_len=7, state_dim + action_embedding_dim]
       state_dim = 5,  action_embedding_dim = 8  →  input_size = 13

LSTM Layer 1: hidden=64, dropout=0.2
LSTM Layer 2: hidden=64, dropout=0.2
FC Layer:     64 → state_dim (5)

Loss: MSE(s_pred_{t+1}, s_true_{t+1})
Optimizer: Adam(lr=1e-3)
Batch size: 64, Epochs: 50, seq_len: 7
```

---

## 5. Policy Networks

### REINFORCE Actor
```
Input: state_dim (5)
FC1: 5 → 64, ReLU
FC2: 64 → 32, ReLU
FC3: 32 → N_ACTIONS (6)
Output: Softmax → categorical distribution
```

### A2C Actor (same as REINFORCE)
### A2C Critic
```
Input: state_dim (5)
FC1: 5 → 64, ReLU
FC2: 64 → 32, ReLU
FC3: 32 → 1
Output: scalar V(s)
```

---

## 6. Training Protocols

### REINFORCE
- Episodes: 500
- Episode length: 28 days
- Return: G_t = Σ_{k≥t} γ^{k-t} r_k  (γ = 0.99)
- Baseline: subtract mean(G_t) per episode to reduce variance
- Optimizer: Adam(lr=3e-4)

### A2C
- Episodes: 500
- Episode length: 28 days
- γ = 0.99
- Actor lr: 3e-4, Critic lr: 1e-3
- Value loss coefficient: 0.5
- Entropy bonus coefficient: 0.01

---

## 7. Data Flow

```
CSV  ──►  DataService.load()
              │
              ▼
         daily_summaries (DataFrame: N_days × features)
              │
         DataService.cluster_actions() → action labels (K-Means k=6)
              │
         DataService.build_windows() → (X_seq, y_next_state) tensors
              │
              ▼
         LSTMTrainer.train() ── saved weights ──► RLEnvironment.step()
                                                       │
                                                  PolicyTrainer.train()
                                                       │
                                                  Plotter.save_all()
```

---

## 8. Config Hierarchy

```
config/setup.json          ← global seed, LSTM hyperparams, RL hyperparams, reward λ values
config/rate_limits.json    ← Gatekeeper limits (for any Kaggle API calls)
.env                       ← KAGGLE_USERNAME, KAGGLE_KEY  (git-ignored)
.env-example               ← placeholder values (committed)
src/fitness_rl/constants.py ← Immutable Enums (ACTION names, feature indices)
```

---

## 9. Testing Strategy

| Layer | Scope | Tool |
|-------|-------|------|
| Unit | DataService transformations, reward calc, state normalisation | pytest |
| Unit | Scaler fitted on train split only — val MSE differs from train baseline | pytest |
| Unit | LSTM forward pass shape correctness | pytest + torch |
| Unit | Policy network output is valid probability distribution | pytest |
| Integration | End-to-end episode generation (LSTM + Policy) | pytest |
| Integration | REINFORCE loss decreases over 10 episodes | pytest |
| Coverage gate | ≥ 85% | `pytest --cov` |
| Lint gate | 0 violations | `ruff check` |
