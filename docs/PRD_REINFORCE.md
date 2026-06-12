# PRD — REINFORCE Policy Gradient (Part D)
**Version:** 1.00  
**Project:** BIU DRL Ex03 — Fitness RL  
**Date:** 2026-06-12

---

## 1. Purpose

REINFORCE trains a stochastic policy π_θ(a|s) to maximise expected cumulative reward over a 28-day training episode. It uses the LSTM transition model as the environment and updates the policy **after each complete episode** using Monte Carlo returns.

REINFORCE answers:  
> *"Which workout type should be chosen today to improve long-term outcomes?"*

---

## 2. Policy Formulation

**Policy:** π_θ(a_t | s_t) — categorical distribution over 6 workout types, parameterised by a neural network θ.

**Episode structure:**
```
for t = 0 to 27:
    a_t ~ π_θ(· | s_t)
    s_{t+1} = LSTM(s_t, a_t)      # LSTM as world model
    r_t = reward(s_t, a_t, s_{t+1})
```

**Return (Monte Carlo):**
```
G_t = Σ_{k=t}^{T} γ^{k-t} · r_k     γ = 0.99
```

---

## 3. Network Architecture

| Layer | Size | Activation |
|-------|------|-----------|
| Input | state_dim = 4 | — |
| FC1 | 64 | ReLU |
| FC2 | 32 | ReLU |
| Output | N_ACTIONS = 6 | Softmax |

---

## 4. Update Rule

**Standard REINFORCE with baseline:**

```
b = mean(G_t)  over episode     # reduce variance

∇θ J(θ) ≈ Σ_t ∇_θ log π_θ(a_t | s_t) · (G_t − b)

θ ← θ + α · ∇θ J(θ)
```

| Hyperparameter | Value | Source |
|----------------|-------|--------|
| Discount γ | 0.99 | `config/setup.json` |
| Learning rate α | 3e-4 | `config/setup.json` |
| Optimizer | Adam | `config/setup.json` |
| Episodes | 500 | `config/setup.json` |
| Episode length T | 28 | `config/setup.json` |
| Baseline | Episode mean of G_t | Hardcoded in trainer |

---

## 5. Reward Function

```
r_t = gain_t − λ₁ · overload_penalty_t − λ₂ · imbalance_penalty_t
```

| Term | Formula | Purpose |
|------|---------|---------|
| `gain_t` | `clip(volume_t / VOLUME_TARGET, 0, 1)` | Reward productive training |
| `overload_penalty_t` | `max(0, rolling_load_t − OVERLOAD_THRESHOLD) / OVERLOAD_THRESHOLD` | Penalise overtraining |
| `imbalance_penalty_t` | `std(muscle_distribution over last 7 days)` | Penalise imbalance |
| λ₁ | 0.4 | `config/setup.json` |
| λ₂ | 0.3 | `config/setup.json` |

---

## 6. Expected Behaviour

- Early episodes: random policy, low/negative rewards, high variance
- After ~100 episodes: returns should trend upward
- Converged policy: avoids back-to-back heavy sessions; mixes workout types

---

## 7. Deliverables

- [ ] `src/fitness_rl/services/policy_network.py` — PolicyNetwork class
- [ ] `src/fitness_rl/services/reinforce_trainer.py` — REINFORCETrainer class
- [ ] `results/plots/reinforce_return.png` — average episodic return curve
- [ ] `results/plots/reinforce_variance.png` — return variance per episode
- [ ] Discussion in report: variance and instability observations

---

## 8. Known Limitations (to discuss in Part F)

- High variance due to Monte Carlo returns — no critic to reduce noise
- Slow convergence compared to A2C (credit assignment problem)
- Sensitive to learning rate; may diverge if α too large
