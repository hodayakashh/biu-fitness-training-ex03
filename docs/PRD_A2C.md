# PRD — A2C: Advantage Actor-Critic (Part E)
**Version:** 1.00  
**Project:** BIU DRL Ex03 — Fitness RL  
**Date:** 2026-06-12

---

## 1. Purpose

A2C replaces REINFORCE's high-variance Monte Carlo returns with a **Critic** network that provides a low-variance baseline (the value function V_ψ(s)). The **Actor** selects workout actions; the **Critic** evaluates how good the current state is.

A2C answers:  
> *"Which workout should be chosen now, and is this state above or below average for long-term outcomes?"*

The LSTM transition model is reused as the world model (unchanged from Part C).

---

## 2. Architecture

### Actor — π_θ(a | s)
Identical to REINFORCE policy network:

| Layer | Size | Activation |
|-------|------|-----------|
| Input | 5 | — |
| FC1 | 64 | ReLU |
| FC2 | 32 | ReLU |
| Output | 6 | Softmax |

### Critic — V_ψ(s)

| Layer | Size | Activation |
|-------|------|-----------|
| Input | 5 | — |
| FC1 | 64 | ReLU |
| FC2 | 32 | ReLU |
| Output | 1 | None (regression) |

---

## 3. A2C Update Rule

**TD Error (advantage estimate):**
```
δ_t = r_t + γ · V_ψ(s_{t+1}) − V_ψ(s_t)
```

**Actor update** (policy gradient with advantage):
```
θ ← θ + α_actor · ∇_θ log π_θ(a_t | s_t) · δ_t
```

**Critic update** (minimise TD error squared):
```
ψ ← ψ − α_critic · ∇_ψ (δ_t)²
```

**Entropy bonus** (encourage exploration):
```
θ ← θ + β · ∇_θ H(π_θ(· | s_t))
```

| Hyperparameter | Value | Source |
|----------------|-------|--------|
| γ | 0.99 | `config/setup.json` |
| α_actor | 3e-4 | `config/setup.json` |
| α_critic | 1e-3 | `config/setup.json` |
| Value loss coeff | 0.5 | `config/setup.json` |
| Entropy bonus β | 0.01 | `config/setup.json` |
| Episodes | 500 | `config/setup.json` |
| Episode length T | 28 | `config/setup.json` |

---

## 4. Episode Generation

Same structure as REINFORCE:
```
s_0 = env.reset()
for t = 0 to 27:
    a_t ~ π_θ(· | s_t)
    s_{t+1} = LSTM(s_t, a_t)
    r_t = reward(s_t, a_t, s_{t+1})
    δ_t = r_t + γ·V_ψ(s_{t+1}) − V_ψ(s_t)
    update actor with δ_t
    update critic with δ_t
```

A2C updates **at each step** (online), not after the full episode. This is the key difference from REINFORCE.

---

## 5. Comparison with REINFORCE

| Dimension | REINFORCE | A2C |
|-----------|-----------|-----|
| Return estimate | Monte Carlo (G_t) | TD(0) with critic |
| Update frequency | Per episode | Per step |
| Variance | High | Lower (critic baseline) |
| Bias | Low (unbiased) | Some bias (TD approximation) |
| Convergence speed | Slow | Faster |
| Extra parameters | 0 | Critic network |

---

## 6. Deliverables

- [ ] `src/fitness_rl/services/policy_network.py` — CriticNetwork added
- [ ] `src/fitness_rl/services/a2c_trainer.py` — A2CTrainer class
- [ ] `results/plots/a2c_return.png` — A2C training curve
- [ ] `results/plots/comparison.png` — REINFORCE vs A2C on same axes
- [ ] Discussion in report: why A2C is more stable, trade-offs

---

## 7. Why A2C Is More Stable (to explain in report)

1. **Reduced variance:** The critic subtracts a state-dependent baseline (V_ψ(s_t)) from each reward, focusing the gradient signal on *relative* advantage, not absolute return.
2. **Online updates:** Step-level updates adapt faster than waiting for a full episode.
3. **Credit assignment:** δ_t directly attributes the reward at time t to the action at time t, whereas Monte Carlo G_t includes all future noise.

---

## 8. Known Limitations (to discuss in Part F)

- Critic introduces bias if V_ψ is inaccurate (especially early in training)
- Two networks to tune (actor lr and critic lr must be balanced)
- Potential actor-critic instability if critic diverges
