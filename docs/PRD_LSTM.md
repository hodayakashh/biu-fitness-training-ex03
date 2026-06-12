# PRD — LSTM Transition Model (Part C)
**Version:** 1.00  
**Project:** BIU DRL Ex03 — Fitness RL  
**Date:** 2026-06-12

---

## 1. Purpose

The LSTM serves as a **learned transition model** (world model) for the RL environment.  
It approximates the environment dynamics:

```
s_{t+1} ≈ f_φ(s_t, a_t, h_t)
```

where `h_t` is the hidden state capturing temporal history.  
The LSTM does **not** make decisions — it answers:  
> *"Given the recent training history and today's chosen workout, what is the likely next trainee state?"*

---

## 2. Inputs and Outputs

### Input
| Dimension | Description |
|-----------|-------------|
| `seq_len = 7` | Window of the past 7 days |
| `state_dim = 4` | Features per day: rolling_load, muscle_balance, duration_avg, day_in_cycle |
| `action_embed_dim = 8` | Learned embedding of the discrete action (0–5) |
| `input_size = 12` | Concatenated state + action embedding per timestep |

Input tensor shape: `(batch_size, seq_len=7, input_size=12)`

### Output
| Dimension | Description |
|-----------|-------------|
| `state_dim = 4` | Predicted next-day state vector `ŝ_{t+1}` |

Output tensor shape: `(batch_size, state_dim=4)`

---

## 3. Architecture

```
Embedding(N_ACTIONS=6, action_embed_dim=8)
                │
        concat with s_t
                │
          LSTM Layer 1
          hidden=64, dropout=0.2
                │
          LSTM Layer 2
          hidden=64, dropout=0.2
                │
         FC: 64 → state_dim(4)
                │
         output ŝ_{t+1}
```

- **Number of LSTM layers:** 2
- **Hidden size per layer:** 64
- **Dropout between layers:** 0.2
- **Output activation:** None (regression, MSE loss)

---

## 4. Training Protocol

| Hyperparameter | Value | Source |
|----------------|-------|--------|
| Loss | MSE(ŝ_{t+1}, s_{t+1}) | |
| Optimizer | Adam | `config/setup.json` |
| Learning rate | 1e-3 | `config/setup.json` |
| Batch size | 64 | `config/setup.json` |
| Epochs | 50 | `config/setup.json` |
| Train/val split | 80/20 | `config/setup.json` |
| Sequence length | 7 | `config/setup.json` |

**Data construction:**  
Sliding window over daily summaries:  
`X[i] = daily_summaries[i : i+seq_len]` (states + actions)  
`y[i] = daily_summaries[i+seq_len]` (next state only)

---

## 5. Evaluation Criteria

| Criterion | Target |
|-----------|--------|
| Validation MSE | < train MSE (no overfitting) |
| Loss curve | Monotonically decreasing or plateau |
| Predicted state range | Within [0, 1] (normalised) |
| Temporal coherence | Consecutive predictions don't flip wildly |

---

## 6. Deliverables

- [ ] `src/fitness_rl/services/lstm_model.py` — model architecture
- [ ] `src/fitness_rl/services/lstm_trainer.py` — training loop
- [ ] `results/lstm_weights.pt` — saved weights
- [ ] `results/plots/lstm_loss.png` — train/val loss curves
- [ ] Discussion in report: "Does the LSTM capture meaningful temporal patterns?"

---

## 7. Limitations (to discuss in Part F)

- Trained on *historical* data; may not generalise to policy-induced distributions
- Deterministic predictor — no uncertainty estimate
- 7-day window may miss monthly periodisation cycles
- MSE loss treats all state dimensions equally (may need weighted loss)
