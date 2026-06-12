# BIU DRL Exercise 03 — Fitness RL with LSTM + REINFORCE + A2C

**Course:** Deep Reinforcement Learning — Dr. Yoram Segal, Bar-Ilan University  
**Author:** Hodaya Kashkash

---

## Research Question

> How can a learning system use historical workout data to choose a sequence of daily training actions that improves long-term training quality under fatigue and balance constraints?

The project reframes personal fitness planning as a **sequential decision-making problem**: instead of predicting what will happen tomorrow, an RL agent learns which workout to assign today so that cumulative training quality improves over a 4-week cycle.

---

## Kaggle Dataset

This project uses the Hevy workout-log dataset, pinned in
`config/setup.json → data.kaggle_dataset`:

```
tejaswinimukesh/hevy-app-workout-dataset-from-dumbbells-to-data
```

It provides per-session exercise records with reps, weight, duration, and muscle
group — enough to derive daily summaries (total volume, muscle distribution,
session duration, day-in-cycle). Any structured workout log with the same columns
can be substituted by editing `config/setup.json → data.columns`.

**Download via the Kaggle API (managed through `uv`):**
```bash
mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

uv run kaggle datasets download -d tejaswinimukesh/hevy-app-workout-dataset-from-dumbbells-to-data
unzip hevy-app-workout-dataset-from-dumbbells-to-data.zip -d data/
```

If you swap in a different dataset, update `config/setup.json → data.columns` to
match the actual CSV column names.

---

## State, Action, and Reward

### State vector `s_t` (dim = 5)

| Index | Feature | Encoding |
|-------|---------|----------|
| 0 | `rolling_7day_load` | MinMaxScaler → [0, 1] |
| 1 | `muscle_balance_score` | Shannon entropy of muscle distribution → [0, 1] |
| 2 | `session_duration_avg` | MinMaxScaler → [0, 1] |
| 3 | `day_sin` | sin(2π · t / 28) |
| 4 | `day_cos` | cos(2π · t / 28) |

Sinusoidal encoding for day-in-cycle preserves cyclicality (day 0 ≡ day 28).

### Action space `a_t` (6 discrete actions)

K-Means (k=6) clusters daily workout sessions into archetypes:

| Action | Label |
|--------|-------|
| 0 | Rest / Recovery |
| 1 | Upper Push (chest / shoulders / triceps) |
| 2 | Upper Pull (back / biceps) |
| 3 | Lower Body (legs / glutes) |
| 4 | Full Body |
| 5 | Core / Cardio |

### Reward function

```
r_t = gain_t − λ₁ · overload_penalty_t − λ₂ · imbalance_penalty_t

gain_t            = bell-shaped peak at optimal_load_norm (default 0.5)
overload_penalty_t = max(0, load − overload_threshold_norm) / (1 − threshold)
imbalance_penalty_t= 1 − muscle_balance_score

λ₁ = 0.4,  λ₂ = 0.3  (configurable in config/setup.json)
```

---

## How to Run

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure Kaggle credentials

```bash
cp .env-example .env
# Edit .env: set KAGGLE_USERNAME and KAGGLE_KEY
```

### 3. Download the dataset

```bash
uv run kaggle datasets download -d tejaswinimukesh/hevy-app-workout-dataset-from-dumbbells-to-data
unzip hevy-app-workout-dataset-from-dumbbells-to-data.zip -d data/
```

### 4. Run the notebook

```bash
uv run jupyter notebook notebooks/ex03_fitness_rl.ipynb
```

Or upload `notebooks/ex03_fitness_rl.ipynb` to Google Colab.

---

## Project Structure

```
biu-drl-ex03/
├── src/fitness_rl/
│   ├── sdk/sdk.py               # Single entry point — FitnessRLSDK
│   ├── services/
│   │   ├── data_service.py      # Full data pipeline orchestrator
│   │   ├── data_preprocessor.py # Daily summaries, K-Means action clusters
│   │   ├── data_normalizer.py   # MinMaxScaler (fit on train only)
│   │   ├── data_windows.py      # Sliding window tensors for LSTM
│   │   ├── data_features.py     # Rolling stats, sin/cos encoding
│   │   ├── lstm_model.py        # LSTMTransitionModel
│   │   ├── lstm_trainer.py      # Supervised LSTM training loop
│   │   ├── rl_env.py            # RLEnvironment (step / reset / reward)
│   │   ├── policy_network.py    # PolicyNetwork (actor) + CriticNetwork
│   │   ├── reinforce_trainer.py # REINFORCE policy gradient
│   │   ├── a2c_trainer.py       # A2C advantage actor-critic
│   │   └── plotter.py           # All 5 required figures
│   └── shared/
│       ├── config.py            # ConfigManager (reads setup.json)
│       ├── gatekeeper.py        # API rate-limit gatekeeper
│       └── version.py           # Version compatibility check
├── config/
│   ├── setup.json               # All hyperparameters (no hardcoded values)
│   └── rate_limits.json         # Kaggle API rate limits
├── docs/
│   ├── PRD.md, PLAN.md, TODO.md
│   └── PRD_LSTM.md, PRD_REINFORCE.md, PRD_A2C.md
├── notebooks/
│   └── ex03_fitness_rl.ipynb    # End-to-end Colab notebook
├── tests/                       # 114+ unit + integration tests (≥ 96% coverage)
├── data/                        # Place downloaded CSV files here
├── results/plots/               # Generated figures (auto-created)
├── pyproject.toml               # uv dependencies + ruff + pytest config
└── .env-example                 # Kaggle credentials template
```

---

## Running Tests

```bash
uv run pytest tests/ -v --tb=short
uv run ruff check src/ tests/
```

Coverage ≥ 85% is enforced automatically.
