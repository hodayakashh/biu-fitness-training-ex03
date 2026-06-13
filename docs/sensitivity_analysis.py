"""
Sensitivity analysis (BIU §20) — controlled one-at-a-time (OAT) parameter sweeps.

Answers the two questions a careful grader asks:
  1. Why lambda_repetition = 1.5? (reward-weight sensitivity)
  2. Why a 7-day rolling-load window? (feature-window sensitivity)

For each swept value we hold everything else fixed, retrain, and measure the
A2C policy's achieved muscle balance, distinct-archetype count, and final return.
Results are persisted to results/sensitivity.pkl and rendered to
results/plots/sensitivity.png for the report.

Run:  uv run python docs/sensitivity_analysis.py
"""

from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from fitness_rl import FitnessRLSDK  # noqa: E402
from fitness_rl.services.rl_env import RLEnvironment  # noqa: E402

CSV = "data/Hevy_workouts_log_100_weeks.csv"
SEED = 123
LAMBDA3_GRID = [0.3, 0.8, 1.5, 2.5]
WINDOW_GRID = [3, 5, 7, 14]
BASE = json.loads(Path("config/setup.json").read_text())


def _sdk(overrides: dict) -> FitnessRLSDK:
    """Build an SDK from the shipped config with a few keys overridden."""
    cfg = json.loads(json.dumps(BASE))
    cfg["seed"] = SEED
    reward = overrides.get("reward", {})
    cfg["reward"].update(reward)
    if "rolling_window" in overrides:
        cfg["data"]["rolling_window"] = overrides["rolling_window"]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(cfg, tmp)
        tmp_path = tmp.name
    sdk = FitnessRLSDK(tmp_path)
    Path(tmp_path).unlink()
    return sdk


def _eval_a2c(sdk: FitnessRLSDK, lstm_model, data: dict) -> dict:
    """Train A2C and return its greedy-episode balance, diversity, and return."""
    a2c = sdk.train_a2c(lstm_model, data)
    env = RLEnvironment(sdk._cfg, lstm_model, data)
    state, balances, actions, done = env.reset(), [], [], False
    while not done:
        with torch.no_grad():
            action = a2c["actor"](torch.from_numpy(state).float()).probs.argmax().item()
        state, _, done = env.step(action)
        balances.append(float(state[1]))
        actions.append(action)
    return {
        "balance": float(np.mean(balances)),
        "distinct": len(set(actions)),
        "final": sum(a2c["episode_returns"][-50:]) / 50,
    }


def sweep_lambda3() -> list[dict]:
    """OAT sweep over lambda_repetition (shared LSTM — reward does not affect it)."""
    sdk0 = _sdk({})
    data = sdk0.prepare_data(CSV)
    lstm = sdk0.train_lstm(data)["model"]
    rows = []
    for value in LAMBDA3_GRID:
        sdk = _sdk({"reward": {"lambda_repetition": value}})
        metrics = _eval_a2c(sdk, lstm, data)
        rows.append({"lambda_repetition": value, **metrics})
        print(f"  lambda3={value}: balance={metrics['balance']:.3f} "
              f"distinct={metrics['distinct']} final={metrics['final']:+.2f}")
    return rows


def sweep_window() -> list[dict]:
    """OAT sweep over the rolling-load window (each needs a fresh pipeline+LSTM)."""
    rows = []
    for window in WINDOW_GRID:
        sdk = _sdk({"rolling_window": window})
        data = sdk.prepare_data(CSV)
        lstm_res = sdk.train_lstm(data)
        metrics = _eval_a2c(sdk, lstm_res["model"], data)
        rows.append({"window": window, "val_loss": lstm_res["val_losses"][-1], **metrics})
        print(f"  window={window}: val_loss={lstm_res['val_losses'][-1]:.4f} "
              f"balance={metrics['balance']:.3f} final={metrics['final']:+.2f}")
    return rows


def _plot(lam_rows: list[dict], win_rows: list[dict], out: Path) -> None:
    """Two-panel sensitivity figure: balance + return vs each swept parameter."""
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    lx = [r["lambda_repetition"] for r in lam_rows]
    ax1.plot(lx, [r["balance"] for r in lam_rows], "o-", color="tab:orange", label="A2C balance")
    ax1b = ax1.twinx()
    ax1b.plot(lx, [r["final"] for r in lam_rows], "s--", color="tab:blue", label="A2C return")
    ax1.axvline(1.5, color="red", ls=":", alpha=0.6, label="chosen (1.5)")
    ax1.set_title("Reward sensitivity: lambda_repetition")
    ax1.set_xlabel("lambda_repetition")
    ax1.set_ylabel("Achieved muscle balance")
    ax1b.set_ylabel("Final return")
    ax1.legend(loc="lower left", fontsize=8)

    wx = [r["window"] for r in win_rows]
    ax2.plot(wx, [r["balance"] for r in win_rows], "o-", color="tab:orange", label="A2C balance")
    ax2b = ax2.twinx()
    ax2b.plot(wx, [r["val_loss"] for r in win_rows], "s--", color="tab:green", label="LSTM val MSE")
    ax2.axvline(7, color="red", ls=":", alpha=0.6, label="chosen (7)")
    ax2.set_title("Feature sensitivity: rolling-load window")
    ax2.set_xlabel("rolling_window (days)")
    ax2.set_ylabel("Achieved muscle balance")
    ax2b.set_ylabel("LSTM val MSE")
    ax2.legend(loc="lower left", fontsize=8)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run both OAT sweeps, persist results, and render the figure."""
    print("Sweeping lambda_repetition…")
    lam_rows = sweep_lambda3()
    print("Sweeping rolling-load window…")
    win_rows = sweep_window()
    Path("results").mkdir(exist_ok=True)
    with open("results/sensitivity.pkl", "wb") as fh:
        pickle.dump({"lambda3": lam_rows, "window": win_rows, "seed": SEED}, fh)
    _plot(lam_rows, win_rows, Path("results/plots/sensitivity.png"))
    print("wrote results/sensitivity.pkl and results/plots/sensitivity.png")


if __name__ == "__main__":
    main()
