"""
run_experiments.py — reproducible regeneration of all Ex03 results and figures.

Single canonical producer (supersedes the old episode_analysis.py). It:
  1. trains the LSTM world model + REINFORCE + A2C on the real Hevy data,
  2. runs greedy 28-day episodes for each learned policy,
  3. computes the comparison metrics + a 5-seed robustness sweep,
  4. writes results/training_results.pkl (consumed by docs/generate_report.py),
  5. saves all seven report figures — using the DATA-DRIVEN cluster labels
     (dominant muscle + load tier), not hardcoded archetype names (finding #3).

Usage:
    uv run python docs/run_experiments.py
    uv run python docs/generate_report.py   # then build the PDF
"""

from __future__ import annotations

import json
import os
import pickle
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from fitness_rl import FitnessRLSDK
from fitness_rl.services.plotter import Plotter
from fitness_rl.services.rl_env import RLEnvironment

DATA_CSV = Path("data/Hevy_workouts_log_100_weeks.csv")
CONFIG_PATH = Path("config/setup.json")
PLOTS_DIR = Path("results/plots")
RESULTS_PKL = Path("results/training_results.pkl")

N_EPISODES = 500
SWEEP_SEEDS = [0, 1, 7, 42, 123]
REPORTED_SEED = 123
OPTIMAL_LOAD = 0.5
RF_COLOR, A2C_COLOR = "tab:blue", "tab:orange"

plt.style.use("seaborn-v0_8-darkgrid")


def build_sdk(seed: int, n_episodes: int = N_EPISODES) -> FitnessRLSDK:
    """Return an SDK whose config overrides only the seed and RL episode count."""
    with CONFIG_PATH.open() as fh:
        cfg = json.load(fh)
    cfg["seed"] = seed
    cfg["rl"]["n_episodes"] = n_episodes
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(cfg, tmp)
        tmp_name = tmp.name
    sdk = FitnessRLSDK(tmp_name)
    os.unlink(tmp_name)
    return sdk


def greedy_episode(policy, env: RLEnvironment) -> tuple[list[int], list[np.ndarray]]:
    """Run one greedy (argmax) episode; return per-step actions and states."""
    state = env.reset()
    actions: list[int] = []
    states: list[np.ndarray] = []
    done = False
    while not done:
        with torch.no_grad():
            action = int(policy(torch.from_numpy(state).float()).probs.argmax().item())
        state, _reward, done = env.step(action)
        actions.append(action)
        states.append(state.copy())
    return actions, states


def convergence_episode(returns: list[float], win: int = 20) -> int:
    """Episode at which the rolling-mean return first reaches 90% of init→final."""
    series = pd.Series(returns).rolling(win, min_periods=1).mean().to_numpy()
    init, final = series[0], series[-1]
    if final == init:
        return 1
    target = init + 0.9 * (final - init)
    crossed = series >= target if final > init else series <= target
    return int(np.argmax(crossed)) + 1


def std_pair(returns: list[float]) -> tuple[float, float]:
    """Std of episodic return over the first vs last 100 episodes."""
    return float(np.std(returns[:100])), float(np.std(returns[-100:]))


def action_distribution(actions: list[int], labels: dict[int, str], n: int) -> dict[str, int]:
    """Map each cluster's data-driven label to its count over the episode."""
    return {labels[i]: actions.count(i) for i in range(n)}


def final_return(returns: list[float]) -> float:
    """Mean return over the last 50 episodes."""
    return float(np.mean(returns[-50:]))


# ---------------------------------------------------------------------------
# Episode figures (data-driven labels)
# ---------------------------------------------------------------------------


def plot_action_distribution(rf_a, a2c_a, labels, n, out_path):
    """Side-by-side bar chart of REINFORCE vs A2C action counts."""
    names = [labels[i] for i in range(n)]
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(11, 5))
    for offset, acts, lab, color in [
        (-0.2, rf_a, "REINFORCE", RF_COLOR),
        (0.2, a2c_a, "A2C", A2C_COLOR),
    ]:
        counts = [acts.count(i) for i in range(n)]
        bars = ax.bar(x + offset, counts, 0.4, label=lab, color=color)
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                str(int(bar.get_height())),
                ha="center",
                va="bottom",
                fontsize=8,
                color=color,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Days in 28-day episode")
    ax.set_title("Learned Policy — Action Distribution over 28-Day Episode")
    ax.legend()
    _save(fig, out_path)


def plot_episode_trajectory(actions, states, labels, n, out_path):
    """A2C episode: action per day (top) + normalised rolling load (bottom)."""
    days = list(range(1, len(actions) + 1))
    load = [s[0] for s in states]
    colors = plt.cm.tab10(np.linspace(0, 0.6, n))
    fig, (top, bot) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    top.scatter(days, actions, c=[colors[a] for a in actions], s=70, zorder=3)
    top.vlines(days, 0, actions, colors="grey", linewidth=0.5, alpha=0.5)
    top.set_yticks(range(n))
    top.set_yticklabels([labels[i] for i in range(n)], fontsize=7)
    top.set_title("A2C Policy — 28-Day Episode: action (top) and rolling load (bottom)")
    bot.plot(days, load, color=A2C_COLOR, linewidth=2, marker="o", markersize=4)
    bot.axhline(OPTIMAL_LOAD, linestyle="--", color="red", linewidth=1, label="optimal=0.5")
    bot.set_xlabel("Day")
    bot.set_ylabel("Norm rolling load")
    bot.legend(fontsize=8)
    _save(fig, out_path)


def plot_muscle_exposure(actions, profiles, mg_cols, balance, out_path):
    """
    Cumulative muscle-group exposure the policy produces over the episode.

    The mean of the chosen actions' muscle profiles directly shows whether any single
    muscle group is overloaded — the professor's central question. A flat bar chart
    means balanced training; a single tall bar means over-concentration.
    """
    exposure = np.mean([profiles[a] for a in actions], axis=0)
    order = np.argsort(exposure)[::-1]
    names = [mg_cols[i].removeprefix("mg_") for i in order]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(range(len(names)), exposure[order], color="tab:purple")
    ax.set_ylabel("Share of training volume")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    ax.set_title(
        f"A2C Policy — Cumulative Muscle-Group Exposure over 28 Days "
        f"(balance entropy = {balance:.2f} / 1.0)"
    )
    _save(fig, out_path)


def plot_state_analysis(states, out_path):
    """Normalised rolling load over the A2C-generated episode."""
    days = list(range(1, len(states) + 1))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(days, [s[0] for s in states], color="tab:green", linewidth=2, marker="o", markersize=4)
    ax.axhline(OPTIMAL_LOAD, linestyle="--", color="red", linewidth=1, label="optimal=0.5")
    ax.set_xlabel("Day")
    ax.set_ylabel("Normalised Rolling Load (state[0])")
    ax.set_title("A2C Policy — Generated Episode: Normalised Rolling Load over 28 Days")
    ax.legend(fontsize=8)
    _save(fig, out_path)


def _save(fig, out_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def mean_balance(states: list[np.ndarray]) -> float:
    """Mean achieved muscle_balance (state[1]) over an episode — higher = better balanced."""
    return float(np.mean([s[1] for s in states]))


def run_seed(seed: int, lstm_model, data: dict) -> dict:
    """Train REINFORCE + A2C for one seed on the shared LSTM world model."""
    sdk = build_sdk(seed)
    rf = sdk.train_reinforce(lstm_model, data)
    a2c = sdk.train_a2c(lstm_model, data)
    env_rf = RLEnvironment(sdk._cfg, lstm_model, data)
    env_a2c = RLEnvironment(sdk._cfg, lstm_model, data)
    rf_actions, rf_states = greedy_episode(rf["policy"], env_rf)
    a2c_actions, a2c_states = greedy_episode(a2c["actor"], env_a2c)
    return {
        "sdk": sdk,
        "rf": rf,
        "a2c": a2c,
        "rf_actions": rf_actions,
        "a2c_actions": a2c_actions,
        "rf_states": rf_states,
        "a2c_states": a2c_states,
        "rf_balance": mean_balance(rf_states),
        "a2c_balance": mean_balance(a2c_states),
    }


def main() -> None:
    """Train, evaluate, sweep seeds, persist results, and render all figures."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Preparing data + training LSTM (seed {REPORTED_SEED})…")
    sdk = build_sdk(REPORTED_SEED)
    data = sdk.prepare_data(DATA_CSV)
    labels: dict[int, str] = data["action_labels"]
    n_actions = len(labels)
    print(f"  data-driven action labels: {labels}")

    lstm = sdk.train_lstm(data)
    lstm_model = lstm["model"]

    print("Training + evaluating REINFORCE and A2C across seeds…")
    by_seed = {s: run_seed(s, lstm_model, data) for s in SWEEP_SEEDS}
    main_run = by_seed[REPORTED_SEED]
    rf, a2c = main_run["rf"], main_run["a2c"]
    rf_actions, a2c_actions = main_run["rf_actions"], main_run["a2c_actions"]
    a2c_states = main_run["a2c_states"]

    sweep = [
        (
            s,
            by_seed[s]["rf_balance"],
            final_return(by_seed[s]["rf"]["episode_returns"]),
            by_seed[s]["a2c_balance"],
            final_return(by_seed[s]["a2c"]["episode_returns"]),
        )
        for s in SWEEP_SEEDS
    ]

    results = {
        "seed": REPORTED_SEED,
        "n_days": len(data["daily_df"]),
        "n_train_windows": int(data["X_train"].shape[0]),
        "n_val_windows": int(data["X_val"].shape[0]),
        "lstm_train_losses": lstm["train_losses"],
        "lstm_val_losses": lstm["val_losses"],
        "rf_returns": rf["episode_returns"],
        "a2c_returns": a2c["episode_returns"],
        "rf_std": std_pair(rf["episode_returns"]),
        "a2c_std": std_pair(a2c["episode_returns"]),
        "rf_dist": action_distribution(rf_actions, labels, n_actions),
        "a2c_dist": action_distribution(a2c_actions, labels, n_actions),
        "rf_distinct": len(set(rf_actions)),
        "a2c_distinct": len(set(a2c_actions)),
        "rf_balance": main_run["rf_balance"],
        "a2c_balance": main_run["a2c_balance"],
        "rf_conv": convergence_episode(rf["episode_returns"]),
        "a2c_conv": convergence_episode(a2c["episode_returns"]),
        "action_labels": labels,
        "sweep": sweep,
    }
    RESULTS_PKL.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PKL.open("wb") as fh:
        pickle.dump(results, fh)
    print(f"  wrote {RESULTS_PKL}")

    print("Rendering figures…")
    Plotter(sdk._cfg).save_all(lstm, rf, a2c, data)  # lstm_loss, returns, comparison
    plot_action_distribution(
        rf_actions, a2c_actions, labels, n_actions, PLOTS_DIR / "action_distribution.png"
    )
    plot_episode_trajectory(
        a2c_actions, a2c_states, labels, n_actions, PLOTS_DIR / "episode_trajectory.png"
    )
    plot_muscle_exposure(
        a2c_actions,
        data["action_muscle_profiles"],
        data["muscle_cols"],
        main_run["a2c_balance"],
        PLOTS_DIR / "muscle_exposure.png",
    )
    plot_state_analysis(a2c_states, PLOTS_DIR / "state_analysis.png")  # overwrite dataset version

    print(
        f"\nDone. REINFORCE balance={results['rf_balance']:.3f} "
        f"({results['rf_distinct']}/{n_actions} archetypes, final "
        f"{final_return(rf['episode_returns']):+.2f}); "
        f"A2C balance={results['a2c_balance']:.3f} "
        f"({results['a2c_distinct']}/{n_actions}, final "
        f"{final_return(a2c['episode_returns']):+.2f})."
    )


if __name__ == "__main__":
    main()
