"""
episode_analysis.py — Empirical episode analysis for the BIU DRL Ex03 report.

Trains LSTM + REINFORCE + A2C (N_EPISODES each) and runs a
28-day greedy episode for each policy. Saves three plots to results/plots/:

    action_distribution.png  — side-by-side bar chart of action counts
    episode_trajectory.png   — A2C daily actions + rolling_load over 28 days
    state_analysis.png       — A2C rolling_load for the 28-day episode (replaces
                               the raw-dataset version)

Usage:
    cd /path/to/biu-drl-ex03
    uv run python docs/episode_analysis.py
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from fitness_rl import FitnessRLSDK
from fitness_rl.constants import ACTION_LABELS, N_ACTIONS
from fitness_rl.services.rl_env import RLEnvironment

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PLOTS_DIR = Path("results/plots")
DATA_CSV = Path("data/Hevy_workouts_log_100_weeks.csv")
CONFIG_PATH = Path("config/setup.json")

REINFORCE_COLOR = "tab:blue"
A2C_COLOR = "tab:orange"

# Number of RL training episodes for both REINFORCE and A2C.
# Lower this (e.g. to 20) for a fast smoke test of the plotting pipeline.
N_EPISODES = 500

# Optimal normalised rolling load the agent is steered toward.
OPTIMAL_LOAD = 0.5

plt.style.use("seaborn-v0_8-darkgrid")


def _build_sdk(n_episodes: int = N_EPISODES) -> FitnessRLSDK:
    """Return an SDK whose RL config uses *n_episodes* for fast training."""
    with CONFIG_PATH.open() as fh:
        cfg_data = json.load(fh)
    cfg_data["rl"]["n_episodes"] = n_episodes

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(cfg_data, tmp)
        tmp_name = tmp.name
    sdk = FitnessRLSDK(tmp_name)
    os.unlink(tmp_name)
    return sdk


def _run_episode_greedy(policy_net, env: RLEnvironment) -> tuple[list[int], list[np.ndarray]]:
    """
    Execute one 28-day greedy episode.

    Returns:
        actions: list of integer actions (length = episode_length).
        states:  list of state arrays *after* each step (length = episode_length).
    """
    state = env.reset()
    actions: list[int] = []
    states: list[np.ndarray] = []
    done = False

    while not done:
        state_t = torch.from_numpy(state).float()
        with torch.no_grad():
            dist = policy_net(state_t)
            action = int(dist.probs.argmax().item())  # greedy
        state, _reward, done = env.step(action)
        actions.append(action)
        states.append(state.copy())

    return actions, states


# ---------------------------------------------------------------------------
# Plot 1: action_distribution.png
# ---------------------------------------------------------------------------


def plot_action_distribution(
    rf_actions: list[int],
    a2c_actions: list[int],
    out_path: Path,
) -> None:
    """Side-by-side bar chart of REINFORCE vs A2C action counts."""
    labels = [ACTION_LABELS[i] for i in range(N_ACTIONS)]
    x = np.arange(N_ACTIONS)
    width = 0.35

    rf_counts = [rf_actions.count(i) for i in range(N_ACTIONS)]
    a2c_counts = [a2c_actions.count(i) for i in range(N_ACTIONS)]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_rf = ax.bar(x - width / 2, rf_counts, width, label="REINFORCE", color=REINFORCE_COLOR)
    bars_a2c = ax.bar(x + width / 2, a2c_counts, width, label="A2C", color=A2C_COLOR)

    # Label each bar with its count
    for bar in bars_rf:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.1,
            str(int(h)),
            ha="center",
            va="bottom",
            fontsize=9,
            color=REINFORCE_COLOR,
        )
    for bar in bars_a2c:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.1,
            str(int(h)),
            ha="center",
            va="bottom",
            fontsize=9,
            color=A2C_COLOR,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Count (days in 28-day episode)")
    ax.set_title("Action Distribution: REINFORCE vs A2C (28-day episode, greedy)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: episode_trajectory.png
# ---------------------------------------------------------------------------


def plot_episode_trajectory(
    a2c_actions: list[int],
    a2c_states: list[np.ndarray],
    out_path: Path,
) -> None:
    """A2C episode: top=action scatter, bottom=rolling_load line."""
    days = list(range(1, len(a2c_actions) + 1))
    rolling_load = [s[0] for s in a2c_states]

    # Colour each scatter point by action type
    action_colors = plt.cm.tab10(np.linspace(0, 0.6, N_ACTIONS))
    point_colors = [action_colors[a] for a in a2c_actions]

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    # Top: action scatter
    ax_top.scatter(days, a2c_actions, c=point_colors, s=80, zorder=3)
    ax_top.vlines(days, 0, a2c_actions, colors="grey", linewidth=0.5, alpha=0.5)
    ax_top.set_yticks(list(range(N_ACTIONS)))
    ax_top.set_yticklabels([ACTION_LABELS[i] for i in range(N_ACTIONS)], fontsize=8)
    ax_top.set_ylabel("Action")
    ax_top.set_title("A2C Policy — Generated Episode: Actions over 28 Days")

    # Bottom: rolling_load line with the optimal-load reference
    ax_bot.plot(days, rolling_load, color=A2C_COLOR, linewidth=2, marker="o", markersize=4)
    ax_bot.axhline(
        OPTIMAL_LOAD,
        linestyle="--",
        color="grey",
        linewidth=1,
        label=f"Optimal load ({OPTIMAL_LOAD})",
    )
    ax_bot.legend(loc="best", fontsize=8)
    ax_bot.set_xlabel("Day in episode")
    ax_bot.set_ylabel("Normalised Rolling Load (state[0])")
    ax_bot.set_title("A2C Policy — Generated Episode: Rolling Load over 28 Days")
    ax_bot.set_xlim(0.5, len(days) + 0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 3: state_analysis.png  (replaces raw-dataset version)
# ---------------------------------------------------------------------------


def plot_state_analysis(
    a2c_states: list[np.ndarray],
    out_path: Path,
) -> None:
    """Single line plot of rolling_load from the A2C-generated episode."""
    days = list(range(1, len(a2c_states) + 1))
    rolling_load = [s[0] for s in a2c_states]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(days, rolling_load, color=A2C_COLOR, linewidth=2, marker="o", markersize=5)
    ax.set_xlabel("Day in episode")
    ax.set_ylabel("Normalised Rolling Load (state[0])")
    ax.set_title("A2C Policy — Generated Episode: Rolling Load over 28 Days")
    ax.set_xlim(0.5, len(days) + 0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Train LSTM + REINFORCE + A2C, run greedy episodes, save three plots."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Building SDK (n_episodes={N_EPISODES})…")
    sdk = _build_sdk(n_episodes=N_EPISODES)

    print("Preparing data…")
    data = sdk.prepare_data(DATA_CSV)

    print("Training LSTM (50 epochs)…")
    lstm_r = sdk.train_lstm(data)
    lstm_model = lstm_r["model"]

    print(f"Training REINFORCE ({N_EPISODES} episodes)…")
    rf_r = sdk.train_reinforce(lstm_model, data)

    print(f"Training A2C ({N_EPISODES} episodes)…")
    a2c_r = sdk.train_a2c(lstm_model, data)

    # Create environments for episode rollout
    cfg = sdk._cfg
    env_rf = RLEnvironment(cfg, lstm_model, data)
    env_a2c = RLEnvironment(cfg, lstm_model, data)

    print("Running greedy episodes…")
    rf_actions, _rf_states = _run_episode_greedy(rf_r["policy"], env_rf)
    a2c_actions, a2c_states = _run_episode_greedy(a2c_r["actor"], env_a2c)

    rf_counts = [rf_actions.count(i) for i in range(N_ACTIONS)]
    a2c_counts = [a2c_actions.count(i) for i in range(N_ACTIONS)]
    print(f"\nREINFORCE action counts: {dict(zip(ACTION_LABELS.values(), rf_counts, strict=True))}")
    print(f"A2C action counts:       {dict(zip(ACTION_LABELS.values(), a2c_counts, strict=True))}")

    print("\nSaving plots…")
    plot_action_distribution(rf_actions, a2c_actions, PLOTS_DIR / "action_distribution.png")
    plot_episode_trajectory(a2c_actions, a2c_states, PLOTS_DIR / "episode_trajectory.png")
    plot_state_analysis(a2c_states, PLOTS_DIR / "state_analysis.png")

    print("\nDone. All 3 plots saved to results/plots/")


if __name__ == "__main__":
    main()
