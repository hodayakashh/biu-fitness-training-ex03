"""
Plotter — generate and save all 5 required assignment figures.

Figures (saved to config paths.plots_dir):
  1. lstm_loss.png       — train/val MSE loss vs epoch
  2. reinforce_return.png — REINFORCE episodic return + rolling mean
  3. a2c_return.png       — A2C episodic return + rolling mean
  4. comparison.png       — REINFORCE vs A2C on the same axes
  5. state_analysis.png   — rolling_7day_load over time from dataset
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..shared.config import ConfigManager

plt.style.use("seaborn-v0_8-darkgrid")
_ROLL_WIN = 20


class Plotter:
    """
    Save all 5 required figures from trainer result dicts.

    Input:  result dicts from LSTMTrainer, REINFORCETrainer, A2CTrainer, DataService
    Output: list[Path] of saved PNG files
    Setup:  ConfigManager with paths.plots_dir key
    """

    def __init__(self, cfg: ConfigManager) -> None:
        """Read plots directory from config."""
        plots_dir = (cfg.get("paths") or {}).get("plots_dir", "results/plots/")
        self._plots_dir = Path(plots_dir)
        self._plots_dir.mkdir(parents=True, exist_ok=True)

    def save_all(
        self,
        lstm_result: dict,
        reinforce_result: dict,
        a2c_result: dict,
        data: dict,
    ) -> list[Path]:
        """
        Generate and save all 5 required figures.

        Args:
            lstm_result:      {"train_losses": list, "val_losses": list}
            reinforce_result: {"episode_returns": list, "return_variance": list}
            a2c_result:       {"episode_returns": list}
            data:             {"daily_df": pd.DataFrame}

        Returns:
            List of 5 saved Path objects.
        """
        paths = [
            self._lstm_loss(lstm_result),
            self._returns_curve(
                reinforce_result["episode_returns"], "REINFORCE", "tab:blue", "reinforce_return.png"
            ),
            self._returns_curve(
                a2c_result["episode_returns"], "A2C", "tab:orange", "a2c_return.png"
            ),
            self._comparison(reinforce_result["episode_returns"], a2c_result["episode_returns"]),
            self._state_analysis(data["daily_df"]),
        ]
        return paths

    # ------------------------------------------------------------------
    # Private helpers — one method per figure
    # ------------------------------------------------------------------

    def _lstm_loss(self, lstm_result: dict) -> Path:
        """Plot LSTM train and validation MSE loss vs epoch."""
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(lstm_result["train_losses"], label="Train loss", color="tab:blue")
        ax.plot(lstm_result["val_losses"], label="Val loss", color="tab:orange", linestyle="--")
        ax.set_title("LSTM Transition Model — Training Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend()
        return self._save(fig, "lstm_loss.png")

    def _returns_curve(self, returns: list[float], label: str, color: str, fname: str) -> Path:
        """Plot episodic returns with a rolling mean overlay."""
        fig, ax = plt.subplots(figsize=(8, 4))
        eps = list(range(1, len(returns) + 1))
        ax.plot(eps, returns, alpha=0.35, color=color, label="Raw return")
        if len(returns) >= 2:
            win = min(_ROLL_WIN, max(2, len(returns) // 5))
            rolling = pd.Series(returns).rolling(win, min_periods=1).mean()
            ax.plot(eps, rolling, color=color, linewidth=2, label=f"Rolling mean (w={win})")
        ax.set_title(f"{label} — Training Curve")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Return")
        ax.legend()
        return self._save(fig, fname)

    def _comparison(self, rf_returns: list[float], a2c_returns: list[float]) -> Path:
        """Overlay REINFORCE and A2C rolling means on the same axes."""
        fig, ax = plt.subplots(figsize=(8, 4))
        for returns, label, color in [
            (rf_returns, "REINFORCE", "tab:blue"),
            (a2c_returns, "A2C", "tab:orange"),
        ]:
            if len(returns) < 2:
                continue
            win = min(_ROLL_WIN, max(2, len(returns) // 5))
            rolling = pd.Series(returns).rolling(win, min_periods=1).mean()
            ax.plot(rolling, label=f"{label} (rolling mean)", color=color, linewidth=2)
        ax.set_title("REINFORCE vs A2C — Episodic Return Comparison")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Return")
        ax.legend()
        return self._save(fig, "comparison.png")

    def _state_analysis(self, daily_df: pd.DataFrame) -> Path:
        """Plot rolling_7day_load over training days from the dataset."""
        fig, ax = plt.subplots(figsize=(8, 4))
        if "rolling_7day_load" in daily_df.columns:
            ax.plot(daily_df["rolling_7day_load"].values, color="tab:green", linewidth=1.5)
        ax.set_title("Trainee State — Rolling 7-Day Load Over Time")
        ax.set_xlabel("Day")
        ax.set_ylabel("Normalised Rolling Load")
        return self._save(fig, "state_analysis.png")

    def _save(self, fig: plt.Figure, fname: str) -> Path:
        """Tight-layout, save to plots_dir, close figure, return Path."""
        path = self._plots_dir / fname
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
