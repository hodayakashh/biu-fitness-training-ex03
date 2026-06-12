"""
FitnessRLSDK — single entry point for all fitness-RL business logic.

All external consumers (notebooks, CLI, tests) MUST call methods here.
No business logic lives outside this SDK layer and the services it calls.
"""

from __future__ import annotations

from pathlib import Path

from ..shared.config import ConfigManager
from ..shared.version import check_version


class FitnessRLSDK:
    """
    Unified SDK for the fitness-RL pipeline.

    Input:  config_path pointing to config/setup.json
    Output: trained models, plots, and analysis results
    Setup:  call __init__ once; all methods use the shared config and gatekeeper
    """

    def __init__(self, config_path: str | Path = "config/setup.json") -> None:
        """
        Initialise the SDK, load config, and verify version compatibility.

        Args:
            config_path: Path to setup.json relative to the project root.
        """
        self._cfg = ConfigManager(config_path)
        check_version(self._cfg.get("version"))

    # ------------------------------------------------------------------
    # Data pipeline
    # ------------------------------------------------------------------

    def prepare_data(self, csv_path: str | Path) -> dict:
        """
        Load raw CSV, compute daily summaries, cluster actions.

        Args:
            csv_path: Path to the downloaded Kaggle workout CSV.

        Returns:
            dict with keys: daily_df, X_train, X_actions_train, y_train,
            X_val, X_actions_val, y_val, scaler, kmeans.
        """
        from ..services.data_service import DataService

        svc = DataService(self._cfg)
        return svc.run_pipeline(csv_path)

    # ------------------------------------------------------------------
    # LSTM transition model
    # ------------------------------------------------------------------

    def train_lstm(self, data: dict) -> dict:
        """
        Train the LSTM transition model on prepared data windows.

        Args:
            data: Output of prepare_data().

        Returns:
            dict with keys: model, train_losses, val_losses
        """
        from ..services.lstm_trainer import LSTMTrainer

        trainer = LSTMTrainer(self._cfg)
        return trainer.train(data)

    # ------------------------------------------------------------------
    # REINFORCE
    # ------------------------------------------------------------------

    def train_reinforce(self, lstm_model, data: dict) -> dict:
        """
        Train REINFORCE policy using LSTM as the world model.

        Args:
            lstm_model: Trained LSTMModel instance.
            data: Output of prepare_data() (for env initialisation).

        Returns:
            dict with keys: policy, episode_returns, return_variance
        """
        from ..services.reinforce_trainer import REINFORCETrainer

        trainer = REINFORCETrainer(self._cfg, lstm_model, data)
        return trainer.train()

    # ------------------------------------------------------------------
    # A2C
    # ------------------------------------------------------------------

    def train_a2c(self, lstm_model, data: dict) -> dict:
        """
        Train A2C (Actor-Critic) using LSTM as the world model.

        Args:
            lstm_model: Trained LSTMModel instance.
            data: Output of prepare_data() (for env initialisation).

        Returns:
            dict with keys: actor, critic, episode_returns
        """
        from ..services.a2c_trainer import A2CTrainer

        trainer = A2CTrainer(self._cfg, lstm_model, data)
        return trainer.train()

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def save_all_plots(
        self,
        lstm_result: dict,
        reinforce_result: dict,
        a2c_result: dict,
        data: dict,
    ) -> list[Path]:
        """
        Generate and save all required figures to results/plots/.

        Returns:
            List of saved file paths.
        """
        from ..services.plotter import Plotter

        plotter = Plotter(self._cfg)
        return plotter.save_all(lstm_result, reinforce_result, a2c_result, data)
