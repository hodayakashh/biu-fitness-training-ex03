"""
DataService — full data pipeline orchestrator.

Loads raw workout CSV → daily summaries → rolling features → sinusoidal
day encoding → K-Means action clusters → train/val split → normalisation
(fit on train only) → sliding-window tensors for LSTM training.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..shared.config import ConfigManager
from .data_features import add_rolling_features, add_sinusoidal_encoding
from .data_normalizer import DataNormalizer
from .data_preprocessor import DataPreprocessor
from .data_windows import DataWindows


class DataService:
    """
    Orchestrate the full data pipeline for the fitness RL environment.

    Input:  path to raw Kaggle workout CSV
    Output: dict with train/val tensors, scaler, kmeans — ready for LSTM + RL
    Setup:  ConfigManager containing data/*, lstm/* keys from setup.json
    """

    def __init__(self, cfg: ConfigManager) -> None:
        """Initialise sub-services; all share the same ConfigManager."""
        self._cfg = cfg
        self._prep = DataPreprocessor(cfg)
        self._norm = DataNormalizer(cfg)
        self._wins = DataWindows(cfg)

    def load_raw(self, csv_path: str | Path) -> pd.DataFrame:
        """
        Load raw workout CSV into a DataFrame.

        Args:
            csv_path: Path to the downloaded Kaggle CSV file.

        Returns:
            Raw DataFrame with all original columns intact.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path.resolve()}")
        return pd.read_csv(path)

    def run_pipeline(self, csv_path: str | Path) -> dict:
        """
        Execute the full data pipeline end-to-end.

        Steps (order matters — each step depends on the previous):
            1. Load CSV.
            2. Aggregate exercise rows to daily summaries.
            3. Add rolling features and muscle-balance entropy.
            4. Encode day_in_cycle as sin/cos.
            5. Cluster days into K discrete action types (K-Means).
            6. Temporal train/val split (no shuffle — order preserved).
            7. Fit MinMaxScaler on train split only; transform both.
            8. Build sliding-window tensors (X_states, X_actions, y).

        Args:
            csv_path: Path to raw Kaggle workout CSV.

        Returns:
            dict with keys: daily_df, X_train, X_actions_train, y_train,
            X_val, X_actions_val, y_val, scaler, kmeans.
        """
        raw_df = self.load_raw(csv_path)
        daily_df = self._prep.compute_daily_summaries(raw_df)
        daily_df = add_rolling_features(daily_df)
        daily_df = add_sinusoidal_encoding(daily_df)
        daily_df = self._prep.cluster_actions(daily_df)

        split_idx = int(len(daily_df) * self._cfg.get_nested("data", "train_val_split"))
        train_df = daily_df.iloc[:split_idx].copy()
        val_df = daily_df.iloc[split_idx:].copy()

        train_states, scaler = self._norm.fit_transform(train_df)
        val_states = self._norm.transform(val_df)

        seq_len: int = self._cfg.get_nested("data", "seq_len")
        x_tr, xa_tr, y_tr = self._wins.build(
            train_states, train_df["action_label"].values, seq_len
        )
        x_vl, xa_vl, y_vl = self._wins.build(
            val_states, val_df["action_label"].values, seq_len
        )

        return {
            "daily_df": daily_df,
            "X_train": x_tr,
            "X_actions_train": xa_tr,
            "y_train": y_tr,
            "X_val": x_vl,
            "X_actions_val": xa_vl,
            "y_val": y_vl,
            "scaler": scaler,
            "kmeans": self._prep.kmeans,
        }
