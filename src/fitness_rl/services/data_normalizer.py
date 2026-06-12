"""
DataNormalizer — MinMaxScaler fitted ONLY on the training split.

Prevents data leakage: val/test statistics must never influence the scaler
parameters used during LSTM training. The sin/cos features are pre-computed
and passed through unchanged (they are already in [-1, 1]).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from ..shared.config import ConfigManager

_CONTINUOUS = ["rolling_7day_load", "muscle_balance_score", "session_duration_avg"]
_SINUSOIDAL = ["day_sin", "day_cos"]


class DataNormalizer:
    """
    Scale continuous state features to [0, 1]; append sin/cos unchanged.

    Input:  daily summary DataFrames (train portion for fit; val/test for transform)
    Output: numpy arrays of shape (N, state_dim=5)
    Setup:  call fit_transform once on train; then transform on val — never the reverse
    """

    def __init__(self, cfg: ConfigManager) -> None:
        """Initialise with an unfitted MinMaxScaler."""
        self._cfg = cfg
        self._scaler = MinMaxScaler()
        self._fitted = False

    def fit_transform(self, train_df: pd.DataFrame) -> tuple[np.ndarray, MinMaxScaler]:
        """
        Fit scaler on training data and return scaled state matrix.

        Args:
            train_df: Training portion of daily summaries (80% of data).

        Returns:
            Tuple of (scaled array (N, 5), fitted MinMaxScaler for later reuse).
        """
        cont = self._get_continuous(train_df)
        scaled = self._scaler.fit_transform(cont)
        self._fitted = True
        return self._attach_sinusoidal(scaled, train_df), self._scaler

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Apply already-fitted scaler to validation or test data.

        Args:
            df: Val/test portion of daily summaries DataFrame.

        Returns:
            Scaled array of shape (N, 5).

        Raises:
            RuntimeError: If fit_transform has not been called first.
        """
        if not self._fitted:
            raise RuntimeError("Call fit_transform on training data before transform.")
        cont = self._get_continuous(df)
        scaled = self._scaler.transform(cont)
        return self._attach_sinusoidal(scaled, df)

    def _get_continuous(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and zero-fill the three continuous state feature columns."""
        arr = df.reindex(columns=_CONTINUOUS).fillna(0).values
        return arr.astype(np.float32)

    def _attach_sinusoidal(self, scaled: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        """Concatenate scaled continuous features with pre-computed sin/cos."""
        sin_cos = df.reindex(columns=_SINUSOIDAL).fillna(0).values.astype(np.float32)
        return np.concatenate([scaled, sin_cos], axis=1)
