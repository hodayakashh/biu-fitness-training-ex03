"""
Stateless feature-engineering helpers for daily workout summaries.

Separated from DataPreprocessor so each file stays under 150 lines
and the helpers are independently testable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..constants import CYCLE_LENGTH


def add_rolling_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling_7day_load, session_duration_avg, and muscle_balance_score.

    Rolling windows use min_periods=1 so early days are not dropped.
    muscle_balance_score = Shannon entropy of the muscle-group distribution;
    higher values mean more balanced training across groups.

    Args:
        daily_df: Daily summaries with total_volume and session_duration.

    Returns:
        New DataFrame with three added columns.
    """
    df = daily_df.copy()
    df["rolling_7day_load"] = df["total_volume"].rolling(7, min_periods=1).sum()
    df["session_duration_avg"] = df["session_duration"].rolling(7, min_periods=1).mean()

    mg_cols = [c for c in df.columns if c.startswith("mg_")]
    if mg_cols:
        dist = df[mg_cols].values.clip(0) + 1e-8
        dist = dist / dist.sum(axis=1, keepdims=True)
        df["muscle_balance_score"] = -np.sum(dist * np.log(dist), axis=1)
    else:
        df["muscle_balance_score"] = 0.0

    return df


def add_sinusoidal_encoding(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode day_in_cycle as (sin, cos) to preserve cyclicality.

    Why sin/cos instead of t/27:
    A scalar fraction maps day 0 → 0.0 and day 27 → 1.0, making them
    numerically far apart even though they are adjacent in a repeating cycle.
    sin(2π·t/28) and cos(2π·t/28) place day 0 and day 28 at the same point
    on the unit circle, so the LSTM sees smooth angular continuity.

    Args:
        daily_df: DataFrame with day_in_cycle column (integer 0..CYCLE_LENGTH-1).

    Returns:
        New DataFrame with day_sin and day_cos columns appended.
    """
    df = daily_df.copy()
    t = df["day_in_cycle"].values
    df["day_sin"] = np.sin(2 * np.pi * t / CYCLE_LENGTH)
    df["day_cos"] = np.cos(2 * np.pi * t / CYCLE_LENGTH)
    return df
