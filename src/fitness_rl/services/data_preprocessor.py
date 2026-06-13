"""
DataPreprocessor — aggregate raw exercise records to daily summaries
and cluster days into discrete workout-type actions via K-Means.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from ..constants import CYCLE_LENGTH
from ..shared.config import ConfigManager
from .data_clusters import compute_action_profiles, describe_clusters


class DataPreprocessor:
    """
    Convert raw exercise-level CSV rows to daily workout summaries,
    then label each day with a K-Means action cluster.

    Input:  raw DataFrame (one row per exercise set)
    Output: daily DataFrame with state features and action_label column
    Setup:  ConfigManager with data.columns and data.n_actions keys
    """

    def __init__(self, cfg: ConfigManager) -> None:
        """Initialise from config; kmeans is None until cluster_actions is called."""
        self._cfg = cfg
        self.kmeans: KMeans | None = None
        self._cols: dict = cfg.get_nested("data", "columns") or {}

    def compute_daily_summaries(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate exercise rows into one row per workout day.

        Computed columns: total_volume, session_duration, mg_<group> per
        muscle group (normalised volume), day_index, day_in_cycle.

        Args:
            raw_df: Raw workout CSV loaded as a DataFrame.

        Returns:
            Daily summaries DataFrame sorted by date, index reset.
        """
        date_col = self._cols.get("date", "Date")
        sets_col = self._cols.get("sets", "Sets")
        reps_col = self._cols.get("reps", "Reps")
        weight_col = self._cols.get("weight", "Weight_kg")
        dur_col = self._cols.get("duration", "Duration_min")
        muscle_col = self._cols.get("muscle_group", "Muscle_Group")

        df = raw_df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        df["volume"] = self._compute_volume(df, sets_col, reps_col, weight_col)

        agg_dur = (dur_col, "sum") if dur_col in df.columns else ("volume", "count")
        daily = (
            df.groupby(date_col)
            .agg(total_volume=("volume", "sum"), session_duration=agg_dur)
            .reset_index()
            .sort_values(date_col)
            .reset_index(drop=True)
        )

        if muscle_col in df.columns:
            daily = self._add_muscle_cols(daily, df, date_col, muscle_col)

        daily["day_index"] = range(len(daily))
        daily["day_in_cycle"] = daily["day_index"] % CYCLE_LENGTH
        return daily

    def cluster_actions(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit K-Means on daily summaries and assign a discrete action label.

        Clustering features: total_volume, session_duration, muscle_balance_score
        (any subset that exists in the DataFrame is used).

        Args:
            daily_df: Daily summaries — must contain rolling features already.

        Returns:
            DataFrame with action_label column (int 0..n_actions-1).
        """
        n_clusters = self._cfg.get_nested("data", "n_actions")
        feature_cols = [
            c
            for c in ["total_volume", "session_duration", "muscle_balance_score"]
            if c in daily_df.columns
        ]
        features = daily_df[feature_cols].fillna(0).values
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df = daily_df.copy()
        df["action_label"] = self.kmeans.fit_predict(features)
        return df

    def action_profiles(self, daily_df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """
        Per-cluster muscle-group profiles (and the muscle column names).

        Thin delegation to ``data_clusters.compute_action_profiles`` so the heavy
        logic lives in a separate ≤150-line helper module. The profiles drive the
        action-conditioned muscle-balance dynamics of the RL environment (ADR-001).

        Args:
            daily_df: Daily summaries with action_label + mg_* columns.

        Returns:
            Tuple (profiles array (n_actions, n_mg), mg column names).
        """
        n_actions = self._cfg.get_nested("data", "n_actions")
        return compute_action_profiles(daily_df, n_actions)

    def describe_clusters(
        self, daily_df: pd.DataFrame, profiles: np.ndarray, mg_cols: list[str]
    ) -> dict[int, str]:
        """
        Data-driven, human-readable label per cluster (review finding #3).

        Thin delegation to ``data_clusters.describe_clusters``. Labels are grounded in
        each cluster's dominant muscle group + load tier rather than hardcoded against
        arbitrary K-Means IDs.

        Args:
            daily_df: Daily summaries with action_label + total_volume.
            profiles: Output of ``action_profiles``.
            mg_cols:  Muscle column names aligned to the profile columns.

        Returns:
            Mapping {cluster_id: label}.
        """
        n_actions = self._cfg.get_nested("data", "n_actions")
        return describe_clusters(daily_df, profiles, mg_cols, n_actions)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_volume(
        df: pd.DataFrame, sets_col: str, reps_col: str, weight_col: str
    ) -> pd.Series:
        """Compute per-row training volume = sets × reps × weight (weight optional)."""
        sets = df[sets_col].fillna(1) if sets_col in df.columns else 1
        reps = df[reps_col].fillna(1) if reps_col in df.columns else 1
        if weight_col in df.columns:
            return sets * reps * df[weight_col].fillna(1)
        return sets * reps

    @staticmethod
    def _add_muscle_cols(
        daily: pd.DataFrame, raw_df: pd.DataFrame, date_col: str, muscle_col: str
    ) -> pd.DataFrame:
        """Pivot normalised muscle-group volume per day and merge into daily."""
        mg = raw_df.groupby([date_col, muscle_col])["volume"].sum().unstack(fill_value=0)
        mg = mg.div(mg.sum(axis=1).replace(0, 1), axis=0)
        mg.columns = [f"mg_{c.lower()}" for c in mg.columns]
        return daily.merge(mg.reset_index(), on=date_col, how="left").fillna(0.0)
