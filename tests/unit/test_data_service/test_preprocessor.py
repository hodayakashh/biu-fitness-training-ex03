"""Unit tests for DataPreprocessor — daily summaries and K-Means clustering."""

import numpy as np
import pandas as pd
import pytest

from fitness_rl.constants import CYCLE_LENGTH
from fitness_rl.services.data_preprocessor import DataPreprocessor
from fitness_rl.shared.config import ConfigManager


@pytest.fixture()
def cfg(tmp_path) -> ConfigManager:
    """Minimal config with column names and n_actions=3 for fast tests."""
    import json

    setup = {
        "version": "1.00",
        "data": {
            "n_actions": 3,
            "seq_len": 7,
            "state_dim": 5,
            "train_val_split": 0.8,
            "columns": {
                "date": "Date",
                "sets": "Sets",
                "reps": "Reps",
                "weight": "Weight_kg",
                "duration": "Duration_min",
                "muscle_group": "Muscle_Group",
            },
        },
    }
    p = tmp_path / "setup.json"
    p.write_text(json.dumps(setup))
    return ConfigManager(p)


@pytest.fixture()
def raw_df() -> pd.DataFrame:
    """Synthetic raw workout CSV with 30 sessions × 4 exercises each."""
    rng = np.random.default_rng(42)
    n_days, ex_per_day = 30, 4
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    records = []
    muscles = ["Chest", "Back", "Legs", "Shoulders"]
    for d in dates:
        for i in range(ex_per_day):
            records.append(
                {
                    "Date": d,
                    "Sets": int(rng.integers(2, 5)),
                    "Reps": int(rng.integers(6, 15)),
                    "Weight_kg": float(rng.uniform(20, 100)),
                    "Duration_min": float(rng.uniform(5, 15)),
                    "Muscle_Group": muscles[i % len(muscles)],
                }
            )
    return pd.DataFrame(records)


@pytest.fixture()
def prep(cfg) -> DataPreprocessor:
    return DataPreprocessor(cfg)


class TestComputeDailySummaries:
    def test_one_row_per_day(self, prep, raw_df):
        daily = prep.compute_daily_summaries(raw_df)
        assert len(daily) == raw_df["Date"].nunique()

    def test_total_volume_positive(self, prep, raw_df):
        daily = prep.compute_daily_summaries(raw_df)
        assert (daily["total_volume"] > 0).all()

    def test_muscle_cols_normalised(self, prep, raw_df):
        daily = prep.compute_daily_summaries(raw_df)
        mg_cols = [c for c in daily.columns if c.startswith("mg_")]
        assert len(mg_cols) > 0
        row_sums = daily[mg_cols].sum(axis=1)
        np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-5)

    def test_day_in_cycle_range(self, prep, raw_df):
        daily = prep.compute_daily_summaries(raw_df)
        assert daily["day_in_cycle"].between(0, CYCLE_LENGTH - 1).all()

    def test_missing_weight_fallback(self, prep, raw_df):
        df_no_weight = raw_df.drop(columns=["Weight_kg"])
        daily = prep.compute_daily_summaries(df_no_weight)
        assert (daily["total_volume"] > 0).all()

    def test_invalid_dates_dropped(self, prep):
        df = pd.DataFrame(
            {
                "Date": ["2024-01-01", "not-a-date", "2024-01-03"],
                "Sets": [3, 3, 3],
                "Reps": [10, 10, 10],
                "Weight_kg": [50.0, 50.0, 50.0],
                "Duration_min": [30.0, 30.0, 30.0],
                "Muscle_Group": ["Chest", "Back", "Legs"],
            }
        )
        daily = prep.compute_daily_summaries(df)
        assert len(daily) == 2  # "not-a-date" row dropped → 2 valid days


class TestClusterActions:
    def test_action_label_range(self, prep, raw_df):
        from fitness_rl.services.data_features import add_rolling_features, add_sinusoidal_encoding

        daily = prep.compute_daily_summaries(raw_df)
        daily = add_rolling_features(daily)
        daily = add_sinusoidal_encoding(daily)
        daily = prep.cluster_actions(daily)

        n_actions = 3
        assert daily["action_label"].between(0, n_actions - 1).all()

    def test_kmeans_stored(self, prep, raw_df):
        from fitness_rl.services.data_features import add_rolling_features, add_sinusoidal_encoding

        daily = prep.compute_daily_summaries(raw_df)
        daily = add_rolling_features(daily)
        daily = add_sinusoidal_encoding(daily)
        prep.cluster_actions(daily)
        assert prep.kmeans is not None
