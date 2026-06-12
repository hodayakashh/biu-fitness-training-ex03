"""Unit tests for data_features.py — rolling stats and sinusoidal encoding."""

import numpy as np
import pandas as pd
import pytest

from fitness_rl.constants import CYCLE_LENGTH
from fitness_rl.services.data_features import add_rolling_features, add_sinusoidal_encoding


@pytest.fixture()
def simple_daily_df() -> pd.DataFrame:
    """Minimal daily DataFrame with 14 days of data."""
    rng = np.random.default_rng(0)
    n = 14
    return pd.DataFrame(
        {
            "total_volume": rng.uniform(5000, 15000, n),
            "session_duration": rng.uniform(30, 90, n),
            "day_index": range(n),
            "day_in_cycle": [i % CYCLE_LENGTH for i in range(n)],
        }
    )


@pytest.fixture()
def daily_with_mg(simple_daily_df: pd.DataFrame) -> pd.DataFrame:
    """Daily DataFrame that also has muscle-group columns."""
    df = simple_daily_df.copy()
    rng = np.random.default_rng(1)
    raw = rng.uniform(0, 1, (len(df), 3))
    raw = raw / raw.sum(axis=1, keepdims=True)
    df["mg_chest"], df["mg_back"], df["mg_legs"] = raw[:, 0], raw[:, 1], raw[:, 2]
    return df


class TestAddRollingFeatures:
    def test_columns_added(self, simple_daily_df):
        result = add_rolling_features(simple_daily_df)
        assert "rolling_7day_load" in result.columns
        assert "session_duration_avg" in result.columns
        assert "muscle_balance_score" in result.columns

    def test_rolling_load_first_day(self, simple_daily_df):
        result = add_rolling_features(simple_daily_df)
        assert result["rolling_7day_load"].iloc[0] == pytest.approx(
            simple_daily_df["total_volume"].iloc[0], rel=1e-5
        )

    def test_rolling_load_7th_day(self, simple_daily_df):
        result = add_rolling_features(simple_daily_df)
        expected = simple_daily_df["total_volume"].iloc[:7].sum()
        assert result["rolling_7day_load"].iloc[6] == pytest.approx(expected, rel=1e-5)

    def test_no_nans_in_output(self, simple_daily_df):
        result = add_rolling_features(simple_daily_df)
        assert (
            not result[["rolling_7day_load", "session_duration_avg", "muscle_balance_score"]]
            .isna()
            .any()
            .any()
        )

    def test_muscle_balance_zero_without_mg_cols(self, simple_daily_df):
        result = add_rolling_features(simple_daily_df)
        assert (result["muscle_balance_score"] == 0.0).all()

    def test_muscle_balance_positive_with_mg_cols(self, daily_with_mg):
        result = add_rolling_features(daily_with_mg)
        assert (result["muscle_balance_score"] > 0).all()

    def test_input_not_modified(self, simple_daily_df):
        original_cols = set(simple_daily_df.columns)
        _ = add_rolling_features(simple_daily_df)
        assert set(simple_daily_df.columns) == original_cols


class TestAddSinusoidalEncoding:
    def test_columns_added(self, simple_daily_df):
        result = add_sinusoidal_encoding(simple_daily_df)
        assert "day_sin" in result.columns
        assert "day_cos" in result.columns

    def test_unit_circle_identity(self, simple_daily_df):
        result = add_sinusoidal_encoding(simple_daily_df)
        norms = result["day_sin"] ** 2 + result["day_cos"] ** 2
        np.testing.assert_allclose(norms.values, 1.0, atol=1e-6)

    def test_cyclicality_day0_equals_day28(self):
        df = pd.DataFrame({"day_in_cycle": [0, CYCLE_LENGTH]})
        result = add_sinusoidal_encoding(df)
        assert result["day_sin"].iloc[0] == pytest.approx(result["day_sin"].iloc[1], abs=1e-10)
        assert result["day_cos"].iloc[0] == pytest.approx(result["day_cos"].iloc[1], abs=1e-10)

    def test_input_not_modified(self, simple_daily_df):
        original_cols = set(simple_daily_df.columns)
        _ = add_sinusoidal_encoding(simple_daily_df)
        assert set(simple_daily_df.columns) == original_cols
