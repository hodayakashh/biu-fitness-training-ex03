"""Unit tests for DataNormalizer — train-only scaler fit and leakage prevention."""

import json

import numpy as np
import pandas as pd
import pytest

from fitness_rl.services.data_normalizer import DataNormalizer
from fitness_rl.shared.config import ConfigManager


@pytest.fixture()
def cfg(tmp_path) -> ConfigManager:
    p = tmp_path / "setup.json"
    p.write_text(json.dumps({"version": "1.00", "data": {"state_dim": 5}}))
    return ConfigManager(p)


@pytest.fixture()
def daily_df() -> pd.DataFrame:
    """Synthetic daily DataFrame with all required state columns."""
    rng = np.random.default_rng(7)
    n = 40
    return pd.DataFrame(
        {
            "rolling_7day_load": rng.uniform(1000, 50000, n),
            "muscle_balance_score": rng.uniform(0.5, 2.0, n),
            "session_duration_avg": rng.uniform(20, 90, n),
            "day_sin": np.sin(2 * np.pi * np.arange(n) / 28),
            "day_cos": np.cos(2 * np.pi * np.arange(n) / 28),
        }
    )


class TestDataNormalizer:
    def test_fit_transform_output_shape(self, cfg, daily_df):
        norm = DataNormalizer(cfg)
        result, _ = norm.fit_transform(daily_df)
        assert result.shape == (len(daily_df), 5)

    def test_continuous_features_in_unit_range(self, cfg, daily_df):
        norm = DataNormalizer(cfg)
        result, _ = norm.fit_transform(daily_df)
        # First 3 columns are continuous — should be scaled to [0, 1]
        assert result[:, :3].min() >= -1e-6
        assert result[:, :3].max() <= 1.0 + 1e-6

    def test_sinusoidal_passthrough(self, cfg, daily_df):
        norm = DataNormalizer(cfg)
        result, _ = norm.fit_transform(daily_df)
        expected_sin = daily_df["day_sin"].values.astype(np.float32)
        np.testing.assert_allclose(result[:, 3], expected_sin, atol=1e-5)

    def test_transform_raises_without_fit(self, cfg, daily_df):
        norm = DataNormalizer(cfg)
        with pytest.raises(RuntimeError, match="fit_transform"):
            norm.transform(daily_df)

    def test_transform_same_shape(self, cfg, daily_df):
        norm = DataNormalizer(cfg)
        train = daily_df.iloc[:30]
        val = daily_df.iloc[30:]
        norm.fit_transform(train)
        result = norm.transform(val)
        assert result.shape == (len(val), 5)

    def test_scaler_fitted_on_train_only(self, cfg, daily_df):
        """Val data outside train range should not be clipped by the scaler."""
        norm = DataNormalizer(cfg)
        train = daily_df.iloc[:20].copy()
        # Force val to have values larger than train range
        val = daily_df.iloc[20:].copy()
        val["rolling_7day_load"] = 1e9
        norm.fit_transform(train)
        val_scaled = norm.transform(val)
        # Values beyond [0,1] are expected — scaler was not re-fitted on val
        assert val_scaled[:, 0].max() > 1.0
