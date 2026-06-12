"""
Integration test — full DataService pipeline on a synthetic CSV.

Covers DataService.load_raw(), run_pipeline(), and the SDK init path.
"""

import json

import numpy as np
import pandas as pd
import pytest

from fitness_rl.services.data_service import DataService
from fitness_rl.shared.config import ConfigManager


@pytest.fixture()
def cfg(tmp_path) -> ConfigManager:
    setup = {
        "version": "1.00",
        "data": {
            "seq_len": 5,
            "n_actions": 3,
            "state_dim": 5,
            "train_val_split": 0.8,
            "kaggle_dataset": "placeholder",
            "columns": {
                "date": "Date",
                "sets": "Sets",
                "reps": "Reps",
                "weight": "Weight_kg",
                "duration": "Duration_min",
                "muscle_group": "Muscle_Group",
            },
            "muscle_groups": ["Chest", "Back", "Legs"],
        },
        "lstm": {
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": 0.0,
            "action_embed_dim": 4,
            "learning_rate": 0.001,
            "batch_size": 8,
            "epochs": 2,
        },
        "rl": {},
        "reward": {},
        "paths": {},
    }
    p = tmp_path / "setup.json"
    p.write_text(json.dumps(setup))
    return ConfigManager(p)


@pytest.fixture()
def synthetic_csv(tmp_path) -> str:
    """50 workout days × 3 exercises each."""
    rng = np.random.default_rng(123)
    n_days, ex_per = 50, 3
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    muscles = ["Chest", "Back", "Legs"]
    rows = [
        {
            "Date": d,
            "Sets": int(rng.integers(2, 5)),
            "Reps": int(rng.integers(6, 12)),
            "Weight_kg": float(rng.uniform(20, 80)),
            "Duration_min": float(rng.uniform(5, 20)),
            "Muscle_Group": muscles[i % len(muscles)],
        }
        for d in dates
        for i in range(ex_per)
    ]
    path = tmp_path / "workout.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


class TestDataServiceIntegration:
    def test_load_raw_returns_dataframe(self, cfg, synthetic_csv):
        svc = DataService(cfg)
        df = svc.load_raw(synthetic_csv)
        assert len(df) == 150  # 50 days × 3 exercises

    def test_load_raw_raises_on_missing_file(self, cfg):
        svc = DataService(cfg)
        with pytest.raises(FileNotFoundError):
            svc.load_raw("/nonexistent/path/workout.csv")

    def test_run_pipeline_returns_expected_keys(self, cfg, synthetic_csv):
        result = DataService(cfg).run_pipeline(synthetic_csv)
        required = {
            "daily_df",
            "X_train",
            "X_actions_train",
            "y_train",
            "X_val",
            "X_actions_val",
            "y_val",
            "scaler",
            "kmeans",
        }
        assert required.issubset(result.keys())

    def test_run_pipeline_tensor_shapes(self, cfg, synthetic_csv):
        result = DataService(cfg).run_pipeline(synthetic_csv)
        seq_len, state_dim = 5, 5
        n_train = result["X_train"].shape[0]
        assert result["X_train"].shape == (n_train, seq_len, state_dim)
        assert result["X_actions_train"].shape == (n_train, seq_len)
        assert result["y_train"].shape == (n_train, state_dim)

    def test_run_pipeline_no_nan_in_tensors(self, cfg, synthetic_csv):
        result = DataService(cfg).run_pipeline(synthetic_csv)
        assert not result["X_train"].isnan().any()
        assert not result["y_train"].isnan().any()

    def test_run_pipeline_action_labels_valid(self, cfg, synthetic_csv):
        result = DataService(cfg).run_pipeline(synthetic_csv)
        xa = result["X_actions_train"]
        assert xa.min() >= 0
        assert xa.max() < 3  # n_actions=3
