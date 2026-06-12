"""Unit tests for Plotter — figure generation and file I/O."""

import json

import numpy as np
import pandas as pd
import pytest

from fitness_rl.services.plotter import Plotter
from fitness_rl.shared.config import ConfigManager

N_EPISODES = 30


@pytest.fixture()
def cfg(tmp_path) -> ConfigManager:
    setup = {
        "version": "1.00",
        "paths": {"plots_dir": str(tmp_path / "plots")},
    }
    p = tmp_path / "setup.json"
    p.write_text(json.dumps(setup))
    return ConfigManager(p)


@pytest.fixture()
def lstm_result() -> dict:
    return {"train_losses": [0.5, 0.4, 0.3], "val_losses": [0.6, 0.5, 0.4]}


@pytest.fixture()
def reinforce_result() -> dict:
    return {
        "episode_returns": list(range(N_EPISODES)),
        "return_variance": [0.1] * N_EPISODES,
    }


@pytest.fixture()
def a2c_result() -> dict:
    return {"episode_returns": list(range(N_EPISODES))}


@pytest.fixture()
def data() -> dict:
    rng = np.random.default_rng(42)
    return {
        "daily_df": pd.DataFrame(
            {
                "rolling_7day_load": rng.random(50),
                "muscle_balance_score": rng.random(50),
                "action_label": rng.integers(0, 6, 50),
                "day_in_cycle": list(range(50)),
            }
        )
    }


class TestPlotter:
    def test_save_all_returns_five_paths(
        self, cfg, lstm_result, reinforce_result, a2c_result, data
    ):
        plotter = Plotter(cfg)
        paths = plotter.save_all(lstm_result, reinforce_result, a2c_result, data)
        assert len(paths) == 5

    def test_all_files_exist(
        self, cfg, lstm_result, reinforce_result, a2c_result, data
    ):
        plotter = Plotter(cfg)
        paths = plotter.save_all(lstm_result, reinforce_result, a2c_result, data)
        for p in paths:
            assert p.exists(), f"Plot file not found: {p}"

    def test_all_files_are_png(
        self, cfg, lstm_result, reinforce_result, a2c_result, data
    ):
        plotter = Plotter(cfg)
        paths = plotter.save_all(lstm_result, reinforce_result, a2c_result, data)
        assert all(p.suffix == ".png" for p in paths)

    def test_expected_filenames_present(
        self, cfg, lstm_result, reinforce_result, a2c_result, data
    ):
        plotter = Plotter(cfg)
        paths = plotter.save_all(lstm_result, reinforce_result, a2c_result, data)
        names = {p.name for p in paths}
        expected = {
            "lstm_loss.png", "reinforce_return.png", "a2c_return.png",
            "comparison.png", "state_analysis.png",
        }
        assert expected == names

    def test_handles_short_episode_list(self, cfg, lstm_result, data):
        """Rolling mean must not crash with very short episode lists."""
        short_rf = {"episode_returns": [1.0, 2.0], "return_variance": [0.1, 0.1]}
        short_a2c = {"episode_returns": [1.0, 2.0]}
        plotter = Plotter(cfg)
        paths = plotter.save_all(lstm_result, short_rf, short_a2c, data)
        assert len(paths) == 5

    def test_missing_rolling_load_column(self, cfg, lstm_result, reinforce_result, a2c_result):
        """State analysis should not crash when rolling_7day_load is absent."""
        df_no_load = pd.DataFrame({"other_col": range(10)})
        plotter = Plotter(cfg)
        paths = plotter.save_all(lstm_result, reinforce_result, a2c_result,
                                 {"daily_df": df_no_load})
        state_path = next(p for p in paths if p.name == "state_analysis.png")
        assert state_path.exists()
