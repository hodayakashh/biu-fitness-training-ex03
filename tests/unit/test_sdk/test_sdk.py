"""Unit tests for FitnessRLSDK — the single entry point.

Heavy services (data pipeline, torch trainers, plotting) are mocked so these
tests exercise the SDK delegation logic without touching live dependencies.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from fitness_rl.sdk import FitnessRLSDK


@pytest.fixture()
def config_file(tmp_path):
    """Minimal valid setup.json with the version the code expects."""
    p = tmp_path / "setup.json"
    p.write_text(json.dumps({"version": "1.01", "data": {"seq_len": 7}}))
    return p


@pytest.fixture()
def sdk(config_file) -> FitnessRLSDK:
    return FitnessRLSDK(config_file)


class TestInit:
    def test_init_loads_config(self, sdk):
        """__init__ must load the config and expose it via _cfg."""
        assert sdk._cfg.get("version") == "1.01"

    def test_init_rejects_version_mismatch(self, tmp_path):
        """check_version must raise when config version != code version."""
        p = tmp_path / "bad.json"
        p.write_text(json.dumps({"version": "9.99"}))
        with pytest.raises(RuntimeError, match="does not match"):
            FitnessRLSDK(p)

    def test_init_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            FitnessRLSDK(tmp_path / "nope.json")


class TestDownloadDataset:
    def test_delegates_to_data_service(self, sdk):
        """download_dataset must build a DataService and call download_dataset."""
        svc = MagicMock()
        svc.download_dataset.return_value = "data/"
        with patch("fitness_rl.services.data_service.DataService", return_value=svc) as cls:
            result = sdk.download_dataset("data/")
        cls.assert_called_once_with(sdk._cfg)
        svc.download_dataset.assert_called_once_with("data/")
        assert result == "data/"


class TestPrepareData:
    def test_delegates_to_data_service(self, sdk):
        """prepare_data must build a DataService and call run_pipeline."""
        svc = MagicMock()
        svc.run_pipeline.return_value = {"daily_df": "ok"}
        with patch("fitness_rl.services.data_service.DataService", return_value=svc) as cls:
            result = sdk.prepare_data("data/workout.csv")
        cls.assert_called_once_with(sdk._cfg)
        svc.run_pipeline.assert_called_once_with("data/workout.csv")
        assert result == {"daily_df": "ok"}


class TestTrainLSTM:
    def test_delegates_to_lstm_trainer(self, sdk):
        """train_lstm must build an LSTMTrainer and call train(data)."""
        trainer = MagicMock()
        trainer.train.return_value = {"model": "m"}
        data = {"X_train": 1}
        with patch("fitness_rl.services.lstm_trainer.LSTMTrainer", return_value=trainer) as cls:
            result = sdk.train_lstm(data)
        cls.assert_called_once_with(sdk._cfg)
        trainer.train.assert_called_once_with(data)
        assert result == {"model": "m"}


class TestTrainReinforce:
    def test_delegates_to_reinforce_trainer(self, sdk):
        """train_reinforce must wire cfg, lstm_model and data into the trainer."""
        trainer = MagicMock()
        trainer.train.return_value = {"policy": "p"}
        model, data = MagicMock(), {"X_train": 1}
        with patch(
            "fitness_rl.services.reinforce_trainer.REINFORCETrainer",
            return_value=trainer,
        ) as cls:
            result = sdk.train_reinforce(model, data)
        cls.assert_called_once_with(sdk._cfg, model, data)
        trainer.train.assert_called_once_with()
        assert result == {"policy": "p"}


class TestTrainA2C:
    def test_delegates_to_a2c_trainer(self, sdk):
        """train_a2c must wire cfg, lstm_model and data into the trainer."""
        trainer = MagicMock()
        trainer.train.return_value = {"actor": "a", "critic": "c"}
        model, data = MagicMock(), {"X_train": 1}
        with patch("fitness_rl.services.a2c_trainer.A2CTrainer", return_value=trainer) as cls:
            result = sdk.train_a2c(model, data)
        cls.assert_called_once_with(sdk._cfg, model, data)
        trainer.train.assert_called_once_with()
        assert result == {"actor": "a", "critic": "c"}


class TestSaveAllPlots:
    def test_delegates_to_plotter(self, sdk):
        """save_all_plots must forward all four result/data args to Plotter."""
        plotter = MagicMock()
        plotter.save_all.return_value = ["a.png", "b.png"]
        lstm, rf, a2c, data = {"l": 1}, {"r": 2}, {"a": 3}, {"d": 4}
        with patch("fitness_rl.services.plotter.Plotter", return_value=plotter) as cls:
            result = sdk.save_all_plots(lstm, rf, a2c, data)
        cls.assert_called_once_with(sdk._cfg)
        plotter.save_all.assert_called_once_with(lstm, rf, a2c, data)
        assert result == ["a.png", "b.png"]
