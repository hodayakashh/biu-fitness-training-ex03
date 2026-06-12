"""Unit tests for trainer __init__ wiring (A2C + REINFORCE).

These cover the constructor paths that other trainer tests bypass via __new__.
RLEnvironment is patched so no real LSTM/world-model is built.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from fitness_rl.services.a2c_trainer import A2CTrainer
from fitness_rl.services.reinforce_trainer import REINFORCETrainer
from fitness_rl.shared.config import ConfigManager


@pytest.fixture()
def cfg(tmp_path) -> ConfigManager:
    p = tmp_path / "setup.json"
    p.write_text(json.dumps({"version": "1.00", "rl": {"n_episodes": 1}}))
    return ConfigManager(p)


class TestA2CTrainerInit:
    def test_init_wires_env_and_cfg(self, cfg):
        """__init__ must store cfg and build an RLEnvironment from model+data."""
        model, data = MagicMock(), {"X_train": 1}
        fake_env = MagicMock()
        with patch(
            "fitness_rl.services.a2c_trainer.RLEnvironment", return_value=fake_env
        ) as env_cls:
            trainer = A2CTrainer(cfg, model, data)
        env_cls.assert_called_once_with(cfg, model, data)
        assert trainer._cfg is cfg
        assert trainer._env is fake_env


class TestReinforceTrainerInit:
    def test_init_wires_env_and_cfg(self, cfg):
        """__init__ must store cfg and build an RLEnvironment from model+data."""
        model, data = MagicMock(), {"X_train": 1}
        fake_env = MagicMock()
        with patch(
            "fitness_rl.services.reinforce_trainer.RLEnvironment",
            return_value=fake_env,
        ) as env_cls:
            trainer = REINFORCETrainer(cfg, model, data)
        env_cls.assert_called_once_with(cfg, model, data)
        assert trainer._cfg is cfg
        assert trainer._env is fake_env
