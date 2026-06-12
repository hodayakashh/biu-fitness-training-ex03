"""Unit tests for A2CTrainer — per-step A2C updates, output structure, types."""

import json

import numpy as np
import pytest

from fitness_rl.services.a2c_trainer import A2CTrainer
from fitness_rl.services.policy_network import CriticNetwork, PolicyNetwork
from fitness_rl.shared.config import ConfigManager

STATE_DIM = 5
N_ACTIONS = 3
EPISODE_LEN = 6


class _MockEnv:
    """Minimal deterministic environment for testing trainers without LSTM."""

    def __init__(self) -> None:
        self._step = 0
        self._rng = np.random.default_rng(42)

    @property
    def state_dim(self) -> int:
        return STATE_DIM

    def reset(self) -> np.ndarray:
        self._step = 0
        return self._rng.random(STATE_DIM).astype(np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        self._step += 1
        next_s = self._rng.random(STATE_DIM).astype(np.float32)
        reward = float(self._rng.uniform(-0.5, 1.0))
        return next_s, reward, self._step >= EPISODE_LEN


@pytest.fixture()
def cfg(tmp_path) -> ConfigManager:
    setup = {
        "version": "1.00",
        "data": {"seq_len": 5, "n_actions": N_ACTIONS, "state_dim": STATE_DIM},
        "rl": {
            "episode_length": EPISODE_LEN,
            "n_episodes": 5,
            "gamma": 0.99,
            "actor_lr": 0.0003,
            "critic_lr": 0.001,
            "value_loss_coeff": 0.5,
            "entropy_bonus": 0.01,
        },
        "reward": {
            "overload_threshold_norm": 0.8,
            "optimal_load_norm": 0.5,
            "lambda_overload": 0.4,
            "lambda_imbalance": 0.3,
        },
    }
    p = tmp_path / "setup.json"
    p.write_text(json.dumps(setup))
    return ConfigManager(p)


def _make_trainer(cfg: ConfigManager) -> A2CTrainer:
    """Create an A2CTrainer bypassing __init__ (no real LSTM needed)."""
    trainer = A2CTrainer.__new__(A2CTrainer)
    trainer._cfg = cfg
    trainer._env = _MockEnv()
    return trainer


class TestA2CTrainerOutputStructure:
    def test_train_returns_expected_keys(self, cfg):
        """Result dict must contain actor, critic, and episode_returns."""
        trainer = _make_trainer(cfg)
        result = trainer.train()
        assert {"actor", "critic", "episode_returns"}.issubset(result.keys())

    def test_episode_returns_length(self, cfg):
        """episode_returns list length must equal n_episodes."""
        trainer = _make_trainer(cfg)
        result = trainer.train()
        assert len(result["episode_returns"]) == 5  # n_episodes=5

    def test_actor_is_policy_network(self, cfg):
        """Returned actor must be a PolicyNetwork instance."""
        trainer = _make_trainer(cfg)
        result = trainer.train()
        assert isinstance(result["actor"], PolicyNetwork)

    def test_critic_is_critic_network(self, cfg):
        """Returned critic must be a CriticNetwork instance."""
        trainer = _make_trainer(cfg)
        result = trainer.train()
        assert isinstance(result["critic"], CriticNetwork)

    def test_episode_returns_are_finite_floats(self, cfg):
        """Every episode return must be a finite float (no NaN / Inf)."""
        trainer = _make_trainer(cfg)
        result = trainer.train()
        for ep_ret in result["episode_returns"]:
            assert isinstance(ep_ret, float)
            assert np.isfinite(ep_ret), f"Non-finite return: {ep_ret}"
