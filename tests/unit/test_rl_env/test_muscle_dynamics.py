"""
Unit tests for the action-conditioned muscle-balance dynamics (PLAN ADR-001).

These verify the core v1.01 fix: the agent's action choices causally drive the
muscle_balance state dimension (and therefore the imbalance penalty) through the
environment, independently of the LSTM's action-blind prediction.
"""

import json

import numpy as np
import pytest
import torch

from fitness_rl.services.lstm_model import LSTMTransitionModel
from fitness_rl.services.rl_env import RLEnvironment
from fitness_rl.shared.config import ConfigManager

STATE_DIM = 5
SEQ_LEN = 5
N_ACTIONS = 3
EPISODE_LEN = 6


@pytest.fixture()
def cfg(tmp_path) -> ConfigManager:
    setup = {
        "version": "1.01",
        "data": {"seq_len": SEQ_LEN, "n_actions": N_ACTIONS, "state_dim": STATE_DIM},
        "rl": {"episode_length": EPISODE_LEN},
        "reward": {"lambda_imbalance": 0.5, "lambda_repetition": 0.5, "variety_window": 5},
        "lstm": {"hidden_size": 8, "num_layers": 1, "dropout": 0.0, "action_embed_dim": 4},
    }
    p = tmp_path / "setup.json"
    p.write_text(json.dumps(setup))
    return ConfigManager(p)


@pytest.fixture()
def model() -> LSTMTransitionModel:
    torch.manual_seed(0)
    return LSTMTransitionModel(
        state_dim=STATE_DIM,
        n_actions=N_ACTIONS,
        action_embed_dim=4,
        hidden_size=8,
        num_layers=1,
        dropout=0.0,
    )


@pytest.fixture()
def data_with_profiles() -> dict:
    """One-hot muscle profiles: each action targets a distinct muscle group."""
    rng = np.random.default_rng(1)
    n = 12
    return {
        "X_train": torch.from_numpy(rng.random((n, SEQ_LEN, STATE_DIM)).astype(np.float32)),
        "X_actions_train": torch.from_numpy(rng.integers(0, N_ACTIONS, (n, SEQ_LEN))),
        "action_muscle_profiles": np.eye(N_ACTIONS, dtype=np.float32),
    }


@pytest.fixture()
def env(cfg, model, data_with_profiles) -> RLEnvironment:
    return RLEnvironment(cfg, model, data_with_profiles)


class TestActionConditionedBalance:
    def test_repeating_one_action_gives_zero_balance(self, env):
        """All-same one-hot profiles → concentrated exposure → entropy 0 → balance 0."""
        env.reset()
        last = None
        for _ in range(EPISODE_LEN):
            last, _r, _d = env.step(0)
        assert last[1] == pytest.approx(0.0, abs=1e-6)

    def test_varied_actions_give_high_balance(self, env):
        """Cycling all distinct archetypes → uniform mean profile → balance ≈ 1."""
        env.reset()
        last = None
        for t in range(EPISODE_LEN):
            last, _r, _d = env.step(t % N_ACTIONS)
        assert last[1] > 0.9

    def test_varied_beats_repeated_in_reward(self, env):
        """Action-aware imbalance penalty → varied policy earns more than repetition."""

        def mean_reward(actions):
            env.reset()
            rewards = [env.step(a)[1] for a in actions]
            return float(np.mean(rewards))

        repeated = mean_reward([0] * EPISODE_LEN)
        varied = mean_reward([t % N_ACTIONS for t in range(EPISODE_LEN)])
        assert varied > repeated

    def test_empty_window_balance_is_zero(self, env):
        """Helper returns 0 when no action has been taken yet (defensive branch)."""
        env.reset()
        assert env._muscle_balance_from_actions() == 0.0

    def test_profiles_loaded(self, env):
        assert env._profiles is not None
        assert env._profiles.shape == (N_ACTIONS, N_ACTIONS)


class TestOverrideDisabledWithoutProfiles:
    def test_no_profiles_keeps_lstm_prediction(self, cfg, model):
        """Without profiles the env must not touch muscle_balance (backward compat)."""
        rng = np.random.default_rng(2)
        data = {
            "X_train": torch.from_numpy(rng.random((6, SEQ_LEN, STATE_DIM)).astype(np.float32)),
            "X_actions_train": torch.from_numpy(rng.integers(0, N_ACTIONS, (6, SEQ_LEN))),
        }
        env = RLEnvironment(cfg, model, data)
        assert env._profiles is None
        env.reset()
        # Should run without error and leave muscle dynamics to the LSTM.
        next_state, _r, _d = env.step(1)
        assert next_state.shape == (STATE_DIM,)
