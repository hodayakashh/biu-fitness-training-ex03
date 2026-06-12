"""Unit tests for RLEnvironment — reset, step, reward, and episode mechanics."""

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
EPISODE_LEN = 4


@pytest.fixture()
def cfg(tmp_path) -> ConfigManager:
    setup = {
        "version": "1.00",
        "data": {
            "seq_len": SEQ_LEN,
            "n_actions": N_ACTIONS,
            "state_dim": STATE_DIM,
            "train_val_split": 0.8,
        },
        "rl": {"episode_length": EPISODE_LEN},
        "reward": {
            "overload_threshold_norm": 0.8,
            "lambda_overload": 0.4,
            "lambda_imbalance": 0.3,
        },
        "lstm": {"hidden_size": 8, "num_layers": 1, "dropout": 0.0, "action_embed_dim": 4},
    }
    p = tmp_path / "setup.json"
    p.write_text(json.dumps(setup))
    return ConfigManager(p)


@pytest.fixture()
def model() -> LSTMTransitionModel:
    """Tiny untrained LSTM (functional but not meaningful predictions)."""
    torch.manual_seed(42)
    return LSTMTransitionModel(
        state_dim=STATE_DIM,
        n_actions=N_ACTIONS,
        action_embed_dim=4,
        hidden_size=8,
        num_layers=1,
        dropout=0.0,
    )


@pytest.fixture()
def fake_data() -> dict:
    """Synthetic data dict mimicking DataService output."""
    rng = np.random.default_rng(7)
    n = 20
    x_train = torch.from_numpy(rng.random((n, SEQ_LEN, STATE_DIM)).astype(np.float32))
    xa_train = torch.from_numpy(rng.integers(0, N_ACTIONS, (n, SEQ_LEN)))
    return {"X_train": x_train, "X_actions_train": xa_train}


@pytest.fixture()
def env(cfg, model, fake_data) -> RLEnvironment:
    return RLEnvironment(cfg, model, fake_data)


class TestStateDim:
    def test_state_dim_matches_data(self, env):
        """state_dim property must report the training window's feature width."""
        assert env.state_dim == STATE_DIM


class TestReset:
    def test_returns_state_array(self, env):
        state = env.reset()
        assert isinstance(state, np.ndarray)
        assert state.shape == (STATE_DIM,)

    def test_state_dtype_float(self, env):
        state = env.reset()
        assert state.dtype == np.float32

    def test_step_count_zero_after_reset(self, env):
        env.reset()
        assert env._step_count == 0

    def test_reset_gives_different_states(self, env):
        """Multiple resets should (almost always) yield different starting states."""
        states = [env.reset() for _ in range(10)]
        unique = {tuple(s.tolist()) for s in states}
        assert len(unique) > 1


class TestStep:
    def test_returns_three_values(self, env):
        env.reset()
        result = env.step(0)
        assert len(result) == 3

    def test_next_state_shape(self, env):
        env.reset()
        next_state, _, _ = env.step(1)
        assert next_state.shape == (STATE_DIM,)

    def test_reward_is_scalar_float(self, env):
        env.reset()
        _, reward, _ = env.step(0)
        assert isinstance(reward, float)

    def test_done_false_before_episode_end(self, env):
        env.reset()
        for _ in range(EPISODE_LEN - 1):
            _, _, done = env.step(0)
        assert not done

    def test_done_true_at_episode_end(self, env):
        env.reset()
        done = False
        for _ in range(EPISODE_LEN):
            _, _, done = env.step(0)
        assert done

    def test_full_episode_length(self, env):
        env.reset()
        steps = 0
        done = False
        while not done:
            _, _, done = env.step(steps % N_ACTIONS)
            steps += 1
        assert steps == EPISODE_LEN


class TestReward:
    def test_reward_optimal_load_balanced(self, env):
        """Load at optimal (0.5) with perfect balance → maximal positive reward."""
        next_state = np.zeros(STATE_DIM, dtype=np.float32)
        next_state[0] = 0.5  # rolling_load at optimal
        next_state[1] = 1.0  # muscle_balance (perfect)
        reward = env._compute_reward(next_state)
        assert reward > 0.5, f"Expected reward > 0.5 at optimal load, got {reward}"

    def test_reward_penalises_overload(self, env):
        """High load (0.95) above threshold (0.8) → penalty reduces reward."""
        balanced = np.zeros(STATE_DIM, dtype=np.float32)
        balanced[0] = 0.5
        balanced[1] = 1.0
        r_balanced = env._compute_reward(balanced)

        overloaded = np.zeros(STATE_DIM, dtype=np.float32)
        overloaded[0] = 0.95
        overloaded[1] = 1.0
        r_overloaded = env._compute_reward(overloaded)

        assert r_overloaded < r_balanced

    def test_reward_penalises_imbalance(self, env):
        """Same load, poor muscle balance → lower reward than balanced."""
        balanced = np.zeros(STATE_DIM, dtype=np.float32)
        balanced[0] = 0.5
        balanced[1] = 1.0
        r_good = env._compute_reward(balanced)

        imbalanced = np.zeros(STATE_DIM, dtype=np.float32)
        imbalanced[0] = 0.5
        imbalanced[1] = 0.0
        r_bad = env._compute_reward(imbalanced)

        assert r_bad < r_good

    def test_reward_bounds_reasonable(self, env):
        """Reward should stay within [-2, 2] for normalised state values."""
        for _ in range(100):
            s = np.random.rand(STATE_DIM).astype(np.float32)
            r = env._compute_reward(s)
            assert -2.0 <= r <= 2.0
