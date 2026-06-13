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
            "lambda_repetition": 0.5,
            "variety_window": 4,
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
        reward = env._reward.compute(next_state)
        assert reward > 0.5, f"Expected reward > 0.5 at optimal load, got {reward}"

    def test_reward_penalises_overload(self, env):
        """High load (0.95) above threshold (0.8) → penalty reduces reward."""
        balanced = np.zeros(STATE_DIM, dtype=np.float32)
        balanced[0] = 0.5
        balanced[1] = 1.0
        r_balanced = env._reward.compute(balanced)

        overloaded = np.zeros(STATE_DIM, dtype=np.float32)
        overloaded[0] = 0.95
        overloaded[1] = 1.0
        r_overloaded = env._reward.compute(overloaded)

        assert r_overloaded < r_balanced

    def test_reward_penalises_imbalance(self, env):
        """Same load, poor muscle balance → lower reward than balanced."""
        balanced = np.zeros(STATE_DIM, dtype=np.float32)
        balanced[0] = 0.5
        balanced[1] = 1.0
        r_good = env._reward.compute(balanced)

        imbalanced = np.zeros(STATE_DIM, dtype=np.float32)
        imbalanced[0] = 0.5
        imbalanced[1] = 0.0
        r_bad = env._reward.compute(imbalanced)

        assert r_bad < r_good

    def test_reward_bounds_reasonable(self, env):
        """Reward should stay within [-2, 2] for normalised state values."""
        for _ in range(100):
            s = np.random.rand(STATE_DIM).astype(np.float32)
            r = env._reward.compute(s)
            assert -2.0 <= r <= 2.0


class TestRepetitionPenalty:
    """The variety penalty must read the agent's REAL action history."""

    def _state(self):
        s = np.zeros(STATE_DIM, dtype=np.float32)
        s[0] = 0.5  # optimal load
        s[1] = 1.0  # perfect balance
        return s

    def test_repeated_actions_penalised_more_than_varied(self, env):
        """Repeating one action over the window → lower reward than varied."""
        from collections import deque

        state = self._state()
        repeated = deque([1, 1, 1, 1], maxlen=4)
        varied = deque([0, 1, 2, 1], maxlen=4)

        r_repeated = env._reward.compute(state, repeated)
        r_varied = env._reward.compute(state, varied)

        assert r_repeated < r_varied

    def test_single_action_no_penalty(self, env):
        """A window with one element carries no info → zero penalty."""
        from collections import deque

        assert env._reward.repetition_penalty(deque([1], maxlen=4)) == 0.0
        assert env._reward.repetition_penalty(None) == 0.0

    def test_uniform_actions_zero_penalty(self, env):
        """A perfectly uniform window → maximal entropy → ~0 penalty."""
        from collections import deque

        penalty = env._reward.repetition_penalty(deque([0, 1, 2], maxlen=3))
        assert penalty == pytest.approx(0.0, abs=1e-6)

    def test_repetition_penalty_full_for_single_repeated_action(self, env):
        """All-same window → zero entropy → penalty 1.0."""
        from collections import deque

        assert env._reward.repetition_penalty(deque([2, 2, 2, 2], maxlen=4)) == pytest.approx(1.0)

    def test_step_records_real_actions(self, env):
        """step() should accumulate the agent's actual choices in the window."""
        env.reset()
        env.step(1)
        env.step(2)
        assert list(env._action_window) == [1, 2]


class TestConfigKeys:
    """New reward config keys are read, with sensible defaults when absent."""

    def test_reads_configured_keys(self, env):
        assert env._variety_window == 4
        assert env._reward_cfg.get("lambda_repetition") == 0.5

    def test_defaults_when_keys_absent(self, model, fake_data, tmp_path):
        """Omitting the new keys falls back to defaults (no crash, no hardcode)."""
        setup = {
            "version": "1.00",
            "data": {"seq_len": SEQ_LEN, "n_actions": N_ACTIONS, "state_dim": STATE_DIM},
            "rl": {"episode_length": EPISODE_LEN},
            "reward": {"lambda_overload": 0.4, "lambda_imbalance": 0.3},
            "lstm": {"hidden_size": 8, "num_layers": 1, "dropout": 0.0, "action_embed_dim": 4},
        }
        p = tmp_path / "setup_no_variety.json"
        p.write_text(json.dumps(setup))
        env_default = RLEnvironment(ConfigManager(p), model, fake_data)

        assert env_default._variety_window == 7  # default
        # the repetition penalty default is applied inside RewardFunction.compute
        from collections import deque

        state = np.zeros(STATE_DIM, dtype=np.float32)
        state[0] = 0.5
        state[1] = 1.0
        r_rep = env_default._reward.compute(state, deque([1, 1, 1], maxlen=3))
        r_var = env_default._reward.compute(state, deque([0, 1, 2], maxlen=3))
        assert r_rep < r_var  # default lambda_repetition is active
