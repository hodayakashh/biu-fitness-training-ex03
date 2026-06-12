"""Unit tests for DataWindows — sliding window construction and leakage checks."""

import json

import numpy as np
import pytest
import torch

from fitness_rl.services.data_windows import DataWindows
from fitness_rl.shared.config import ConfigManager


@pytest.fixture()
def cfg(tmp_path) -> ConfigManager:
    p = tmp_path / "setup.json"
    p.write_text(json.dumps({
        "version": "1.00",
        "data": {"seq_len": 7, "n_actions": 6, "state_dim": 5},
    }))
    return ConfigManager(p)


@pytest.fixture()
def seq_data():
    """50-day synthetic state + action sequence."""
    rng = np.random.default_rng(99)
    states = rng.random((50, 5)).astype(np.float32)
    actions = rng.integers(0, 6, size=50)
    return states, actions


class TestDataWindows:
    def test_output_shapes(self, cfg, seq_data):
        states, actions = seq_data
        wins = DataWindows(cfg)
        seq_len = 7
        x_s, x_a, y = wins.build(states, actions, seq_len)
        n = len(states) - seq_len
        assert x_s.shape == (n, seq_len, 5)
        assert x_a.shape == (n, seq_len)
        assert y.shape == (n, 5)

    def test_output_dtypes(self, cfg, seq_data):
        states, actions = seq_data
        x_s, x_a, y = DataWindows(cfg).build(states, actions, seq_len=7)
        assert x_s.dtype == torch.float32
        assert x_a.dtype == torch.int64
        assert y.dtype == torch.float32

    def test_no_data_leakage(self, cfg, seq_data):
        """y[i] must equal states[i + seq_len] (strictly after the window)."""
        states, actions = seq_data
        seq_len = 7
        x_s, x_a, y = DataWindows(cfg).build(states, actions, seq_len)
        for i in range(len(y)):
            np.testing.assert_allclose(
                y[i].numpy(), states[i + seq_len], atol=1e-6,
                err_msg=f"Leakage detected at sample {i}"
            )

    def test_window_content_correct(self, cfg, seq_data):
        """X_states[i] must equal states[i : i+seq_len]."""
        states, actions = seq_data
        seq_len = 7
        x_s, x_a, y = DataWindows(cfg).build(states, actions, seq_len)
        np.testing.assert_allclose(x_s[0].numpy(), states[:seq_len], atol=1e-6)
        np.testing.assert_array_equal(x_a[0].numpy(), actions[:seq_len])

    def test_raises_on_short_sequence(self, cfg):
        states = np.random.rand(5, 5).astype(np.float32)
        actions = np.zeros(5, dtype=np.int64)
        with pytest.raises(ValueError, match="Too few samples"):
            DataWindows(cfg).build(states, actions, seq_len=7)

    def test_all_action_values_valid(self, cfg, seq_data):
        states, actions = seq_data
        x_s, x_a, y = DataWindows(cfg).build(states, actions, seq_len=7)
        assert x_a.min() >= 0
        assert x_a.max() < 6
