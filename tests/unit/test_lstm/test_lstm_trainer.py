"""Unit tests for LSTMTrainer — training loop, loss curves, weight saving."""

import json

import numpy as np
import pytest
import torch

from fitness_rl.services.lstm_model import LSTMTransitionModel
from fitness_rl.services.lstm_trainer import LSTMTrainer
from fitness_rl.shared.config import ConfigManager


@pytest.fixture()
def cfg(tmp_path) -> ConfigManager:
    """Minimal config: tiny model, 3 epochs, small batch."""
    setup = {
        "version": "1.00",
        "data": {"seq_len": 5, "n_actions": 3, "state_dim": 5, "train_val_split": 0.8},
        "lstm": {
            "hidden_size": 8,
            "num_layers": 1,
            "dropout": 0.0,
            "action_embed_dim": 4,
            "learning_rate": 0.01,
            "batch_size": 8,
            "epochs": 3,
        },
        "paths": {"lstm_weights": str(tmp_path / "lstm_weights.pt")},
    }
    p = tmp_path / "setup.json"
    p.write_text(json.dumps(setup))
    return ConfigManager(p)


@pytest.fixture()
def fake_data() -> dict:
    """Synthetic 40-sample train / 10-sample val tensors."""
    torch.manual_seed(1)
    rng = np.random.default_rng(1)
    state_dim, seq_len, n_actions = 5, 5, 3

    def make_split(n):
        x_s = torch.from_numpy(rng.random((n, seq_len, state_dim)).astype(np.float32))
        x_a = torch.from_numpy(rng.integers(0, n_actions, (n, seq_len)))
        y = torch.from_numpy(rng.random((n, state_dim)).astype(np.float32))
        return x_s, x_a, y

    x_tr, xa_tr, y_tr = make_split(40)
    x_vl, xa_vl, y_vl = make_split(10)
    return {
        "X_train": x_tr,
        "X_actions_train": xa_tr,
        "y_train": y_tr,
        "X_val": x_vl,
        "X_actions_val": xa_vl,
        "y_val": y_vl,
    }


class TestLSTMTrainer:
    def test_train_returns_expected_keys(self, cfg, fake_data):
        result = LSTMTrainer(cfg).train(fake_data)
        assert set(result.keys()) == {"model", "train_losses", "val_losses"}

    def test_model_is_lstm_instance(self, cfg, fake_data):
        result = LSTMTrainer(cfg).train(fake_data)
        assert isinstance(result["model"], LSTMTransitionModel)

    def test_loss_lists_length_equals_epochs(self, cfg, fake_data):
        result = LSTMTrainer(cfg).train(fake_data)
        assert len(result["train_losses"]) == 3
        assert len(result["val_losses"]) == 3

    def test_no_nan_in_losses(self, cfg, fake_data):
        result = LSTMTrainer(cfg).train(fake_data)
        assert all(not np.isnan(v) for v in result["train_losses"])
        assert all(not np.isnan(v) for v in result["val_losses"])

    def test_losses_are_positive(self, cfg, fake_data):
        result = LSTMTrainer(cfg).train(fake_data)
        assert all(v > 0 for v in result["train_losses"])
        assert all(v > 0 for v in result["val_losses"])

    def test_weights_file_saved(self, cfg, fake_data, tmp_path):
        LSTMTrainer(cfg).train(fake_data)
        weights_path = tmp_path / "lstm_weights.pt"
        assert weights_path.exists(), "Model weights file not created"

    def test_saved_weights_loadable(self, cfg, fake_data, tmp_path):
        result = LSTMTrainer(cfg).train(fake_data)
        weights_path = tmp_path / "lstm_weights.pt"
        state_dict = torch.load(weights_path, weights_only=True)
        # Verify the saved dict can be loaded back into a fresh model
        fresh = result["model"].__class__(
            state_dim=5,
            n_actions=3,
            action_embed_dim=4,
            hidden_size=8,
            num_layers=1,
            dropout=0.0,
        )
        fresh.load_state_dict(state_dict)  # must not raise

    def test_model_in_eval_mode_after_training(self, cfg, fake_data):
        result = LSTMTrainer(cfg).train(fake_data)
        # After training the model should be callable in eval mode
        result["model"].eval()
        x_s = torch.randn(1, 5, 5)
        x_a = torch.randint(0, 3, (1, 5))
        with torch.no_grad():
            out = result["model"](x_s, x_a)
        assert out.shape == (1, 5)
