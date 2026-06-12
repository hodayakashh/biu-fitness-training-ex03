"""Unit tests for LSTMTransitionModel — shape, dtype, and gradient flow."""

import pytest
import torch

from fitness_rl.services.lstm_model import LSTMTransitionModel


@pytest.fixture()
def small_model() -> LSTMTransitionModel:
    """Tiny model (hidden=8) for fast unit tests."""
    return LSTMTransitionModel(
        state_dim=5,
        n_actions=6,
        action_embed_dim=4,
        hidden_size=8,
        num_layers=2,
        dropout=0.0,
    )


@pytest.fixture()
def sample_batch():
    """(batch=4, seq_len=7, state_dim=5) state + action tensors."""
    torch.manual_seed(0)
    x_s = torch.randn(4, 7, 5)
    x_a = torch.randint(0, 6, (4, 7))
    return x_s, x_a


class TestLSTMTransitionModel:
    def test_forward_output_shape(self, small_model, sample_batch):
        x_s, x_a = sample_batch
        out = small_model(x_s, x_a)
        assert out.shape == (4, 5), f"Expected (4,5), got {out.shape}"

    def test_forward_output_dtype(self, small_model, sample_batch):
        x_s, x_a = sample_batch
        out = small_model(x_s, x_a)
        assert out.dtype == torch.float32

    def test_batch_size_one(self, small_model):
        x_s = torch.randn(1, 7, 5)
        x_a = torch.randint(0, 6, (1, 7))
        out = small_model(x_s, x_a)
        assert out.shape == (1, 5)

    def test_different_seq_lengths(self, small_model):
        for seq_len in [3, 7, 14]:
            x_s = torch.randn(2, seq_len, 5)
            x_a = torch.randint(0, 6, (2, seq_len))
            out = small_model(x_s, x_a)
            assert out.shape == (2, 5)

    def test_gradients_flow_to_all_params(self, small_model, sample_batch):
        x_s, x_a = sample_batch
        out = small_model(x_s, x_a)
        loss = out.sum()
        loss.backward()
        for name, param in small_model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_deterministic_with_eval_mode(self, small_model, sample_batch):
        small_model.eval()
        x_s, x_a = sample_batch
        with torch.no_grad():
            out1 = small_model(x_s, x_a)
            out2 = small_model(x_s, x_a)
        assert torch.allclose(out1, out2)

    def test_single_layer_no_dropout(self):
        """Dropout is suppressed when num_layers=1 to avoid PyTorch warning."""
        model = LSTMTransitionModel(
            state_dim=5,
            n_actions=6,
            action_embed_dim=4,
            hidden_size=8,
            num_layers=1,
            dropout=0.5,
        )
        x_s = torch.randn(2, 5, 5)
        x_a = torch.randint(0, 6, (2, 5))
        out = model(x_s, x_a)
        assert out.shape == (2, 5)

    def test_action_index_out_of_range_raises(self, small_model):
        x_s = torch.randn(1, 7, 5)
        x_a = torch.tensor([[0, 1, 2, 3, 4, 5, 99]])  # index 99 > n_actions-1
        with pytest.raises(IndexError):
            small_model(x_s, x_a)
