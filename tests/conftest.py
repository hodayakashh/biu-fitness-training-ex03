"""Shared pytest fixtures for all test modules."""

import numpy as np
import pytest


@pytest.fixture()
def state_dim() -> int:
    """State vector dimensionality."""
    return 4


@pytest.fixture()
def n_actions() -> int:
    """Number of discrete workout actions."""
    return 6


@pytest.fixture()
def seq_len() -> int:
    """LSTM input sequence length (days)."""
    return 7


@pytest.fixture()
def batch_size() -> int:
    """Standard batch size for model tests."""
    return 8


@pytest.fixture()
def sample_state(state_dim) -> np.ndarray:
    """A single normalised state vector."""
    rng = np.random.default_rng(42)
    return rng.random(state_dim).astype(np.float32)


@pytest.fixture()
def sample_episode(state_dim, n_actions) -> dict:
    """A minimal synthetic 28-day episode."""
    rng = np.random.default_rng(0)
    return {
        "states": rng.random((28, state_dim)).astype(np.float32),
        "actions": rng.integers(0, n_actions, size=28),
        "rewards": rng.uniform(-1, 1, size=28).astype(np.float32),
    }
