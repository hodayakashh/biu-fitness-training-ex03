"""Unit tests for set_global_seed — reproducibility of numpy and torch RNGs."""

import numpy as np
import torch

from fitness_rl.shared.seeding import set_global_seed


def test_numpy_reproducible():
    set_global_seed(123)
    a = np.random.rand(5)
    set_global_seed(123)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


def test_torch_reproducible():
    set_global_seed(7)
    a = torch.randn(5)
    set_global_seed(7)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_different_seeds_differ():
    set_global_seed(1)
    a = np.random.rand(5)
    set_global_seed(2)
    b = np.random.rand(5)
    assert not np.array_equal(a, b)


def test_none_seed_is_noop():
    # Should not raise and should not reset the RNG.
    set_global_seed(None)
    a = np.random.rand(3)
    set_global_seed(None)
    b = np.random.rand(3)
    assert not np.array_equal(a, b)
