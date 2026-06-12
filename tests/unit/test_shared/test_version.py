"""Unit tests for version.py — version compatibility check."""

import pytest

from fitness_rl.shared.version import VERSION, check_version


def test_check_version_passes_with_matching_version():
    check_version(VERSION)  # should not raise


def test_check_version_raises_on_mismatch():
    with pytest.raises(RuntimeError, match="does not match"):
        check_version("9.99")


def test_version_constant_is_string():
    assert isinstance(VERSION, str)
    assert VERSION == "1.00"
