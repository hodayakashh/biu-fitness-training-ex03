"""Unit tests for ConfigManager — loading, get, get_nested."""

import json

import pytest

from fitness_rl.shared.config import ConfigManager


@pytest.fixture()
def config_file(tmp_path):
    data = {
        "version": "1.00",
        "data": {"seq_len": 7, "n_actions": 6},
        "lstm": {"hidden_size": 64},
    }
    p = tmp_path / "setup.json"
    p.write_text(json.dumps(data))
    return p


def test_load_valid_config(config_file):
    cfg = ConfigManager(config_file)
    assert cfg.get("version") == "1.00"


def test_get_missing_key_returns_default(config_file):
    cfg = ConfigManager(config_file)
    assert cfg.get("nonexistent", "fallback") == "fallback"


def test_get_nested_valid_path(config_file):
    cfg = ConfigManager(config_file)
    assert cfg.get_nested("data", "seq_len") == 7
    assert cfg.get_nested("lstm", "hidden_size") == 64


def test_get_nested_missing_returns_default(config_file):
    cfg = ConfigManager(config_file)
    assert cfg.get_nested("data", "missing_key", default=99) == 99
    assert cfg.get_nested("no_section", "key") is None


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ConfigManager(tmp_path / "nonexistent.json")
