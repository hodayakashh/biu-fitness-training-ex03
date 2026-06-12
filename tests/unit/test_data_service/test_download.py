"""Unit tests for DataService.download_dataset — Kaggle download via gatekeeper.

The Kaggle client and the gatekeeper are injected so these tests never hit the
network or require real credentials; they assert that the download is routed
through the gatekeeper with the config-driven dataset slug.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from fitness_rl.services.data_service import DataService
from fitness_rl.shared.config import ConfigManager


@pytest.fixture()
def cfg(tmp_path) -> ConfigManager:
    """Minimal config carrying the Kaggle dataset slug and data dir."""
    setup = {
        "version": "1.00",
        "data": {"kaggle_dataset": "owner/some-dataset"},
        "paths": {"data_dir": "data/"},
    }
    p = tmp_path / "setup.json"
    p.write_text(json.dumps(setup))
    return ConfigManager(p)


class TestDownloadDataset:
    def test_routes_download_through_gatekeeper(self, cfg, tmp_path):
        """The Kaggle download call must go through gatekeeper.execute, not direct."""
        gatekeeper = MagicMock()
        api = MagicMock()
        dest = tmp_path / "out"

        result = DataService(cfg).download_dataset(dest_dir=dest, gatekeeper=gatekeeper, api=api)

        gatekeeper.execute.assert_called_once_with(
            api.dataset_download_files,
            "owner/some-dataset",
            path=str(dest),
            unzip=True,
        )
        # The raw api download must NOT be invoked directly (only via gatekeeper).
        api.dataset_download_files.assert_not_called()
        assert result == dest
        assert dest.is_dir()

    def test_defaults_destination_to_config_data_dir(self, cfg, tmp_path, monkeypatch):
        """When no dest_dir is given, it falls back to config paths.data_dir."""
        monkeypatch.chdir(tmp_path)
        gatekeeper = MagicMock()
        api = MagicMock()

        result = DataService(cfg).download_dataset(gatekeeper=gatekeeper, api=api)

        assert result == Path("data/")
        assert (tmp_path / "data").is_dir()
