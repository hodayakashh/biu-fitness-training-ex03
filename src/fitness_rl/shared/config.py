"""
ConfigManager — central reader for config/setup.json.

All hyperparameters and file paths come from here. No hardcoded values
in service code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ConfigManager:
    """
    Load and expose configuration from a versioned JSON file.

    Input:  path to setup.json
    Output: typed config values via .get()
    Setup:  instantiated once in FitnessRLSDK.__init__
    """

    def __init__(self, config_path: str | Path) -> None:
        """
        Load config from JSON file.

        Args:
            config_path: Path to setup.json (relative to project root).

        Raises:
            FileNotFoundError: If config file does not exist.
            ValueError: If file is not valid JSON.
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path.resolve()}")
        with path.open() as fh:
            self._data: dict[str, Any] = json.load(fh)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a top-level config key.

        Args:
            key: Top-level key in setup.json.
            default: Fallback if key is absent.

        Returns:
            Config value or default.
        """
        return self._data.get(key, default)

    def get_nested(self, *keys: str, default: Any = None) -> Any:
        """
        Retrieve a nested config value by a chain of keys.

        Args:
            *keys: Sequence of keys to traverse (e.g. "lstm", "hidden_size").
            default: Fallback if path is absent.

        Returns:
            Nested config value or default.
        """
        node = self._data
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node
