"""Version tracking — code must match config version "1.00"."""

VERSION = "1.00"


def check_version(config_version: str) -> None:
    """
    Raise if config version does not match code version.

    Why: Prevents silent drift between config and code expectations.

    Args:
        config_version: The "version" field read from setup.json.

    Raises:
        RuntimeError: If versions differ.
    """
    if config_version != VERSION:
        raise RuntimeError(
            f"Config version '{config_version}' does not match code version '{VERSION}'. "
            "Update config/setup.json or src/fitness_rl/shared/version.py."
        )
