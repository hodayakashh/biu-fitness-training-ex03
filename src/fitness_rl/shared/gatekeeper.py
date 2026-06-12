"""
ApiGatekeeper — centralised manager for all external API calls.

All Kaggle API calls (download, search) must go through this class.
Enforces rate limits from config/rate_limits.json; queues when limit
is reached rather than dropping or crashing.
"""

from __future__ import annotations

import json
import logging
import queue
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .version import check_version

logger = logging.getLogger(__name__)


class ApiGatekeeper:
    """
    Rate-limited, queued API call manager.

    Input:  callable API functions + their args/kwargs
    Output: API call results, in order
    Setup:  instantiated with rate_limits.json path; shared across services
    """

    def __init__(
        self,
        rate_config_path: str | Path = "config/rate_limits.json",
        service: str = "kaggle",
    ) -> None:
        """
        Load rate limit config and initialise the call queue.

        Why: limits are per-service — Kaggle is throttled harder than the
        generic default — so the gatekeeper must apply the caller's own
        service limits, falling back to ``default`` only when absent.

        Args:
            rate_config_path: Path to rate_limits.json.
            service: Service key in rate_limits.json whose limits to enforce.
        """
        path = Path(rate_config_path)
        with path.open() as fh:
            cfg = json.load(fh)
        rate_cfg = cfg["rate_limits"]
        check_version(rate_cfg.get("version"))
        services = rate_cfg["services"]
        svc_cfg = services.get(service, services.get("default", {}))
        self._rpm: int = svc_cfg.get("requests_per_minute", 30)
        self._max_retries: int = svc_cfg.get("max_retries", 3)
        self._retry_after: float = svc_cfg.get("retry_after_seconds", 30)
        self._call_times: list[float] = []
        self._pending: queue.Queue = queue.Queue()

    def execute(self, api_call: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Execute an API call through the gatekeeper.

        Checks rate limits before execution, queues if limit is reached,
        retries on transient failures, and logs all calls.

        Args:
            api_call: The callable to invoke.
            *args: Positional arguments for api_call.
            **kwargs: Keyword arguments for api_call.

        Returns:
            Result of api_call(*args, **kwargs).

        Raises:
            RuntimeError: If max retries are exhausted.
        """
        self._wait_for_capacity()
        for attempt in range(1, self._max_retries + 1):
            try:
                logger.info("Calling %s (attempt %d)", api_call.__name__, attempt)
                result = api_call(*args, **kwargs)
                self._call_times.append(time.monotonic())
                return result
            except Exception as exc:  # noqa: BLE001
                logger.warning("Attempt %d failed: %s", attempt, exc)
                if attempt < self._max_retries:
                    time.sleep(self._retry_after)
        raise RuntimeError(
            f"API call {api_call.__name__} failed after {self._max_retries} retries."
        )

    def get_queue_status(self) -> dict[str, int]:
        """Return current queue depth and recent call count."""
        now = time.monotonic()
        recent = sum(1 for t in self._call_times if now - t < 60)
        return {"pending": self._pending.qsize(), "calls_last_minute": recent}

    def _wait_for_capacity(self) -> None:
        """Block until the per-minute rate limit allows another call."""
        while True:
            now = time.monotonic()
            self._call_times = [t for t in self._call_times if now - t < 60]
            if len(self._call_times) < self._rpm:
                break
            time.sleep(1)
