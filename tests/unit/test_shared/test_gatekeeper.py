"""Unit tests for ApiGatekeeper — execute, retry, and queue status."""

import json
from unittest.mock import patch

import pytest

from fitness_rl.shared.gatekeeper import ApiGatekeeper


@pytest.fixture()
def rate_config(tmp_path) -> str:
    """Rate-limits config with high RPM so tests never block on capacity."""
    config = {
        "rate_limits": {
            "version": "1.01",
            "services": {
                "default": {
                    "requests_per_minute": 100,
                    "requests_per_hour": 1000,
                    "concurrent_max": 10,
                    "retry_after_seconds": 0,
                    "max_retries": 3,
                }
            },
        }
    }
    p = tmp_path / "rate_limits.json"
    p.write_text(json.dumps(config))
    return str(p)


def _identity(value):
    """Return value unchanged — used as a simple API stub."""
    return value


def _add(x, y):
    """Return x + y — tests arg/kwarg forwarding."""
    return x + y


class TestApiGatekeeper:
    def test_execute_returns_result(self, rate_config):
        gk = ApiGatekeeper(rate_config)
        assert gk.execute(_identity, 42) == 42

    def test_execute_passes_args_and_kwargs(self, rate_config):
        gk = ApiGatekeeper(rate_config)
        assert gk.execute(_add, 3, 4) == 7

    def test_get_queue_status_keys(self, rate_config):
        gk = ApiGatekeeper(rate_config)
        status = gk.get_queue_status()
        assert "pending" in status
        assert "calls_last_minute" in status

    def test_get_queue_status_after_call(self, rate_config):
        gk = ApiGatekeeper(rate_config)
        gk.execute(_identity, 1)
        status = gk.get_queue_status()
        assert status["calls_last_minute"] == 1

    def test_execute_retries_on_transient_failure(self, rate_config):
        call_log = []

        def flaky():
            call_log.append(1)
            if len(call_log) < 3:
                raise RuntimeError("transient error")
            return "recovered"

        gk = ApiGatekeeper(rate_config)
        assert gk.execute(flaky) == "recovered"
        assert len(call_log) == 3

    def test_wait_for_capacity_sleeps_when_at_limit(self, tmp_path):
        """At the per-minute limit, _wait_for_capacity must sleep until freed."""
        config = {
            "rate_limits": {
                "version": "1.01",
                "services": {
                    "default": {
                        "requests_per_minute": 1,
                        "requests_per_hour": 1000,
                        "concurrent_max": 10,
                        "retry_after_seconds": 0,
                        "max_retries": 3,
                    }
                },
            }
        }
        p = tmp_path / "rate_limits.json"
        p.write_text(json.dumps(config))
        gk = ApiGatekeeper(str(p))
        gk._call_times = [__import__("time").monotonic()]  # already at limit (rpm=1)

        def clear_after_sleep(_seconds):
            gk._call_times = []  # free up capacity so the loop exits

        with patch(
            "fitness_rl.shared.gatekeeper.time.sleep", side_effect=clear_after_sleep
        ) as sleep_mock:
            gk._wait_for_capacity()
        sleep_mock.assert_called_once()

    def test_execute_raises_after_max_retries_exhausted(self, rate_config):
        def always_fails():
            raise RuntimeError("permanent failure")

        gk = ApiGatekeeper(rate_config)
        with pytest.raises(RuntimeError, match="failed after"):
            gk.execute(always_fails)


def _write_config(tmp_path, version="1.01"):
    """Write a rate-limits config with distinct kaggle (10) and default (30) rpm."""
    config = {
        "rate_limits": {
            "version": version,
            "services": {
                "kaggle": {"requests_per_minute": 10, "max_retries": 3},
                "default": {"requests_per_minute": 30, "max_retries": 3},
            },
        }
    }
    p = tmp_path / "rate_limits.json"
    p.write_text(json.dumps(config))
    return str(p)


class TestServiceLimits:
    def test_uses_service_specific_limits(self, tmp_path):
        """The kaggle service must apply its own stricter 10 rpm, not default."""
        gk = ApiGatekeeper(_write_config(tmp_path), service="kaggle")
        assert gk._rpm == 10

    def test_falls_back_to_default_when_service_absent(self, tmp_path):
        """An unknown service must fall back to the default limits."""
        gk = ApiGatekeeper(_write_config(tmp_path), service="unknown")
        assert gk._rpm == 30

    def test_rejects_rate_limits_version_mismatch(self, tmp_path):
        """A config version != code version must raise at construction."""
        with pytest.raises(RuntimeError, match="does not match"):
            ApiGatekeeper(_write_config(tmp_path, version="9.99"))
