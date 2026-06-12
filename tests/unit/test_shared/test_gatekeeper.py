"""Unit tests for ApiGatekeeper — execute, retry, and queue status."""

import json

import pytest

from fitness_rl.shared.gatekeeper import ApiGatekeeper


@pytest.fixture()
def rate_config(tmp_path) -> str:
    """Rate-limits config with high RPM so tests never block on capacity."""
    config = {
        "rate_limits": {
            "version": "1.00",
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

    def test_execute_raises_after_max_retries_exhausted(self, rate_config):
        def always_fails():
            raise RuntimeError("permanent failure")

        gk = ApiGatekeeper(rate_config)
        with pytest.raises(RuntimeError, match="failed after"):
            gk.execute(always_fails)
