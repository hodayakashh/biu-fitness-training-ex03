"""
Global RNG seeding for reproducible, fair algorithm comparison.

Why: REINFORCE and A2C must be compared head-to-head on the SAME environment.
Re-seeding numpy and torch to the same value at the start of each training run
guarantees both algorithms see identical episode-initialisation streams and
identical action-sampling noise, so any performance difference is attributable
to the algorithm — not to luck. It also makes every reported number reproducible.
"""

from __future__ import annotations

import numpy as np
import torch


def set_global_seed(seed: int | None) -> None:
    """
    Seed numpy and torch RNGs.

    Args:
        seed: Integer seed. If None, seeding is skipped (non-deterministic run).
    """
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
