"""
fitness_rl — BIU DRL Ex03: LSTM + REINFORCE + A2C for personal fitness planning.

Public API is exposed entirely through the SDK class. All external consumers
(notebooks, CLI, tests) should import from here and use FitnessRLSDK.

v1.01: hybrid world model — action-conditioned muscle-balance dynamics (PLAN ADR-001).
"""

__version__ = "1.01"
__all__ = ["FitnessRLSDK"]

from .sdk.sdk import FitnessRLSDK
