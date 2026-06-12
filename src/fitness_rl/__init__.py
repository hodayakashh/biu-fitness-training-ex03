"""
fitness_rl — BIU DRL Ex03: LSTM + REINFORCE + A2C for personal fitness planning.

Public API is exposed entirely through the SDK class. All external consumers
(notebooks, CLI, tests) should import from here and use FitnessRLSDK.
"""

__version__ = "1.00"
__all__ = ["FitnessRLSDK"]

from .sdk.sdk import FitnessRLSDK
