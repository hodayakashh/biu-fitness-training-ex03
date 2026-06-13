"""
RewardFunction — composable reward for the fitness RL environment.

Extracted from RLEnvironment as an independent building block (BIU §18):
single responsibility, dependency-injected config, independently testable.

    r_t = gain_t − λ₁·overload_penalty − λ₂·imbalance_penalty − λ₃·repetition_penalty

Why the repetition term is action-driven (not state-driven): the LSTM world
model barely differentiates between actions, so a penalty read from the
predicted state never reacts to the agent's actual choices and the policy
collapses onto a single action. Penalising low action-variety directly —
using the agent's OWN recent action history — forces genuine diversity.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from ..constants import STATE_MUSCLE_BALANCE, STATE_ROLLING_LOAD


class RewardFunction:
    """
    Compute the balanced fitness reward from a predicted state + action history.

    Input:  reward config dict, number of discrete actions
    Output: scalar reward via compute()
    Setup:  all weights (λ values, thresholds) injected from config — no literals
    """

    def __init__(self, reward_cfg: dict, n_actions: int) -> None:
        """Store config weights and action count for entropy normalisation."""
        self._cfg = reward_cfg
        self._n_actions = n_actions

    def compute(
        self,
        next_state: np.ndarray,
        recent_actions: deque[int] | None = None,
    ) -> float:
        """
        Compute r_t = gain − λ₁·overload − λ₂·imbalance − λ₃·repetition.

        Args:
            next_state:     Predicted next state array (state_dim,), normalised.
            recent_actions: Rolling window of the agent's last-K actually-taken
                            actions. When None/empty the repetition term is 0.

        Returns:
            Scalar reward value.
        """
        rolling_load = float(np.clip(next_state[STATE_ROLLING_LOAD], 0.0, 1.0))
        muscle_balance = float(np.clip(next_state[STATE_MUSCLE_BALANCE], 0.0, 1.0))

        overload_thr: float = self._cfg.get("overload_threshold_norm", 0.8)
        optimal_load: float = self._cfg.get("optimal_load_norm", 0.5)
        lambda1: float = self._cfg.get("lambda_overload", 0.4)
        lambda2: float = self._cfg.get("lambda_imbalance", 0.3)
        lambda3: float = self._cfg.get("lambda_repetition", 0.5)

        # Bell-shaped gain: peaks at optimal_load, falls symmetrically either side.
        gain = max(0.0, 1.0 - abs(rolling_load - optimal_load) / max(optimal_load, 1e-8))
        overload = max(0.0, rolling_load - overload_thr) / max(1.0 - overload_thr, 1e-8)
        imbalance = 1.0 - muscle_balance
        repetition = self.repetition_penalty(recent_actions)

        return gain - lambda1 * overload - lambda2 * imbalance - lambda3 * repetition

    def repetition_penalty(self, recent_actions: deque[int] | None) -> float:
        """
        Penalise low variety in the agent's recently chosen actions.

        Uses the Shannon entropy of the empirical action distribution over the
        rolling window, normalised by log(n_actions) into [0, 1]:

            repetition_penalty = 1 − H / log(n_actions)

        All-same window → entropy 0 → penalty 1.0. Uniform action use →
        penalty 0.0. A single action (or empty window) carries no information,
        so its penalty is 0.0 to avoid penalising the very first step.

        Args:
            recent_actions: Rolling window of recently taken action indices.

        Returns:
            Repetition penalty in [0, 1].
        """
        if recent_actions is None or len(recent_actions) <= 1:
            return 0.0

        counts = np.bincount(np.asarray(recent_actions, dtype=np.int64))
        probs = counts[counts > 0] / counts.sum()
        entropy = float(-np.sum(probs * np.log(probs)))
        max_entropy = np.log(max(self._n_actions, 2))
        return float(np.clip(1.0 - entropy / max_entropy, 0.0, 1.0))
