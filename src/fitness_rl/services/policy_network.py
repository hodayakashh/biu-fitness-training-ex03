"""
PolicyNetwork and CriticNetwork — actor and critic MLP architectures.

Both share the same trunk (two ReLU hidden layers). Only the output head differs:
  - PolicyNetwork → Categorical distribution over workout actions (used by REINFORCE + A2C).
  - CriticNetwork → scalar V(s) estimate (used by A2C only).

Sharing the trunk in _SharedTrunk avoids duplicating hidden-layer code across
two nearly-identical networks (BIU no-duplication rule).
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor
from torch.distributions import Categorical


class _SharedTrunk(nn.Module):
    """Two-layer ReLU MLP shared by PolicyNetwork and CriticNetwork."""

    def __init__(self, state_dim: int, hidden: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class PolicyNetwork(nn.Module):
    """
    Stochastic policy π_θ(a|s) — maps a state to a Categorical distribution.

    Input:  state tensor of shape (..., state_dim)
    Output: Categorical distribution over N_ACTIONS discrete workout types
    Setup:  state_dim, n_actions, hidden loaded from config via REINFORCETrainer/A2CTrainer
    """

    def __init__(self, state_dim: int, n_actions: int, hidden: int = 64) -> None:
        """Build shared trunk + classification head."""
        super().__init__()
        self._trunk = _SharedTrunk(state_dim, hidden)
        self._head = nn.Linear(hidden // 2, n_actions)

    def forward(self, state: Tensor) -> Categorical:
        """
        Return a Categorical distribution given the current state.

        Args:
            state: Float tensor (..., state_dim).

        Returns:
            Categorical distribution — call .sample() to act, .log_prob(a) for PG update.
        """
        return Categorical(logits=self._head(self._trunk(state)))


class CriticNetwork(nn.Module):
    """
    Value function V_ψ(s) — maps state to scalar expected return.

    Used only in A2C to compute the TD advantage δ_t = r_t + γV(s') − V(s).

    Input:  state tensor of shape (..., state_dim)
    Output: scalar tensor of shape (...)
    Setup:  state_dim, hidden loaded from config via A2CTrainer
    """

    def __init__(self, state_dim: int, hidden: int = 64) -> None:
        """Build shared trunk + value head."""
        super().__init__()
        self._trunk = _SharedTrunk(state_dim, hidden)
        self._head = nn.Linear(hidden // 2, 1)

    def forward(self, state: Tensor) -> Tensor:
        """
        Estimate V(s).

        Args:
            state: Float tensor (..., state_dim).

        Returns:
            Scalar value tensor of shape (...).
        """
        return self._head(self._trunk(state)).squeeze(-1)
