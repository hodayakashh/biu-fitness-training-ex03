"""
DataWindows — sliding window tensor builder for LSTM supervised training.

Target rule: y[i] = states[i + seq_len].
The prediction target is strictly after the input window → zero data leakage.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from ..shared.config import ConfigManager


class DataWindows:
    """
    Build (state_window, action_window, next_state) training tuples.

    Input:  normalised state array (N, state_dim), integer action array (N,)
    Output: three tensors — X_states (float32), X_actions (int64), y (float32)
    Setup:  seq_len and n_actions come from ConfigManager
    """

    def __init__(self, cfg: ConfigManager) -> None:
        """Initialise with config reference (no mutable state)."""
        self._cfg = cfg

    def build(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        seq_len: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Slide a window of length seq_len over the state/action sequence.

        Window layout per sample i:
            X_states[i]  = states[i : i+seq_len]      shape (seq_len, state_dim)
            X_actions[i] = actions[i : i+seq_len]     shape (seq_len,)  — long ints
            y[i]         = states[i + seq_len]         shape (state_dim,)

        y[i] is always one step past the end of the input window, so the
        LSTM is never trained on information from the future.

        Args:
            states:  Normalised state array of shape (N, state_dim).
            actions: Integer action labels of shape (N,).
            seq_len: Number of time steps per window (days).

        Returns:
            Tuple (X_states, X_actions, y) as PyTorch tensors.

        Raises:
            ValueError: If the sequence is too short to form at least one window.
        """
        n = len(states) - seq_len
        if n <= 0:
            raise ValueError(
                f"Too few samples ({len(states)}) for seq_len={seq_len}. "
                "Need at least seq_len + 1 samples."
            )

        state_dim = states.shape[1]
        x_states = np.zeros((n, seq_len, state_dim), dtype=np.float32)
        x_actions = np.zeros((n, seq_len), dtype=np.int64)
        y = np.zeros((n, state_dim), dtype=np.float32)

        for i in range(n):
            x_states[i] = states[i : i + seq_len]
            x_actions[i] = actions[i : i + seq_len]
            y[i] = states[i + seq_len]

        return torch.from_numpy(x_states), torch.from_numpy(x_actions), torch.from_numpy(y)
