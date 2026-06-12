"""
LSTMTransitionModel — learned dynamics model for the RL environment.

Role in the pipeline:
    s_{t+1} ≈ f_φ(s_t, a_t, h_t)

The LSTM is NOT the decision-maker. It answers:
    "Given the past seq_len days of (state, action) pairs, what state
    is likely tomorrow?"
REINFORCE and A2C use this model as their world model to generate episodes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class LSTMTransitionModel(nn.Module):
    """
    Two-layer LSTM with action embeddings that predicts the next trainee state.

    Input:
        x_states:  float tensor (batch, seq_len, state_dim)
        x_actions: long  tensor (batch, seq_len) — discrete action indices
    Output:
        float tensor (batch, state_dim) — predicted next state ŝ_{t+1}
    Setup:
        All hyperparameters loaded from config/setup.json via LSTMTrainer.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        action_embed_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        """
        Build embedding, LSTM stack, and linear projection head.

        Args:
            state_dim:        Dimension of the state vector (5).
            n_actions:        Number of discrete workout actions (6).
            action_embed_dim: Learned embedding size for each action (8).
            hidden_size:      LSTM hidden units per layer (64).
            num_layers:       Number of stacked LSTM layers (2).
            dropout:          Dropout between LSTM layers (0.2).
        """
        super().__init__()
        self.action_embedding = nn.Embedding(n_actions, action_embed_dim)
        self.lstm = nn.LSTM(
            input_size=state_dim + action_embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, state_dim)

    def forward(self, x_states: Tensor, x_actions: Tensor) -> Tensor:
        """
        Predict next trainee state from a window of past (state, action) pairs.

        Args:
            x_states:  Float tensor of shape (batch, seq_len, state_dim).
            x_actions: Long  tensor of shape (batch, seq_len).

        Returns:
            Float tensor of shape (batch, state_dim) — predicted s_{t+1}.
        """
        action_emb = self.action_embedding(x_actions)  # (B, T, embed)
        lstm_in = torch.cat([x_states, action_emb], dim=2)  # (B, T, state+embed)
        out, _ = self.lstm(lstm_in)  # (B, T, hidden)
        return self.fc(out[:, -1, :])  # (B, state_dim)
