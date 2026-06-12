"""
RLEnvironment — fitness training environment backed by the LSTM transition model.

The environment wraps the trained LSTM so that RL algorithms (REINFORCE, A2C)
can generate episodes without needing real workout data at inference time.

Episode structure:
    s_0 = reset()
    for t in range(episode_length):
        a_t ~ policy(s_t)
        s_{t+1}, r_t, done = step(a_t)
"""

from __future__ import annotations

import numpy as np
import torch

from ..constants import STATE_MUSCLE_BALANCE, STATE_ROLLING_LOAD
from ..services.lstm_model import LSTMTransitionModel
from ..shared.config import ConfigManager


class RLEnvironment:
    """
    Single-agent episodic environment for workout-plan optimisation.

    Input:  trained LSTMTransitionModel, data dict from DataService.run_pipeline()
    Output: (next_state, reward, done) via step(); numpy arrays throughout
    Setup:  episode_length, seq_len, and reward λ values from ConfigManager
    """

    def __init__(
        self,
        cfg: ConfigManager,
        lstm_model: LSTMTransitionModel,
        data: dict,
    ) -> None:
        """
        Initialise the environment with a trained LSTM world model.

        Training windows are stored to sample realistic starting states
        for each episode.

        Args:
            cfg:        Shared ConfigManager (rl/*, reward/*, data/* keys).
            lstm_model: Trained LSTMTransitionModel used as world model.
            data:       Output of DataService.run_pipeline().
        """
        self._model = lstm_model
        self._reward_cfg: dict = cfg.get("reward") or {}
        self._episode_length: int = cfg.get_nested("rl", "episode_length") or 28
        self._seq_len: int = cfg.get_nested("data", "seq_len") or 7

        # Keep training windows for realistic episode initialisation
        self._x_train: np.ndarray = data["X_train"].numpy()  # (N, seq_len, state_dim)
        self._xa_train: np.ndarray = data["X_actions_train"].numpy()  # (N, seq_len)

        self._state: np.ndarray = np.zeros(self._x_train.shape[2])
        self._history_states: np.ndarray = np.zeros_like(self._x_train[0])
        self._history_actions: np.ndarray = np.zeros(self._seq_len, dtype=np.int64)
        self._step_count: int = 0

    @property
    def state_dim(self) -> int:
        """Dimension of the state vector."""
        return self._x_train.shape[2]

    def reset(self) -> np.ndarray:
        """
        Sample a random training window as the initial episode context.

        Using a real window (not repeated copies of s_0) gives the LSTM
        a plausible hidden-state warm-up from the first step onward.

        Returns:
            Current state s_0 as a float32 numpy array of shape (state_dim,).
        """
        idx = np.random.randint(0, len(self._x_train))
        self._history_states = self._x_train[idx].copy()  # (seq_len, state_dim)
        self._history_actions = self._xa_train[idx].copy()  # (seq_len,)
        self._state = self._history_states[-1].copy()
        self._step_count = 0
        return self._state.copy()

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        """
        Execute one environment step.

        Process:
            1. Place the chosen action at the end of the history window.
            2. Feed (history_states, history_actions) to the LSTM.
            3. Compute reward from the predicted next state.
            4. Slide history window forward.

        Args:
            action: Integer workout-type action (0 .. n_actions-1).

        Returns:
            Tuple (next_state, reward, done).
            next_state: float32 array (state_dim,).
            reward:     scalar float.
            done:       True if the episode has reached episode_length.
        """
        self._history_actions[-1] = action

        x_s = torch.from_numpy(self._history_states[np.newaxis]).float()  # (1, T, D)
        x_a = torch.from_numpy(self._history_actions[np.newaxis])  # (1, T)

        self._model.eval()
        with torch.no_grad():
            next_state = self._model(x_s, x_a).squeeze(0).numpy()

        next_state = np.clip(next_state, -1.5, 1.5)
        reward = self._compute_reward(next_state)

        # Slide window: discard oldest, append new state; keep current action
        self._history_states = np.roll(self._history_states, -1, axis=0)
        self._history_states[-1] = next_state
        self._history_actions = np.roll(self._history_actions, -1)
        self._history_actions[-1] = 0  # placeholder; overwritten on next step

        self._state = next_state
        self._step_count += 1
        return next_state.copy(), float(reward), self._step_count >= self._episode_length

    def _compute_reward(self, next_state: np.ndarray) -> float:
        """
        Compute r_t = gain_t − λ₁·overload_penalty_t − λ₂·imbalance_penalty_t.

        All inputs are normalised state features in [0, 1] space so no
        raw-unit thresholds are needed — overload_threshold_norm is in [0, 1].

        Args:
            next_state: Predicted next state array (state_dim,).

        Returns:
            Scalar reward value.
        """
        rolling_load = float(np.clip(next_state[STATE_ROLLING_LOAD], 0.0, 1.0))
        muscle_balance = float(np.clip(next_state[STATE_MUSCLE_BALANCE], 0.0, 1.0))

        overload_thr: float = self._reward_cfg.get("overload_threshold_norm", 0.8)
        optimal_load: float = self._reward_cfg.get("optimal_load_norm", 0.5)
        lambda1: float = self._reward_cfg.get("lambda_overload", 0.4)
        lambda2: float = self._reward_cfg.get("lambda_imbalance", 0.3)

        # Bell-shaped gain: peaks at optimal_load, falls symmetrically on both sides.
        # Prevents the agent from maximising volume without bound.
        gain = max(0.0, 1.0 - abs(rolling_load - optimal_load) / max(optimal_load, 1e-8))
        overload_penalty = max(0.0, rolling_load - overload_thr) / max(1.0 - overload_thr, 1e-8)
        imbalance_penalty = 1.0 - muscle_balance

        return gain - lambda1 * overload_penalty - lambda2 * imbalance_penalty
