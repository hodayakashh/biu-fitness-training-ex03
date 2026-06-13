"""
RLEnvironment — hybrid world model for the fitness training task (PLAN ADR-001).

Dynamics are split across two sources:
  * the trained LSTM predicts the history-dependent fatigue/temporal dimensions
    (rolling_load, session_duration, day_sin/cos), and
  * the muscle_balance dimension is governed by a KNOWN, action-conditioned rule:
    the normalised entropy of the mean muscle-group profile of the agent's recent
    actions. This makes the agent's choices causally and persistently change the
    state (fixing the near-degenerate MDP) and makes the imbalance penalty
    action-aware through the environment rather than via an out-of-model hack.

Episode structure:
    s_0 = reset()
    for t in range(episode_length):
        a_t ~ policy(s_t)
        s_{t+1}, r_t, done = step(a_t)
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch

from ..constants import N_ACTIONS, STATE_MUSCLE_BALANCE
from ..services.lstm_model import LSTMTransitionModel
from ..services.reward import RewardFunction
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

        # Number of distinct actions, used to normalise the variety entropy.
        self._n_actions: int = cfg.get_nested("data", "n_actions") or N_ACTIONS
        # Rolling window length over the agent's OWN recent actions (not LSTM).
        self._variety_window: int = self._reward_cfg.get("variety_window", 7)
        # Reward is an injected building block (single responsibility, testable).
        self._reward = RewardFunction(self._reward_cfg, self._n_actions)

        # Keep training windows for realistic episode initialisation
        self._x_train: np.ndarray = data["X_train"].numpy()  # (N, seq_len, state_dim)
        self._xa_train: np.ndarray = data["X_actions_train"].numpy()  # (N, seq_len)

        # Action-conditioned muscle dynamics (ADR-001). Each action carries an
        # empirical muscle-group profile; muscle_balance is the normalised entropy of
        # the mean profile over a rolling window of the agent's chosen actions. When
        # the data dict omits profiles (e.g. synthetic unit-test data) the override is
        # disabled and the LSTM's own muscle_balance prediction is kept unchanged.
        profiles = data.get("action_muscle_profiles")
        self._profiles: np.ndarray | None = (
            np.asarray(profiles, dtype=np.float32) if profiles is not None else None
        )
        n_mg = self._profiles.shape[1] if self._profiles is not None else 2
        self._max_entropy: float = float(np.log(max(n_mg, 2)))
        self._muscle_window: deque[np.ndarray] = deque(maxlen=self._variety_window)

        self._state: np.ndarray = np.zeros(self._x_train.shape[2])
        self._history_states: np.ndarray = np.zeros_like(self._x_train[0])
        self._history_actions: np.ndarray = np.zeros(self._seq_len, dtype=np.int64)
        # Rolling window of actions the agent ACTUALLY took this episode. This
        # is the agent's real choice history — unlike _history_actions (which is
        # consumed by the LSTM and warm-started from a training window), this is
        # what the repetition penalty inspects so it reflects true behaviour.
        self._action_window: deque[int] = deque(maxlen=self._variety_window)
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
        self._action_window.clear()
        self._muscle_window.clear()
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
        # Record the agent's real choice so the repetition penalty sees the
        # actual action history rather than the LSTM's prediction.
        self._action_window.append(int(action))
        if self._profiles is not None:
            self._muscle_window.append(self._profiles[action % self._profiles.shape[0]])

        x_s = torch.from_numpy(self._history_states[np.newaxis]).float()  # (1, T, D)
        x_a = torch.from_numpy(self._history_actions[np.newaxis])  # (1, T)

        self._model.eval()
        with torch.no_grad():
            next_state = self._model(x_s, x_a).squeeze(0).numpy()

        next_state = np.clip(next_state, -1.5, 1.5)
        # Override the LSTM's (action-blind) muscle_balance with the action-conditioned
        # value so the agent's choices genuinely shape this state dimension (ADR-001).
        if self._profiles is not None:
            next_state[STATE_MUSCLE_BALANCE] = self._muscle_balance_from_actions()
        reward = self._reward.compute(next_state, self._action_window)

        # Slide window: discard oldest, append new state; keep current action
        self._history_states = np.roll(self._history_states, -1, axis=0)
        self._history_states[-1] = next_state
        self._history_actions = np.roll(self._history_actions, -1)
        self._history_actions[-1] = 0  # placeholder; overwritten on next step

        self._state = next_state
        self._step_count += 1
        return next_state.copy(), float(reward), self._step_count >= self._episode_length

    def _muscle_balance_from_actions(self) -> float:
        """
        Action-conditioned muscle balance ∈ [0, 1] (ADR-001).

        Computes the normalised Shannon entropy of the MEAN muscle-group profile over
        the agent's last-`variety_window` chosen actions. Repeating one archetype
        concentrates exposure on few muscle groups → low entropy → low balance → high
        imbalance penalty; alternating archetypes that target different groups spreads
        exposure → high entropy → high balance → low penalty.

        Returns:
            Normalised entropy in [0, 1]; 0 when the window is empty.
        """
        if not self._muscle_window:
            return 0.0
        mean_profile = np.mean(np.stack(self._muscle_window), axis=0)
        probs = mean_profile[mean_profile > 0]
        entropy = float(-np.sum(probs * np.log(probs)))
        return float(np.clip(entropy / max(self._max_entropy, 1e-8), 0.0, 1.0))
