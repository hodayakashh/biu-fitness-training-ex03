"""
REINFORCETrainer — episodic policy gradient training (Monte Carlo returns).

Update rule (with mean baseline to reduce variance):
    θ ← θ + α Σ_t ∇_θ log π_θ(a_t | s_t) · (G_t − mean(G))

Why baseline: subtracting the mean return per episode keeps the same gradient
direction but substantially lowers the variance of the policy gradient estimate,
making training more stable without introducing bias.
"""

from __future__ import annotations

import torch

from ..services.policy_network import PolicyNetwork
from ..services.rl_env import RLEnvironment
from ..shared.config import ConfigManager


class REINFORCETrainer:
    """
    Train a PolicyNetwork with the REINFORCE algorithm.

    Input:  cfg (hyperparams), trained LSTMTransitionModel, data dict
    Output: dict with policy, episode_returns (list), return_variance (list)
    Setup:  RLEnvironment is created internally from lstm_model + data
    """

    def __init__(
        self,
        cfg: ConfigManager,
        lstm_model,
        data: dict,
    ) -> None:
        """Initialise trainer and wrap the LSTM in an RLEnvironment."""
        self._cfg = cfg
        self._env = RLEnvironment(cfg, lstm_model, data)

    def train(self) -> dict:
        """
        Run the full REINFORCE training loop.

        Each iteration generates one 28-day episode, computes Monte Carlo
        returns G_t, and updates the policy via the policy-gradient theorem.

        Returns:
            dict with keys: policy (PolicyNetwork), episode_returns (list[float]),
            return_variance (list[float]).
        """
        rl_cfg: dict = self._cfg.get("rl") or {}
        n_episodes: int = rl_cfg.get("n_episodes", 500)
        gamma: float = rl_cfg.get("gamma", 0.99)
        lr: float = rl_cfg.get("reinforce_lr", 3e-4)

        policy = self._build_policy()
        optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

        episode_returns: list[float] = []
        return_variance: list[float] = []

        for _ in range(n_episodes):
            log_probs, rewards = self._run_episode(policy)
            returns = self._compute_returns(rewards, gamma)

            loss = self._pg_loss(log_probs, returns)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            episode_returns.append(float(sum(rewards)))
            return_variance.append(float(returns.var().item()))

        return {
            "policy": policy,
            "episode_returns": episode_returns,
            "return_variance": return_variance,
        }

    def _run_episode(self, policy: PolicyNetwork) -> tuple[list[torch.Tensor], list[float]]:
        """
        Generate one episode using the current policy and LSTM environment.

        Args:
            policy: Current PolicyNetwork (in training mode).

        Returns:
            Tuple of (log_probs list, rewards list) — one entry per time step.
        """
        log_probs: list[torch.Tensor] = []
        rewards: list[float] = []
        state = self._env.reset()
        done = False

        while not done:
            state_t = torch.from_numpy(state).float()
            dist = policy(state_t)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            state, reward, done = self._env.step(int(action.item()))
            rewards.append(reward)

        return log_probs, rewards

    @staticmethod
    def _compute_returns(rewards: list[float], gamma: float) -> torch.Tensor:
        """
        Compute discounted Monte Carlo returns G_t = Σ_{k≥t} γ^{k-t} r_k.

        Iterates in reverse to accumulate in O(T) time.

        Args:
            rewards: List of per-step rewards from one episode.
            gamma:   Discount factor in [0, 1].

        Returns:
            Float tensor of shape (T,) with G_t at each position t.
        """
        returns: list[float] = []
        g = 0.0
        for r in reversed(rewards):
            g = r + gamma * g
            returns.insert(0, g)
        return torch.tensor(returns, dtype=torch.float32)

    @staticmethod
    def _pg_loss(log_probs: list[torch.Tensor], returns: torch.Tensor) -> torch.Tensor:
        """
        Compute the (negated) policy gradient loss with mean baseline.

        Args:
            log_probs: Log-probabilities log π_θ(a_t | s_t) for each step.
            returns:   Discounted returns G_t tensor of shape (T,).

        Returns:
            Scalar loss tensor (negated so gradient ascent via Adam becomes descent).
        """
        advantages = (returns - returns.mean()).detach()
        return -(torch.stack(log_probs) * advantages).sum()

    def _build_policy(self) -> PolicyNetwork:
        """Instantiate PolicyNetwork from config hyperparams."""
        data_cfg: dict = self._cfg.get("data") or {}
        return PolicyNetwork(
            state_dim=data_cfg.get("state_dim", 5),
            n_actions=data_cfg.get("n_actions", 6),
        )
