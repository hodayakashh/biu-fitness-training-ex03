"""
A2CTrainer — Advantage Actor-Critic with per-step online updates.

Update rule (TD advantage):
    δ_t = r_t + γ V(s') (1 − done) − V(s)
    loss_actor  = −log π(a|s) · δ.detach()
    loss_critic = value_loss_coeff · δ²
    loss_entropy= −entropy_bonus · H[π(·|s)]

Per-step updates give lower variance than Monte Carlo while still being
fully online — no replay buffer required (see docs/PRD_A2C.md).
"""

from __future__ import annotations

import torch

from ..services.policy_network import CriticNetwork, PolicyNetwork
from ..services.rl_env import RLEnvironment
from ..shared.config import ConfigManager
from ..shared.seeding import set_global_seed


class A2CTrainer:
    """
    Train an actor (PolicyNetwork) and critic (CriticNetwork) with A2C.

    Input:  cfg (hyperparams), trained LSTMTransitionModel, data dict
    Output: dict with actor, critic, episode_returns (list[float])
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
        Run the full A2C training loop.

        Each episode performs per-step TD updates for both actor and critic.
        The advantage δ_t is computed online, avoiding the need to store
        full trajectories before updating (unlike Monte Carlo REINFORCE).

        Returns:
            dict with keys: actor (PolicyNetwork), critic (CriticNetwork),
            episode_returns (list[float]).
        """
        # Re-seed so A2C trains on the same RNG stream as REINFORCE (fair compare).
        set_global_seed(self._cfg.get("seed"))
        rl_cfg: dict = self._cfg.get("rl") or {}
        n_episodes: int = rl_cfg.get("n_episodes", 500)
        gamma: float = rl_cfg.get("gamma", 0.99)
        actor_lr: float = rl_cfg.get("actor_lr", 3e-4)
        critic_lr: float = rl_cfg.get("critic_lr", 1e-3)
        value_coeff: float = rl_cfg.get("value_loss_coeff", 0.5)
        entropy_bonus: float = rl_cfg.get("entropy_bonus", 0.01)

        actor, critic = self._build_networks()
        actor_opt = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        critic_opt = torch.optim.Adam(critic.parameters(), lr=critic_lr)

        episode_returns: list[float] = []

        for _ in range(n_episodes):
            ep_return = self._run_episode(
                actor,
                critic,
                actor_opt,
                critic_opt,
                gamma,
                value_coeff,
                entropy_bonus,
            )
            episode_returns.append(ep_return)

        return {"actor": actor, "critic": critic, "episode_returns": episode_returns}

    def _run_episode(
        self,
        actor: PolicyNetwork,
        critic: CriticNetwork,
        actor_opt: torch.optim.Optimizer,
        critic_opt: torch.optim.Optimizer,
        gamma: float,
        value_coeff: float,
        entropy_bonus: float,
    ) -> float:
        """
        Execute one episode with per-step A2C updates.

        At each step the TD error δ_t is computed and both networks are
        updated immediately — no trajectory buffer is needed.

        Args:
            actor:        PolicyNetwork (in training mode).
            critic:       CriticNetwork (in training mode).
            actor_opt:    Adam optimiser for the actor.
            critic_opt:   Adam optimiser for the critic.
            gamma:        Discount factor in [0, 1].
            value_coeff:  Weight of critic loss in the combined loss.
            entropy_bonus: Weight of entropy regularisation term.

        Returns:
            Total undiscounted episode return (sum of rewards).
        """
        state = self._env.reset()
        done = False
        total_reward = 0.0

        while not done:
            state_t = torch.from_numpy(state).float()
            dist = actor(state_t)
            action = dist.sample()

            next_state, reward, done = self._env.step(int(action.item()))
            total_reward += reward

            v_t = critic(state_t)
            next_state_t = torch.from_numpy(next_state).float()
            v_next = critic(next_state_t).detach()

            # TD advantage: bootstrap from V(s') only if episode not done
            delta = reward + gamma * v_next * (1.0 - float(done)) - v_t

            loss_actor = -dist.log_prob(action) * delta.detach()
            loss_critic = value_coeff * delta.pow(2)
            loss_entropy = -entropy_bonus * dist.entropy()
            loss = loss_actor + loss_critic + loss_entropy

            actor_opt.zero_grad()
            critic_opt.zero_grad()
            loss.backward()
            actor_opt.step()
            critic_opt.step()

            state = next_state

        return total_reward

    def _build_networks(self) -> tuple[PolicyNetwork, CriticNetwork]:
        """Instantiate actor and critic from config hyperparams."""
        data_cfg: dict = self._cfg.get("data") or {}
        state_dim: int = data_cfg.get("state_dim", 5)
        n_actions: int = data_cfg.get("n_actions", 6)
        actor = PolicyNetwork(state_dim=state_dim, n_actions=n_actions)
        critic = CriticNetwork(state_dim=state_dim)
        return actor, critic
