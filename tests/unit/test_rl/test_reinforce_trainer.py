"""Unit tests for REINFORCETrainer — episode generation, returns, PG update."""

import json

import numpy as np
import pytest
import torch

from fitness_rl.services.policy_network import PolicyNetwork
from fitness_rl.services.reinforce_trainer import REINFORCETrainer
from fitness_rl.shared.config import ConfigManager

STATE_DIM = 5
N_ACTIONS = 3
EPISODE_LEN = 6


class _MockEnv:
    """Minimal deterministic environment for testing trainers without LSTM."""

    def __init__(self) -> None:
        self._step = 0
        self._rng = np.random.default_rng(99)

    @property
    def state_dim(self) -> int:
        return STATE_DIM

    def reset(self) -> np.ndarray:
        self._step = 0
        return self._rng.random(STATE_DIM).astype(np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        self._step += 1
        next_s = self._rng.random(STATE_DIM).astype(np.float32)
        reward = float(self._rng.uniform(-0.5, 1.0))
        return next_s, reward, self._step >= EPISODE_LEN


@pytest.fixture()
def cfg(tmp_path) -> ConfigManager:
    setup = {
        "version": "1.00",
        "data": {"seq_len": 5, "n_actions": N_ACTIONS, "state_dim": STATE_DIM},
        "rl": {"episode_length": EPISODE_LEN, "n_episodes": 5, "gamma": 0.99,
               "reinforce_lr": 0.003},
        "reward": {"overload_threshold_norm": 0.8, "optimal_load_norm": 0.5,
                   "lambda_overload": 0.4, "lambda_imbalance": 0.3},
    }
    p = tmp_path / "setup.json"
    p.write_text(json.dumps(setup))
    return ConfigManager(p)


@pytest.fixture()
def policy() -> PolicyNetwork:
    torch.manual_seed(7)
    return PolicyNetwork(state_dim=STATE_DIM, n_actions=N_ACTIONS, hidden=16)


class TestComputeReturns:
    def test_length_equals_rewards(self):
        rewards = [1.0, 2.0, 3.0]
        returns = REINFORCETrainer._compute_returns(rewards, gamma=0.99)
        assert len(returns) == len(rewards)

    def test_last_return_equals_last_reward(self):
        rewards = [1.0, 2.0, 5.0]
        returns = REINFORCETrainer._compute_returns(rewards, gamma=0.99)
        assert returns[-1].item() == pytest.approx(5.0)

    def test_first_return_is_discounted_sum(self):
        rewards = [1.0, 0.0, 0.0]
        returns = REINFORCETrainer._compute_returns(rewards, gamma=0.5)
        assert returns[0].item() == pytest.approx(1.0)

    def test_gamma_zero_equals_immediate_reward(self):
        rewards = [3.0, 99.0, 99.0]
        returns = REINFORCETrainer._compute_returns(rewards, gamma=0.0)
        assert returns[0].item() == pytest.approx(3.0)


class TestPGLoss:
    def test_loss_is_scalar(self, policy):
        state = torch.randn(STATE_DIM)
        dist = policy(state)
        log_probs = [dist.log_prob(dist.sample()) for _ in range(4)]
        returns = torch.tensor([1.0, 2.0, 3.0, 4.0])
        loss = REINFORCETrainer._pg_loss(log_probs, returns)
        assert loss.shape == torch.Size([])

    def test_loss_has_gradient(self, policy):
        state = torch.randn(STATE_DIM)
        dist = policy(state)
        log_probs = [dist.log_prob(dist.sample()) for _ in range(4)]
        returns = torch.tensor([1.0, 2.0, 3.0, 4.0])
        loss = REINFORCETrainer._pg_loss(log_probs, returns)
        loss.backward()
        grads = [p.grad for p in policy.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_zero_advantage_gives_zero_loss(self, policy):
        """When all returns are identical, advantages = 0 → loss = 0."""
        state = torch.randn(STATE_DIM)
        dist = policy(state)
        log_probs = [dist.log_prob(dist.sample()) for _ in range(4)]
        returns = torch.ones(4) * 5.0
        loss = REINFORCETrainer._pg_loss(log_probs, returns)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)


class TestRunEpisodeAndTrain:
    def test_run_episode_length(self, cfg, policy):
        """Episode must produce exactly episode_length steps."""
        trainer = REINFORCETrainer.__new__(REINFORCETrainer)
        trainer._cfg = cfg
        trainer._env = _MockEnv()
        log_probs, rewards = trainer._run_episode(policy)
        assert len(log_probs) == EPISODE_LEN
        assert len(rewards) == EPISODE_LEN

    def test_train_returns_expected_keys(self, cfg):
        trainer = REINFORCETrainer.__new__(REINFORCETrainer)
        trainer._cfg = cfg
        trainer._env = _MockEnv()
        result = trainer.train()
        assert {"policy", "episode_returns", "return_variance"}.issubset(result.keys())

    def test_episode_returns_length(self, cfg):
        trainer = REINFORCETrainer.__new__(REINFORCETrainer)
        trainer._cfg = cfg
        trainer._env = _MockEnv()
        result = trainer.train()
        assert len(result["episode_returns"]) == 5  # n_episodes=5

    def test_policy_is_policy_network(self, cfg):
        trainer = REINFORCETrainer.__new__(REINFORCETrainer)
        trainer._cfg = cfg
        trainer._env = _MockEnv()
        result = trainer.train()
        assert isinstance(result["policy"], PolicyNetwork)
