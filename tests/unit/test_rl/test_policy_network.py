"""Unit tests for PolicyNetwork and CriticNetwork — output shapes, distributions, gradients."""

import pytest
import torch
from torch.distributions import Categorical

from fitness_rl.services.policy_network import CriticNetwork, PolicyNetwork

STATE_DIM = 5
N_ACTIONS = 6


@pytest.fixture()
def policy() -> PolicyNetwork:
    torch.manual_seed(0)
    return PolicyNetwork(state_dim=STATE_DIM, n_actions=N_ACTIONS, hidden=32)


@pytest.fixture()
def critic() -> CriticNetwork:
    torch.manual_seed(0)
    return CriticNetwork(state_dim=STATE_DIM, hidden=32)


class TestPolicyNetwork:
    def test_returns_categorical_distribution(self, policy):
        state = torch.randn(STATE_DIM)
        dist = policy(state)
        assert isinstance(dist, Categorical)

    def test_probs_sum_to_one(self, policy):
        state = torch.randn(STATE_DIM)
        dist = policy(state)
        assert torch.isclose(dist.probs.sum(), torch.tensor(1.0), atol=1e-5)

    def test_action_in_valid_range(self, policy):
        state = torch.randn(STATE_DIM)
        dist = policy(state)
        action = dist.sample()
        assert 0 <= int(action.item()) < N_ACTIONS

    def test_log_prob_is_scalar(self, policy):
        state = torch.randn(STATE_DIM)
        dist = policy(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        assert log_prob.shape == torch.Size([])

    def test_log_prob_is_negative(self, policy):
        """log P(a|s) ≤ 0 for any valid probability."""
        state = torch.randn(STATE_DIM)
        dist = policy(state)
        action = dist.sample()
        assert dist.log_prob(action).item() <= 0.0

    def test_batch_input_shape(self, policy):
        states = torch.randn(8, STATE_DIM)
        dist = policy(states)
        assert dist.probs.shape == (8, N_ACTIONS)

    def test_gradients_flow_through_log_prob(self, policy):
        state = torch.randn(STATE_DIM)
        dist = policy(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        log_prob.backward()
        grads = [p.grad for p in policy.parameters() if p.grad is not None]
        assert len(grads) > 0


class TestCriticNetwork:
    def test_output_is_scalar_for_single_state(self, critic):
        state = torch.randn(STATE_DIM)
        val = critic(state)
        assert val.shape == torch.Size([])

    def test_output_shape_for_batch(self, critic):
        states = torch.randn(8, STATE_DIM)
        vals = critic(states)
        assert vals.shape == (8,)

    def test_output_is_finite(self, critic):
        state = torch.randn(STATE_DIM)
        val = critic(state)
        assert torch.isfinite(val)

    def test_gradients_flow(self, critic):
        state = torch.randn(STATE_DIM, requires_grad=False)
        val = critic(state)
        val.backward()
        grads = [p.grad for p in critic.parameters() if p.grad is not None]
        assert len(grads) > 0
