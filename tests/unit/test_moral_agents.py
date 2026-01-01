"""Unit tests for moral agents."""

import numpy as np
import pytest
import torch

from src.agents.moral_agents import (
    AdaptiveNeuralAgent,
    DeontologicalAgent,
    EgoistAgent,
    SupervisorAgent,
    UtilitarianAgent,
    VirtueEthicsAgent,
    create_agent,
)


class TestAgentFactory:
    """Tests for the agent factory function."""

    def test_create_utilitarian(self):
        """Test creating a utilitarian agent."""
        agent = create_agent("utilitarian", "test_agent")
        assert isinstance(agent, UtilitarianAgent)
        assert agent.agent_id == "test_agent"
        assert agent.moral_framework == "utilitarian"

    def test_create_deontological(self):
        """Test creating a deontological agent with custom params."""
        agent = create_agent("deontological", "test_agent", fair_share_rule=0.3)
        assert isinstance(agent, DeontologicalAgent)
        assert agent.fair_share_rule == 0.3

    def test_create_virtue_ethics(self):
        """Test creating a virtue ethics agent."""
        agent = create_agent("virtue_ethics", "test_agent")
        assert isinstance(agent, VirtueEthicsAgent)

    def test_create_egoist(self):
        """Test creating an egoist agent."""
        agent = create_agent("egoist", "test_agent")
        assert isinstance(agent, EgoistAgent)

    def test_create_adaptive(self):
        """Test creating an adaptive neural agent."""
        agent = create_agent("adaptive", "test_agent", obs_dim=7)
        assert isinstance(agent, AdaptiveNeuralAgent)

    def test_create_supervisor(self):
        """Test creating a supervisor agent."""
        agent = create_agent("supervisor", "test_agent", target_behavior="competitive")
        assert isinstance(agent, SupervisorAgent)
        assert agent.target_behavior == "competitive"

    def test_invalid_agent_type(self):
        """Test that invalid agent type raises error."""
        with pytest.raises(ValueError, match="Unknown agent type"):
            create_agent("invalid_type", "test_agent")


class TestUtilitarianAgent:
    """Tests for utilitarian agent behavior."""

    def test_action_in_valid_range(self, utilitarian_agent, sample_observation):
        """Test that action is in [0, 1]."""
        action = utilitarian_agent.act(sample_observation)
        assert action.shape == (1,)
        assert 0 <= action[0] <= 1

    def test_takes_less_when_above_average(self, utilitarian_agent):
        """Utilitarian should take less when above average."""
        # Own resources (40) > average (25) = be modest
        obs = np.array([40.0, 25.0, 50.0, 0.2, 0.3, 0.3, 0.3], dtype=np.float32)
        action = utilitarian_agent.act(obs)
        assert action[0] < 0.5  # Should be modest

    def test_takes_more_when_below_average(self, utilitarian_agent):
        """Utilitarian should take more when below average."""
        # Own resources (10) < average (30) = can claim more
        obs = np.array([10.0, 30.0, 50.0, 0.2, 0.3, 0.3, 0.3], dtype=np.float32)
        action = utilitarian_agent.act(obs)
        assert action[0] > 0.3  # Should claim more


class TestDeontologicalAgent:
    """Tests for deontological agent behavior."""

    def test_always_takes_fair_share(self, deontological_agent, sample_observation):
        """Deontological agent always takes exactly fair share."""
        action = deontological_agent.act(sample_observation)
        assert action[0] == pytest.approx(0.25, rel=1e-5)

    def test_consistent_regardless_of_situation(self, deontological_agent):
        """Deontological agent is consistent regardless of situation."""
        obs1 = np.array([10.0, 50.0, 80.0, 0.1, 0.9, 0.9, 0.9], dtype=np.float32)
        obs2 = np.array([90.0, 10.0, 20.0, 0.9, 0.1, 0.1, 0.1], dtype=np.float32)

        action1 = deontological_agent.act(obs1)
        action2 = deontological_agent.act(obs2)

        assert action1[0] == action2[0] == 0.25


class TestVirtueEthicsAgent:
    """Tests for virtue ethics agent behavior."""

    def test_action_in_valid_range(self, virtue_ethics_agent, sample_observation):
        """Test that action is in [0, 1]."""
        action = virtue_ethics_agent.act(sample_observation)
        assert action.shape == (1,)
        assert 0 <= action[0] <= 1

    def test_practices_moderation(self, virtue_ethics_agent):
        """Virtue ethics agent practices moderation."""
        obs = np.array([25.0, 25.0, 50.0, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        action = virtue_ethics_agent.act(obs)
        # When equal to average, should be moderate (around 0.4)
        assert 0.2 <= action[0] <= 0.6


class TestEgoistAgent:
    """Tests for egoist agent behavior."""

    def test_greedy_behavior(self, egoist_agent, sample_observation):
        """Egoist should always be greedy."""
        action = egoist_agent.act(sample_observation)
        assert action[0] >= 0.85  # Should be very greedy

    def test_takes_everything_when_scarce(self, egoist_agent):
        """Egoist takes everything when resources are scarce."""
        obs = np.array([50.0, 25.0, 15.0, 0.8, 0.3, 0.3, 0.3], dtype=np.float32)
        action = egoist_agent.act(obs)
        assert action[0] == 1.0


class TestAdaptiveNeuralAgent:
    """Tests for adaptive neural agent behavior."""

    def test_action_in_valid_range(self, adaptive_agent, sample_observation):
        """Test that action is in [0, 1] due to sigmoid."""
        action = adaptive_agent.act(sample_observation)
        assert action.shape == (1,)
        assert 0 <= action[0] <= 1

    def test_network_structure(self, adaptive_agent):
        """Test that network has correct structure."""
        assert adaptive_agent.network is not None
        # Check it can process input
        obs = torch.randn(1, 7)
        output = adaptive_agent.network(obs)
        assert output.shape == (1, 1)
        assert 0 <= output.item() <= 1  # Sigmoid output

    def test_update_with_sufficient_data(self, adaptive_agent, seed):
        """Test that update works with sufficient data."""
        observations = [np.random.randn(7).astype(np.float32) for _ in range(10)]
        actions = [np.array([np.random.rand()], dtype=np.float32) for _ in range(10)]
        rewards = [float(np.random.rand()) for _ in range(10)]

        # Should not raise
        adaptive_agent.update(observations, actions, rewards)

    def test_update_skips_insufficient_data(self, adaptive_agent):
        """Test that update is skipped with insufficient data."""
        observations = [np.random.randn(7).astype(np.float32)]
        actions = [np.array([0.5], dtype=np.float32)]
        rewards = [1.0]

        # Should not raise, just skip
        adaptive_agent.update(observations, actions, rewards)


class TestSupervisorAgent:
    """Tests for supervisor agent behavior."""

    def test_cooperative_sets_example(self, supervisor_agent):
        """Cooperative supervisor sets cooperative example."""
        # Others are being greedy
        obs = np.array([30.0, 25.0, 50.0, 0.3, 0.8, 0.8, 0.8], dtype=np.float32)
        action = supervisor_agent.act(obs)
        assert action[0] <= 0.3  # Should set cooperative example

    def test_competitive_supervisor(self):
        """Competitive supervisor encourages competition."""
        agent = SupervisorAgent("comp_sup", target_behavior="competitive")
        obs = np.array([30.0, 25.0, 50.0, 0.3, 0.3, 0.3, 0.3], dtype=np.float32)
        action = agent.act(obs)
        assert action[0] >= 0.7  # Should encourage competition

    def test_influence_signal(self, supervisor_agent):
        """Test influence signal generation."""
        group_state = {
            "resources": {"agent_0": 30.0, "agent_1": 25.0, "agent_2": 35.0, "agent_3": 10.0}
        }
        signal = supervisor_agent.get_influence_signal(group_state)
        assert isinstance(signal, float)


class TestAgentHistory:
    """Tests for agent history tracking."""

    def test_history_tracking(self, utilitarian_agent, sample_observation):
        """Test that agents track history correctly."""
        action = utilitarian_agent.act(sample_observation)
        utilitarian_agent.update_history(action, reward=5.0)

        assert len(utilitarian_agent.action_history) == 1
        assert len(utilitarian_agent.reward_history) == 1
        assert utilitarian_agent.reward_history[0] == 5.0
