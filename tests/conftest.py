"""Pytest configuration and shared fixtures."""

import pytest
import numpy as np
import torch

from src.environments.moral_dilemma_env import MoralDilemmaEnv
from src.agents.moral_agents import (
    create_agent,
    UtilitarianAgent,
    DeontologicalAgent,
    VirtueEthicsAgent,
    EgoistAgent,
    AdaptiveNeuralAgent,
    SupervisorAgent,
)
from src.metrics.moral_metrics import GreatestGoodBenchmark, PeerPressureAnalyzer


@pytest.fixture
def seed():
    """Set random seeds for reproducibility."""
    np.random.seed(42)
    torch.manual_seed(42)
    return 42


@pytest.fixture
def basic_env():
    """Create a basic environment for testing."""
    return MoralDilemmaEnv(
        num_agents=4,
        total_resources=100,
        episode_length=50,
        reward_structure="mixed",
        peer_influence_strength=0.3,
    )


@pytest.fixture
def small_env():
    """Create a small environment for quick tests."""
    return MoralDilemmaEnv(
        num_agents=2,
        total_resources=50,
        episode_length=10,
        reward_structure="selfish",
        peer_influence_strength=0.0,
    )


@pytest.fixture
def sample_observation():
    """Create a sample observation array."""
    # [own_resources, avg_group_resources, remaining_resources, timestep, other_actions...]
    return np.array([25.0, 25.0, 75.0, 0.1, 0.3, 0.4, 0.5], dtype=np.float32)


@pytest.fixture
def sample_observation_small():
    """Create a sample observation for 2-agent environment."""
    return np.array([25.0, 25.0, 50.0, 0.1, 0.5], dtype=np.float32)


@pytest.fixture
def utilitarian_agent():
    """Create a utilitarian agent."""
    return UtilitarianAgent("test_utilitarian")


@pytest.fixture
def deontological_agent():
    """Create a deontological agent."""
    return DeontologicalAgent("test_deontological", fair_share_rule=0.25)


@pytest.fixture
def virtue_ethics_agent():
    """Create a virtue ethics agent."""
    return VirtueEthicsAgent("test_virtue")


@pytest.fixture
def egoist_agent():
    """Create an egoist agent."""
    return EgoistAgent("test_egoist")


@pytest.fixture
def adaptive_agent():
    """Create an adaptive neural agent."""
    return AdaptiveNeuralAgent("test_adaptive", obs_dim=7, hidden_dim=32)


@pytest.fixture
def supervisor_agent():
    """Create a supervisor agent."""
    return SupervisorAgent("test_supervisor", target_behavior="cooperative")


@pytest.fixture
def all_agents(sample_observation):
    """Create one of each agent type."""
    return {
        "utilitarian": UtilitarianAgent("agent_0"),
        "deontological": DeontologicalAgent("agent_1", fair_share_rule=0.25),
        "virtue_ethics": VirtueEthicsAgent("agent_2"),
        "egoist": EgoistAgent("agent_3"),
    }


@pytest.fixture
def ggb():
    """Create a Greatest Good Benchmark instance."""
    return GreatestGoodBenchmark(num_agents=4)


@pytest.fixture
def peer_analyzer():
    """Create a Peer Pressure Analyzer instance."""
    return PeerPressureAnalyzer()


@pytest.fixture
def sample_actions():
    """Create sample actions dict for testing metrics."""
    return {
        "agent_0": 0.3,
        "agent_1": 0.25,
        "agent_2": 0.4,
        "agent_3": 0.8,
    }


@pytest.fixture
def sample_resources():
    """Create sample resources dict for testing metrics."""
    return {
        "agent_0": 30.0,
        "agent_1": 25.0,
        "agent_2": 35.0,
        "agent_3": 10.0,
    }


@pytest.fixture
def sample_rewards():
    """Create sample rewards dict for testing metrics."""
    return {
        "agent_0": 5.0,
        "agent_1": 4.0,
        "agent_2": 6.0,
        "agent_3": 2.0,
    }
