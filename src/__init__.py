"""
Moral-Decision Multi-Agent Simulator

A research-grade multi-agent reinforcement learning environment for studying
moral decision-making, featuring Claude-powered LLM agents.
"""

__version__ = "0.2.0"

from src.agents.moral_agents import (
    MoralAgent,
    UtilitarianAgent,
    DeontologicalAgent,
    VirtueEthicsAgent,
    EgoistAgent,
    AdaptiveNeuralAgent,
    SupervisorAgent,
    create_agent,
)
from src.environments.moral_dilemma_env import MoralDilemmaEnv
from src.metrics.moral_metrics import (
    MoralMetrics,
    GreatestGoodBenchmark,
    PeerPressureAnalyzer,
)

__all__ = [
    # Agents
    "MoralAgent",
    "UtilitarianAgent",
    "DeontologicalAgent",
    "VirtueEthicsAgent",
    "EgoistAgent",
    "AdaptiveNeuralAgent",
    "SupervisorAgent",
    "create_agent",
    # Environment
    "MoralDilemmaEnv",
    # Metrics
    "MoralMetrics",
    "GreatestGoodBenchmark",
    "PeerPressureAnalyzer",
]
