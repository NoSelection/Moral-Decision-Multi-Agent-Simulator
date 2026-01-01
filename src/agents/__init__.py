"""Agent implementations for the moral decision simulator."""

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

from src.agents.maddpg import (
    MADDPG,
    MADDPGActor,
    MADDPGCritic,
    ReplayBuffer,
)

# LLM agents are optional (require anthropic or google-generativeai packages)
try:
    from src.agents.llm_agents import (
        # Core classes
        LLMAgent,
        LLMAgentConfig,
        LLMProvider,
        ReasoningTrace,
        # Claude agents
        ClaudeUtilitarianAgent,
        ClaudeDeontologicalAgent,
        ClaudeVirtueEthicsAgent,
        ClaudeCareEthicsAgent,
        ClaudeFlexibleAgent,
        # Gemini agents (FREE TIER!)
        GeminiAgent,
        GeminiUtilitarianAgent,
        GeminiDeontologicalAgent,
        GeminiVirtueEthicsAgent,
        GeminiFlexibleAgent,
        # Mock for testing
        MockLLMAgent,
        # Factory functions
        create_llm_agent,
        create_gemini_agent,
        # Availability flags
        ANTHROPIC_AVAILABLE,
        GEMINI_AVAILABLE,
    )

    LLM_AGENTS_AVAILABLE = True
except ImportError:
    LLM_AGENTS_AVAILABLE = False
    ANTHROPIC_AVAILABLE = False
    GEMINI_AVAILABLE = False

__all__ = [
    # Base agents
    "MoralAgent",
    "UtilitarianAgent",
    "DeontologicalAgent",
    "VirtueEthicsAgent",
    "EgoistAgent",
    "AdaptiveNeuralAgent",
    "SupervisorAgent",
    "create_agent",
    # MADDPG
    "MADDPG",
    "MADDPGActor",
    "MADDPGCritic",
    "ReplayBuffer",
    # LLM agents (when available)
    "LLM_AGENTS_AVAILABLE",
]

# Add LLM exports if available
if LLM_AGENTS_AVAILABLE:
    __all__.extend(
        [
            # Core
            "LLMAgent",
            "LLMAgentConfig",
            "LLMProvider",
            "ReasoningTrace",
            # Claude agents
            "ClaudeUtilitarianAgent",
            "ClaudeDeontologicalAgent",
            "ClaudeVirtueEthicsAgent",
            "ClaudeCareEthicsAgent",
            "ClaudeFlexibleAgent",
            # Gemini agents (FREE TIER!)
            "GeminiAgent",
            "GeminiUtilitarianAgent",
            "GeminiDeontologicalAgent",
            "GeminiVirtueEthicsAgent",
            "GeminiFlexibleAgent",
            # Mock and factories
            "MockLLMAgent",
            "create_llm_agent",
            "create_gemini_agent",
            # Availability flags
            "ANTHROPIC_AVAILABLE",
            "GEMINI_AVAILABLE",
        ]
    )
