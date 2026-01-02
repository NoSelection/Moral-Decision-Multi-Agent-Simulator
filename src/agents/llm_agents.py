"""
LLM-based moral agents using Claude and Gemini APIs.

This module implements moral agents that use LLMs for genuine moral reasoning,
enabling comparison between rule-based heuristics, learned policies, and
language model reasoning.

Supported providers:
- Anthropic Claude (requires anthropic package)
- Google Gemini (requires google-generativeai package) - FREE TIER AVAILABLE
"""

import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from src.agents.moral_agents import MoralAgent


class LLMProvider(Enum):
    """Supported LLM providers."""

    CLAUDE = "claude"
    GEMINI = "gemini"


# Try to import anthropic, but make it optional
try:
    from anthropic import Anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    Anthropic = None

# Try to import Google's generative AI, but make it optional
try:
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None


@dataclass
class ReasoningTrace:
    """Container for a single reasoning trace from an LLM agent."""

    observation: List[float]
    reasoning: str
    action: float
    moral_framework: str
    timestamp: int = 0
    confidence: Optional[float] = None


@dataclass
class LLMAgentConfig:
    """Configuration for LLM-based agents."""

    provider: str = "claude"  # "claude" or "gemini"
    model: str = "claude-sonnet-4-20250514"  # or "gemini-2.5-flash" for Gemini
    max_tokens: int = 1024
    temperature: float = 0.7
    reasoning_style: str = "chain_of_thought"  # chain_of_thought, direct, socratic

    @classmethod
    def claude_default(cls) -> "LLMAgentConfig":
        """Default config for Claude."""
        return cls(
            provider="claude",
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            temperature=0.7,
        )

    @classmethod
    def gemini_default(cls) -> "LLMAgentConfig":
        """Default config for Gemini (free tier friendly)."""
        return cls(
            provider="gemini",
            model="gemini-2.5-flash",  # Fast and free tier eligible
            max_tokens=1024,
            temperature=0.7,
        )


# Moral framework system prompts
MORAL_FRAMEWORK_PROMPTS = {
    "utilitarian": """Your moral framework is UTILITARIAN. You believe in maximizing total welfare.

Key principles you follow:
- The greatest good for the greatest number
- Aggregate wellbeing matters more than individual outcomes
- Be willing to sacrifice personal gain for group benefit
- Consider long-term consequences of your actions
- Evaluate outcomes based on total utility produced

When making decisions, calculate the expected total welfare for each option
and choose the one that maximizes overall good.""",
    "deontological": """Your moral framework is DEONTOLOGICAL. You follow moral rules regardless of consequences.

Key principles you follow:
- Always claim your fair share (approximately 1/N of available resources where N is number of agents)
- Never take more than your fair share, even if it would benefit others
- Rules must be followed consistently without exception
- Intentions and duties matter more than outcomes
- Treat moral rules as categorical imperatives

When making decisions, identify the relevant moral rule and follow it strictly,
regardless of the expected consequences.""",
    "virtue_ethics": """Your moral framework is VIRTUE ETHICS. You aim to embody virtuous character.

Key virtues you cultivate:
- Moderation (temperance): Avoid excess in all things
- Justice: Give others what they are due
- Practical wisdom (phronesis): Discern the right action in context
- Generosity: Be willing to give to others
- Fairness: Treat similar cases similarly

When making decisions, ask yourself: "What would a virtuous person do in this situation?"
Seek the golden mean between extremes.""",
    "care_ethics": """Your moral framework is CARE ETHICS. You prioritize relationships and responsiveness.

Key principles you follow:
- Maintain and nurture relationships with other agents
- Respond to the particular needs of others rather than abstract rules
- Consider the context and history of interactions
- Value empathy and emotional responsiveness
- Recognize interdependence and mutual vulnerability

When making decisions, consider how your action affects your relationships
with other agents and how to best care for those who depend on shared resources.""",
    "contractarian": """Your moral framework is CONTRACTARIAN. You act based on rational agreements.

Key principles you follow:
- Act according to principles that rational agents would agree to
- Imagine negotiations from behind a "veil of ignorance"
- Seek mutually beneficial arrangements
- Honor implicit social contracts
- Fairness emerges from rational agreement

When making decisions, ask: "What rule would all agents rationally agree to
if none knew their position in advance?" """,
    "flexible": """You are a FLEXIBLE moral reasoner who considers multiple ethical perspectives.

You draw upon:
- Utilitarianism (consequences and total welfare)
- Deontology (duties and rules)
- Virtue ethics (character and virtues)
- Care ethics (relationships and responsiveness)
- Contractarianism (rational agreement)

When making decisions, consider multiple frameworks and explain which ones
inform your decision and why. Be explicit about trade-offs between frameworks.""",
}


BASE_SYSTEM_PROMPT = """You are a moral agent participating in a multi-agent resource allocation scenario.

## The Scenario
You are one of several agents who must decide how much of a shared resource pool to claim.
This creates a moral dilemma between self-interest and group welfare.

## Your Task
Given observations about the current state, you must decide what fraction of the
remaining resources to claim (a number between 0.0 and 1.0).

## Observations You Receive
- Your current accumulated resources
- Average resources across all agents
- Remaining resources in the shared pool
- Current progress through the episode (0.0 to 1.0)
- Recent claim fractions from other agents

## Important Considerations
- If everyone claims too much, resources deplete quickly and everyone suffers
- If you claim too little, others may take advantage
- Your decision affects both your welfare and the group's welfare
- Other agents have their own moral frameworks and strategies

Think through your reasoning carefully before making a decision.
"""


class LLMAgent(MoralAgent):
    """Base class for LLM-powered moral agents."""

    def __init__(
        self,
        agent_id: str,
        moral_framework: str = "flexible",
        config: Optional[LLMAgentConfig] = None,
        api_key: Optional[str] = None,
        request_timeout: float = 15.0,
        max_attempts: int = 2,
    ):
        """Initialize an LLM-based moral agent.

        Args:
            agent_id: Unique identifier for this agent
            moral_framework: Which moral framework to use (utilitarian, deontological, etc.)
            config: Configuration for the LLM
            api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if not provided)
            request_timeout: Per-request timeout (seconds)
            max_attempts: Number of attempts before falling back to safe default
        """
        super().__init__(agent_id, f"llm_{moral_framework}")

        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "anthropic package not installed. Install with: pip install anthropic"
            )

        self.moral_framework = moral_framework
        self.config = config or LLMAgentConfig()
        self.request_timeout = request_timeout
        self.max_attempts = max_attempts

        # Initialize API client
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self.client = Anthropic(api_key=api_key)

        # Build system prompt
        self.system_prompt = self._build_system_prompt()

        # Store reasoning traces for analysis
        self.reasoning_history: List[ReasoningTrace] = []
        self.timestep = 0

    def _build_system_prompt(self) -> str:
        """Construct the system prompt based on moral framework."""
        framework_prompt = MORAL_FRAMEWORK_PROMPTS.get(
            self.moral_framework, MORAL_FRAMEWORK_PROMPTS["flexible"]
        )
        return BASE_SYSTEM_PROMPT + "\n\n" + framework_prompt

    def _format_observation(self, observation: np.ndarray) -> str:
        """Convert observation array to natural language prompt."""
        own_resources = observation[0]
        avg_resources = observation[1]
        remaining = observation[2]
        progress = observation[3]
        others_actions = observation[4:].tolist()

        prompt = f"""## Current Situation

**Your Resources:** {own_resources:.1f}
**Average Group Resources:** {avg_resources:.1f}
**Remaining Resources in Pool:** {remaining:.1f}
**Episode Progress:** {progress*100:.0f}%
**Other Agents' Recent Claims:** {[f"{a:.2f}" for a in others_actions]}

## Your Analysis

Based on your moral framework, analyze this situation:
1. What are the key moral considerations here?
2. How does your framework guide your decision?
3. What trade-offs do you see?

## Your Decision

After your analysis, provide your claim fraction.

**Format your response exactly as follows:**

REASONING: [Your step-by-step moral reasoning]

DECISION: [A single number between 0.0 and 1.0]
"""
        return prompt

    def _parse_response(self, response: str) -> tuple[str, float]:
        """Parse Claude's response into reasoning and action.

        Args:
            response: Raw response text from Claude

        Returns:
            Tuple of (reasoning_text, action_value)
        """
        reasoning = ""
        action = 0.25  # Default to fair share if parsing fails

        try:
            # Extract reasoning
            if "REASONING:" in response:
                reasoning_part = response.split("REASONING:")[-1]
                if "DECISION:" in reasoning_part:
                    reasoning = reasoning_part.split("DECISION:")[0].strip()
                else:
                    reasoning = reasoning_part.strip()

            # Extract decision
            if "DECISION:" in response:
                decision_part = response.split("DECISION:")[-1].strip()
                # Find the first number in the decision part
                numbers = re.findall(r"(\d+\.?\d*)", decision_part)
                if numbers:
                    action = float(numbers[0])
                    action = np.clip(action, 0.0, 1.0)

        except Exception as e:
            reasoning = f"Parsing error: {e}. Defaulting to fair share."
            action = 0.25

        return reasoning, action

    def act(self, observation: np.ndarray) -> np.ndarray:
        """Make a decision using Claude's moral reasoning.

        Args:
            observation: Environment observation array

        Returns:
            Action array with claim fraction [0, 1]
        """
        # Format observation into natural language
        prompt = self._format_observation(observation)

        # Get Claude's reasoning with guarded retries to avoid hangs
        response_text = self._generate_response(prompt)

        reasoning, action = self._parse_response(response_text)

        # Store reasoning trace
        trace = ReasoningTrace(
            observation=observation.tolist(),
            reasoning=reasoning,
            action=action,
            moral_framework=self.moral_framework,
            timestamp=self.timestep,
        )
        self.reasoning_history.append(trace)
        self.timestep += 1

        return np.array([action], dtype=np.float32)

    def explain_decision(self, observation: Optional[np.ndarray] = None) -> str:
        """Get explanation for the last decision.

        Args:
            observation: Optional observation to explain (uses last if not provided)

        Returns:
            String explanation of the reasoning
        """
        if self.reasoning_history:
            last_trace = self.reasoning_history[-1]
            return f"[{self.moral_framework.upper()} REASONING]\n{last_trace.reasoning}"
        return "No decisions made yet."

    def get_reasoning_history(self) -> List[Dict[str, Any]]:
        """Get full reasoning history as list of dicts."""
        return [
            {
                "observation": trace.observation,
                "reasoning": trace.reasoning,
                "action": trace.action,
                "framework": trace.moral_framework,
                "timestamp": trace.timestamp,
            }
            for trace in self.reasoning_history
        ]


    def _generate_response(self, prompt: str) -> str:
        """Call the LLM with retries and a deterministic fallback on failure."""
        last_error: Optional[Exception] = None

        for attempt in range(self.max_attempts):
            try:
                response = self.client.messages.create(
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    system=self.system_prompt,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=self.request_timeout,
                )
                return response.content[0].text
            except Exception as exc:  # pragma: no cover - defensive against network issues
                last_error = exc
                continue

        # Deterministic safe fallback if all attempts fail
        return f"REASONING: API unavailable ({last_error}). Defaulting to fair share.\nDECISION: 0.25"


class ClaudeUtilitarianAgent(LLMAgent):
    """Claude agent with utilitarian moral framework."""

    def __init__(self, agent_id: str, **kwargs):
        super().__init__(agent_id, moral_framework="utilitarian", **kwargs)


class ClaudeDeontologicalAgent(LLMAgent):
    """Claude agent with deontological moral framework."""

    def __init__(self, agent_id: str, **kwargs):
        super().__init__(agent_id, moral_framework="deontological", **kwargs)


class ClaudeVirtueEthicsAgent(LLMAgent):
    """Claude agent with virtue ethics moral framework."""

    def __init__(self, agent_id: str, **kwargs):
        super().__init__(agent_id, moral_framework="virtue_ethics", **kwargs)


class ClaudeCareEthicsAgent(LLMAgent):
    """Claude agent with care ethics moral framework."""

    def __init__(self, agent_id: str, **kwargs):
        super().__init__(agent_id, moral_framework="care_ethics", **kwargs)


class ClaudeFlexibleAgent(LLMAgent):
    """Claude agent that considers multiple moral frameworks."""

    def __init__(self, agent_id: str, **kwargs):
        super().__init__(agent_id, moral_framework="flexible", **kwargs)


# =============================================================================
# GEMINI AGENTS (FREE TIER AVAILABLE!)
# =============================================================================


class GeminiAgent(MoralAgent):
    """LLM-powered moral agent using Google's Gemini API.

    Gemini offers a generous free tier, making it ideal for experimentation
    and development without API costs.

    Free tier limits (as of 2024):
    - 15 requests per minute
    - 1 million tokens per minute
    - 1500 requests per day
    """

    def __init__(
        self,
        agent_id: str,
        moral_framework: str = "flexible",
        config: Optional[LLMAgentConfig] = None,
        api_key: Optional[str] = None,
        request_timeout: float = 15.0,
        max_attempts: int = 3,
    ):
        """Initialize a Gemini-based moral agent.

        Args:
            agent_id: Unique identifier for this agent
            moral_framework: Which moral framework to use
            config: Configuration for the LLM
            api_key: Google API key (uses GOOGLE_API_KEY or GEMINI_API_KEY env var if not provided)
            request_timeout: Per-request timeout (seconds)
            max_attempts: Retry attempts before falling back
        """
        super().__init__(agent_id, f"gemini_{moral_framework}")

        if not GEMINI_AVAILABLE:
            raise ImportError(
                "google-generativeai package not installed. "
                "Install with: pip install google-generativeai"
            )

        self.moral_framework = moral_framework
        self.config = config or LLMAgentConfig.gemini_default()
        self.request_timeout = request_timeout
        self.max_attempts = max_attempts

        # Initialize API
        api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Google API key required. Set GOOGLE_API_KEY or GEMINI_API_KEY "
                "environment variable or pass api_key parameter."
            )

        genai.configure(api_key=api_key)

        # Create the model
        generation_config = {
            "temperature": self.config.temperature,
            "max_output_tokens": self.config.max_tokens,
        }
        self.model = genai.GenerativeModel(
            model_name=self.config.model,
            generation_config=generation_config,
            system_instruction=self._build_system_prompt(),
        )

        # Store reasoning traces for analysis
        self.reasoning_history: List[ReasoningTrace] = []
        self.timestep = 0

    def _build_system_prompt(self) -> str:
        """Construct the system prompt based on moral framework."""
        framework_prompt = MORAL_FRAMEWORK_PROMPTS.get(
            self.moral_framework, MORAL_FRAMEWORK_PROMPTS["flexible"]
        )
        return BASE_SYSTEM_PROMPT + "\n\n" + framework_prompt

    def _format_observation(self, observation: np.ndarray) -> str:
        """Convert observation array to natural language prompt."""
        own_resources = observation[0]
        avg_resources = observation[1]
        remaining = observation[2]
        progress = observation[3]
        others_actions = observation[4:].tolist()

        prompt = f"""## Current Situation

**Your Resources:** {own_resources:.1f}
**Average Group Resources:** {avg_resources:.1f}
**Remaining Resources in Pool:** {remaining:.1f}
**Episode Progress:** {progress*100:.0f}%
**Other Agents' Recent Claims:** {[f"{a:.2f}" for a in others_actions]}

## Your Analysis

Based on your moral framework, analyze this situation:
1. What are the key moral considerations here?
2. How does your framework guide your decision?
3. What trade-offs do you see?

## Your Decision

After your analysis, provide your claim fraction.

**Format your response exactly as follows:**

REASONING: [Your step-by-step moral reasoning]

DECISION: [A single number between 0.0 and 1.0]
"""
        return prompt

    def _parse_response(self, response: str) -> tuple[str, float]:
        """Parse Gemini's response into reasoning and action."""
        reasoning = ""
        action = 0.25  # Default to fair share if parsing fails

        try:
            if "REASONING:" in response:
                reasoning_part = response.split("REASONING:")[-1]
                if "DECISION:" in reasoning_part:
                    reasoning = reasoning_part.split("DECISION:")[0].strip()
                else:
                    reasoning = reasoning_part.strip()

            if "DECISION:" in response:
                decision_part = response.split("DECISION:")[-1].strip()
                numbers = re.findall(r"(\d+\.?\d*)", decision_part)
                if numbers:
                    action = float(numbers[0])
                    action = np.clip(action, 0.0, 1.0)

        except Exception as e:
            reasoning = f"Parsing error: {e}. Defaulting to fair share."
            action = 0.25

        return reasoning, action

    def act(self, observation: np.ndarray) -> np.ndarray:
        """Make a decision using Gemini's moral reasoning."""
        prompt = self._format_observation(observation)
        response_text = self._generate_response(prompt)

        # Parse response (falls back to fair share on parsing failure)
        reasoning, action = self._parse_response(response_text)

        # Store reasoning trace
        trace = ReasoningTrace(
            observation=observation.tolist(),
            reasoning=reasoning,
            action=action,
            moral_framework=self.moral_framework,
            timestamp=self.timestep,
        )
        self.reasoning_history.append(trace)
        self.timestep += 1

        return np.array([action], dtype=np.float32)

    def explain_decision(self, observation: Optional[np.ndarray] = None) -> str:
        """Get explanation for the last decision."""
        if self.reasoning_history:
            last_trace = self.reasoning_history[-1]
            return f"[GEMINI {self.moral_framework.upper()} REASONING]\n{last_trace.reasoning}"
        return "No decisions made yet."

    def get_reasoning_history(self) -> List[Dict[str, Any]]:
        """Get full reasoning history as list of dicts."""
        return [
            {
                "observation": trace.observation,
                "reasoning": trace.reasoning,
                "action": trace.action,
                "framework": trace.moral_framework,
                "timestamp": trace.timestamp,
                "provider": "gemini",
            }
            for trace in self.reasoning_history
        ]


    def _generate_response(self, prompt: str) -> str:
        """Call Gemini with bounded retries and deterministic fallback."""
        last_error: Optional[Exception] = None

        for attempt in range(self.max_attempts):
            try:
                response = self.model.generate_content(prompt, request_options={"timeout": self.request_timeout})
                return response.text
            except Exception as exc:  # pragma: no cover - external dependency guard
                last_error = exc
                continue

        return f"REASONING: API unavailable ({last_error}). Defaulting to fair share.\nDECISION: 0.25"


class GeminiUtilitarianAgent(GeminiAgent):
    """Gemini agent with utilitarian moral framework."""

    def __init__(self, agent_id: str, **kwargs):
        super().__init__(agent_id, moral_framework="utilitarian", **kwargs)


class GeminiDeontologicalAgent(GeminiAgent):
    """Gemini agent with deontological moral framework."""

    def __init__(self, agent_id: str, **kwargs):
        super().__init__(agent_id, moral_framework="deontological", **kwargs)


class GeminiVirtueEthicsAgent(GeminiAgent):
    """Gemini agent with virtue ethics moral framework."""

    def __init__(self, agent_id: str, **kwargs):
        super().__init__(agent_id, moral_framework="virtue_ethics", **kwargs)


class GeminiFlexibleAgent(GeminiAgent):
    """Gemini agent that considers multiple moral frameworks."""

    def __init__(self, agent_id: str, **kwargs):
        super().__init__(agent_id, moral_framework="flexible", **kwargs)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


# Factory function for creating LLM agents
def create_llm_agent(
    moral_framework: str, agent_id: str, provider: str = "claude", **kwargs
) -> MoralAgent:
    """Factory function to create LLM-based moral agents.

    Args:
        moral_framework: Which moral framework to use (utilitarian, deontological, etc.)
        agent_id: Unique identifier for the agent
        provider: Which LLM provider to use ("claude" or "gemini")
        **kwargs: Additional arguments passed to the agent

    Returns:
        LLM agent instance with the specified moral framework

    Example:
        # Create a Claude agent
        agent = create_llm_agent("utilitarian", "agent_0", provider="claude")

        # Create a Gemini agent (free tier!)
        agent = create_llm_agent("utilitarian", "agent_0", provider="gemini")
    """
    provider = provider.lower()

    if provider == "gemini":
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "google-generativeai package not installed. "
                "Install with: pip install google-generativeai"
            )

        gemini_classes = {
            "utilitarian": GeminiUtilitarianAgent,
            "deontological": GeminiDeontologicalAgent,
            "virtue_ethics": GeminiVirtueEthicsAgent,
            "flexible": GeminiFlexibleAgent,
        }

        agent_class = gemini_classes.get(moral_framework, GeminiAgent)
        if moral_framework not in gemini_classes:
            return GeminiAgent(agent_id, moral_framework=moral_framework, **kwargs)
        return agent_class(agent_id, **kwargs)

    else:  # Default to Claude
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "anthropic package not installed. " "Install with: pip install anthropic"
            )

        claude_classes = {
            "utilitarian": ClaudeUtilitarianAgent,
            "deontological": ClaudeDeontologicalAgent,
            "virtue_ethics": ClaudeVirtueEthicsAgent,
            "care_ethics": ClaudeCareEthicsAgent,
            "flexible": ClaudeFlexibleAgent,
        }

        agent_class = claude_classes.get(moral_framework, LLMAgent)
        if moral_framework not in claude_classes:
            return LLMAgent(agent_id, moral_framework=moral_framework, **kwargs)
        return agent_class(agent_id, **kwargs)


def create_gemini_agent(moral_framework: str, agent_id: str, **kwargs) -> GeminiAgent:
    """Convenience function to create Gemini agents.

    Gemini has a FREE TIER - perfect for experimentation!

    Args:
        moral_framework: Which moral framework to use
        agent_id: Unique identifier for the agent
        **kwargs: Additional arguments (api_key, config, etc.)

    Returns:
        GeminiAgent instance
    """
    return create_llm_agent(moral_framework, agent_id, provider="gemini", **kwargs)


class MockLLMAgent(MoralAgent):
    """Mock LLM agent for testing without API calls.

    This agent simulates LLM behavior based on the moral framework,
    useful for testing and development without API costs.
    """

    def __init__(self, agent_id: str, moral_framework: str = "flexible", **kwargs):
        super().__init__(agent_id, f"mock_llm_{moral_framework}")
        self.moral_framework = moral_framework
        self.reasoning_history: List[ReasoningTrace] = []
        self.timestep = 0

    def act(self, observation: np.ndarray) -> np.ndarray:
        """Make a decision based on simulated moral reasoning."""
        own_resources = observation[0]
        avg_resources = observation[1]
        remaining = observation[2]
        others_actions = observation[4:]

        # Simulate different moral frameworks
        if self.moral_framework == "utilitarian":
            # Maximize group welfare
            if own_resources > avg_resources:
                action = 0.2  # Take less when above average
            else:
                action = 0.4  # Take moderate amount when below
            reasoning = f"As a utilitarian, I consider group welfare. Since my resources ({own_resources:.1f}) are {'above' if own_resources > avg_resources else 'below'} average ({avg_resources:.1f}), I claim {action:.2f}."

        elif self.moral_framework == "deontological":
            # Always take fair share
            action = 0.25
            reasoning = "As a deontologist, I follow the rule of fair share regardless of consequences. I claim exactly 0.25."

        elif self.moral_framework == "virtue_ethics":
            # Practice moderation
            avg_others = np.mean(others_actions) if len(others_actions) > 0 else 0.5
            action = 0.3 + 0.1 * (avg_others - 0.5)  # Moderate, adjusted by context
            action = np.clip(action, 0.15, 0.45)
            reasoning = f"As a virtue ethicist, I practice moderation. Others claim {avg_others:.2f} on average. I claim {action:.2f}."

        elif self.moral_framework == "care_ethics":
            # Respond to others' needs
            if avg_resources < own_resources:
                action = 0.15  # Give more to others
            else:
                action = 0.35  # Take fair amount
            reasoning = f"As a care ethicist, I respond to relationships. Group average is {avg_resources:.1f}, so I claim {action:.2f}."

        else:  # flexible
            # Balance multiple considerations
            action = 0.3
            reasoning = f"Considering multiple frameworks, I balance self-interest with group welfare. I claim {action:.2f}."

        # Add some noise for realism
        action = np.clip(action + np.random.normal(0, 0.05), 0.0, 1.0)

        # Store trace
        trace = ReasoningTrace(
            observation=observation.tolist(),
            reasoning=reasoning,
            action=action,
            moral_framework=self.moral_framework,
            timestamp=self.timestep,
        )
        self.reasoning_history.append(trace)
        self.timestep += 1

        return np.array([action], dtype=np.float32)

    def explain_decision(self, observation: Optional[np.ndarray] = None) -> str:
        """Get explanation for the last decision."""
        if self.reasoning_history:
            return self.reasoning_history[-1].reasoning
        return "No decisions made yet."
