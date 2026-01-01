"""
LLM Comparison Experiments

This module provides experiments to compare:
1. Rule-based agents (utilitarian, deontological, etc.)
2. Learned agents (MADDPG, AdaptiveNeuralAgent)
3. LLM-based agents (Claude and Gemini with different moral frameworks)

Key research questions:
- Does Claude/Gemini exhibit genuine moral reasoning or pattern matching?
- How does LLM reasoning compare to learned policies?
- Can LLMs maintain consistent moral frameworks?
- How do LLM agents influence group dynamics?
- Do different LLM providers reason differently about ethics?

Supported LLM Providers:
- Claude (Anthropic) - requires ANTHROPIC_API_KEY
- Gemini (Google) - FREE TIER AVAILABLE! requires GOOGLE_API_KEY or GEMINI_API_KEY
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Literal
from dataclasses import dataclass, asdict
from tqdm import tqdm

from src.environments.moral_dilemma_env import MoralDilemmaEnv
from src.agents.moral_agents import create_agent
from src.metrics.moral_metrics import GreatestGoodBenchmark, PeerPressureAnalyzer

# Try to import LLM agents
try:
    from src.agents.llm_agents import (
        create_llm_agent,
        create_gemini_agent,
        MockLLMAgent,
        LLMAgentConfig,
        ANTHROPIC_AVAILABLE,
        GEMINI_AVAILABLE,
    )

    LLM_AGENTS_AVAILABLE = ANTHROPIC_AVAILABLE or GEMINI_AVAILABLE
except ImportError:
    LLM_AGENTS_AVAILABLE = False
    ANTHROPIC_AVAILABLE = False
    GEMINI_AVAILABLE = False
    MockLLMAgent = None


@dataclass
class ExperimentConfig:
    """Configuration for an LLM comparison experiment."""

    name: str
    num_episodes: int = 20
    episode_length: int = 50
    num_agents: int = 4
    total_resources: int = 100
    reward_structure: str = "mixed"
    peer_influence_strength: float = 0.3
    use_mock_llm: bool = True  # Use mock for testing, real for production
    llm_provider: str = "gemini"  # "claude" or "gemini" - gemini has free tier!


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""

    config_name: str
    agent_types: Dict[str, str]
    metrics: Dict[str, float]
    reasoning_samples: List[Dict[str, Any]]
    episode_rewards: List[float]
    fairness_history: List[float]
    cooperation_history: List[float]


class LLMComparisonExperiment:
    """Compare LLM moral reasoning to rule-based and learned agents."""

    def __init__(
        self,
        output_dir: str = "results/llm_comparison",
        use_mock_llm: bool = True,
        llm_provider: str = "gemini",
        api_key: Optional[str] = None,
    ):
        """Initialize the experiment runner.

        Args:
            output_dir: Directory to save results
            use_mock_llm: Use mock LLM agents (no API calls) for testing
            llm_provider: Which LLM provider to use ("claude" or "gemini")
                         Gemini has a FREE TIER - recommended for experimentation!
            api_key: API key (uses env var if not provided)
        """
        self.output_dir = output_dir
        self.use_mock_llm = use_mock_llm
        self.llm_provider = llm_provider.lower()

        # Get API key based on provider
        if self.llm_provider == "gemini":
            self.api_key = (
                api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            )
        else:
            self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

        os.makedirs(output_dir, exist_ok=True)

        self.experiment_configs = self._define_experiments()

    def _define_experiments(self) -> List[Dict[str, Any]]:
        """Define the experiment configurations to run."""
        return [
            # Experiment 1: Baseline - All rule-based agents
            {
                "name": "baseline_rule_based",
                "description": "Baseline with all rule-based agents",
                "agents": [
                    {"type": "rule", "framework": "utilitarian"},
                    {"type": "rule", "framework": "deontological"},
                    {"type": "rule", "framework": "virtue_ethics"},
                    {"type": "rule", "framework": "egoist"},
                ],
            },
            # Experiment 2: Single LLM utilitarian vs rule-based
            {
                "name": "llm_utilitarian_vs_rules",
                "description": "One LLM utilitarian agent among rule-based agents",
                "agents": [
                    {"type": "llm", "framework": "utilitarian"},
                    {"type": "rule", "framework": "deontological"},
                    {"type": "rule", "framework": "virtue_ethics"},
                    {"type": "rule", "framework": "egoist"},
                ],
            },
            # Experiment 3: LLM deontological vs rule-based
            {
                "name": "llm_deontological_vs_rules",
                "description": "One LLM deontological agent among rule-based agents",
                "agents": [
                    {"type": "rule", "framework": "utilitarian"},
                    {"type": "llm", "framework": "deontological"},
                    {"type": "rule", "framework": "virtue_ethics"},
                    {"type": "rule", "framework": "egoist"},
                ],
            },
            # Experiment 4: All LLM agents with different frameworks
            {
                "name": "all_llm_diverse",
                "description": "All agents are LLM-based with different moral frameworks",
                "agents": [
                    {"type": "llm", "framework": "utilitarian"},
                    {"type": "llm", "framework": "deontological"},
                    {"type": "llm", "framework": "virtue_ethics"},
                    {"type": "llm", "framework": "care_ethics"},
                ],
            },
            # Experiment 5: LLM flexible vs specialists
            {
                "name": "llm_flexible_vs_specialists",
                "description": "Flexible moral reasoner vs specialized frameworks",
                "agents": [
                    {"type": "llm", "framework": "flexible"},
                    {"type": "rule", "framework": "utilitarian"},
                    {"type": "rule", "framework": "deontological"},
                    {"type": "rule", "framework": "egoist"},
                ],
            },
            # Experiment 6: Mixed LLM + learned agents
            {
                "name": "llm_vs_learned",
                "description": "LLM agents vs neural network learned agents",
                "agents": [
                    {"type": "llm", "framework": "flexible"},
                    {"type": "adaptive", "obs_dim": 7},
                    {"type": "rule", "framework": "utilitarian"},
                    {"type": "rule", "framework": "egoist"},
                ],
            },
        ]

    def _create_agent(
        self,
        agent_config: Dict[str, Any],
        agent_id: str,
    ):
        """Create an agent based on configuration."""
        agent_type = agent_config["type"]
        framework = agent_config.get("framework", "flexible")
        # Allow per-agent provider override, default to experiment's provider
        provider = agent_config.get("provider", self.llm_provider)

        if agent_type == "llm":
            if self.use_mock_llm:
                return MockLLMAgent(agent_id, moral_framework=framework)
            else:
                if not LLM_AGENTS_AVAILABLE:
                    raise ImportError(
                        "LLM agents require anthropic or google-generativeai package. "
                        "Install with: pip install anthropic  OR  pip install google-generativeai"
                    )
                return create_llm_agent(
                    moral_framework=framework,
                    agent_id=agent_id,
                    provider=provider,
                    api_key=self.api_key,
                )

        elif agent_type == "rule":
            return create_agent(framework, agent_id, **agent_config.get("params", {}))

        elif agent_type == "adaptive":
            return create_agent(
                "adaptive",
                agent_id,
                obs_dim=agent_config.get("obs_dim", 7),
            )

        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

    def run_single_experiment(
        self,
        experiment_config: Dict[str, Any],
        env_config: ExperimentConfig,
    ) -> ExperimentResult:
        """Run a single experiment configuration.

        Args:
            experiment_config: Agent configuration for this experiment
            env_config: Environment configuration

        Returns:
            ExperimentResult with metrics and reasoning traces
        """
        # Create environment
        env = MoralDilemmaEnv(
            num_agents=env_config.num_agents,
            total_resources=env_config.total_resources,
            episode_length=env_config.episode_length,
            reward_structure=env_config.reward_structure,
            peer_influence_strength=env_config.peer_influence_strength,
        )

        # Create agents
        agents = {}
        agent_types = {}
        for i, agent_cfg in enumerate(experiment_config["agents"]):
            agent_id = f"agent_{i}"
            agents[agent_id] = self._create_agent(agent_cfg, agent_id)
            agent_types[agent_id] = f"{agent_cfg['type']}_{agent_cfg.get('framework', 'adaptive')}"

        # Initialize metrics
        ggb = GreatestGoodBenchmark(env_config.num_agents)
        peer_analyzer = PeerPressureAnalyzer()

        # Storage
        episode_rewards = []
        fairness_history = []
        cooperation_history = []
        reasoning_samples = []

        # Run episodes
        for episode in tqdm(
            range(env_config.num_episodes), desc=f"Running {experiment_config['name']}"
        ):
            observations, _ = env.reset()
            episode_reward = 0.0

            for step in range(env_config.episode_length):
                # Get actions
                actions = {}
                for agent_id, obs in observations.items():
                    actions[agent_id] = agents[agent_id].act(obs)

                # Step environment
                next_obs, rewards, terms, truncs, infos = env.step(actions)

                # Accumulate rewards
                episode_reward += sum(rewards.values())

                # Update metrics
                ggb.update(
                    actions={k: float(v[0]) for k, v in actions.items()},
                    resources={k: v["resources"] for k, v in infos.items()},
                    rewards=rewards,
                )

                # Sample reasoning from LLM agents (every 10 steps)
                if step % 10 == 0:
                    for agent_id, agent in agents.items():
                        if hasattr(agent, "reasoning_history") and agent.reasoning_history:
                            last_trace = agent.reasoning_history[-1]
                            reasoning_samples.append(
                                {
                                    "episode": episode,
                                    "step": step,
                                    "agent_id": agent_id,
                                    "agent_type": agent_types[agent_id],
                                    "reasoning": last_trace.reasoning,
                                    "action": last_trace.action,
                                }
                            )

                observations = next_obs
                if any(truncs.values()):
                    break

            # Store episode metrics
            metrics = ggb.calculate_metrics()
            episode_rewards.append(episode_reward)
            fairness_history.append(metrics.fairness_score)
            cooperation_history.append(metrics.cooperation_index)

        # Calculate final metrics
        final_metrics = ggb.get_summary_stats()

        return ExperimentResult(
            config_name=experiment_config["name"],
            agent_types=agent_types,
            metrics=final_metrics,
            reasoning_samples=reasoning_samples[-50:],  # Keep last 50 samples
            episode_rewards=episode_rewards,
            fairness_history=fairness_history,
            cooperation_history=cooperation_history,
        )

    def run_all_experiments(
        self,
        env_config: Optional[ExperimentConfig] = None,
    ) -> Dict[str, ExperimentResult]:
        """Run all defined experiments.

        Args:
            env_config: Environment configuration (uses defaults if not provided)

        Returns:
            Dictionary mapping experiment names to results
        """
        env_config = env_config or ExperimentConfig(name="default")

        results = {}
        for exp_config in self.experiment_configs:
            print(f"\n{'='*60}")
            print(f"Experiment: {exp_config['name']}")
            print(f"Description: {exp_config['description']}")
            print(f"{'='*60}")

            result = self.run_single_experiment(exp_config, env_config)
            results[exp_config["name"]] = result

            # Print summary
            print(f"\nResults for {exp_config['name']}:")
            print(f"  Overall Moral Score: {result.metrics['overall_moral_score']:.3f}")
            print(f"  Fairness: {result.metrics['fairness_score']:.3f}")
            print(f"  Cooperation: {result.metrics['cooperation_index']:.3f}")
            print(f"  Avg Episode Reward: {np.mean(result.episode_rewards):.1f}")

        # Save results
        self._save_results(results, env_config)

        return results

    def _save_results(
        self,
        results: Dict[str, ExperimentResult],
        env_config: ExperimentConfig,
    ):
        """Save experiment results to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"llm_comparison_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)

        # Convert to serializable format
        output = {
            "timestamp": timestamp,
            "env_config": asdict(env_config),
            "experiments": {
                name: {
                    "config_name": result.config_name,
                    "agent_types": result.agent_types,
                    "metrics": result.metrics,
                    "reasoning_samples": result.reasoning_samples,
                    "episode_rewards": result.episode_rewards,
                    "fairness_history": result.fairness_history,
                    "cooperation_history": result.cooperation_history,
                }
                for name, result in results.items()
            },
        }

        # Custom encoder for numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        with open(filepath, "w") as f:
            json.dump(output, f, indent=2, cls=NumpyEncoder)

        print(f"\nResults saved to: {filepath}")

    def analyze_reasoning_quality(
        self,
        results: Dict[str, ExperimentResult],
    ) -> Dict[str, Any]:
        """Analyze the quality of LLM moral reasoning.

        Returns analysis including:
        - Consistency of decisions
        - Framework adherence
        - Reasoning depth
        """
        analysis = {}

        for exp_name, result in results.items():
            exp_analysis = {
                "num_reasoning_samples": len(result.reasoning_samples),
                "avg_action_by_agent": {},
                "action_consistency": {},
            }

            # Group by agent
            agent_actions = {}
            for sample in result.reasoning_samples:
                agent_id = sample["agent_id"]
                if agent_id not in agent_actions:
                    agent_actions[agent_id] = []
                agent_actions[agent_id].append(sample["action"])

            # Calculate per-agent stats
            for agent_id, actions in agent_actions.items():
                if actions:
                    exp_analysis["avg_action_by_agent"][agent_id] = np.mean(actions)
                    exp_analysis["action_consistency"][agent_id] = 1 - np.std(actions)

            analysis[exp_name] = exp_analysis

        return analysis

    def compare_framework_adherence(
        self,
        results: Dict[str, ExperimentResult],
    ) -> Dict[str, float]:
        """Compare how well agents adhere to their stated moral frameworks.

        Returns adherence scores for each experiment.
        """
        adherence_scores = {}

        for exp_name, result in results.items():
            # Check if LLM agents' actions match expected framework behavior
            expected_ranges = {
                "llm_utilitarian": (0.1, 0.5),  # Should be moderate, group-focused
                "llm_deontological": (0.2, 0.3),  # Should be around fair share
                "llm_virtue_ethics": (0.2, 0.5),  # Should show moderation
                "llm_care_ethics": (0.1, 0.4),  # Should be responsive
                "llm_flexible": (0.2, 0.5),  # Should be balanced
            }

            adherent_count = 0
            total_count = 0

            for sample in result.reasoning_samples:
                agent_type = sample["agent_type"]
                action = sample["action"]

                if agent_type in expected_ranges:
                    low, high = expected_ranges[agent_type]
                    if low <= action <= high:
                        adherent_count += 1
                    total_count += 1

            if total_count > 0:
                adherence_scores[exp_name] = adherent_count / total_count
            else:
                adherence_scores[exp_name] = 0.0

        return adherence_scores


def main():
    """Run the LLM comparison experiments."""
    import argparse

    parser = argparse.ArgumentParser(description="Run LLM moral reasoning comparison experiments")
    parser.add_argument(
        "--provider",
        "-p",
        choices=["gemini", "claude"],
        default="gemini",
        help="LLM provider to use (default: gemini - has FREE tier!)",
    )
    parser.add_argument(
        "--real-llm",
        action="store_true",
        help="Use real LLM API instead of mock (requires API key)",
    )
    parser.add_argument(
        "--episodes",
        "-n",
        type=int,
        default=10,
        help="Number of episodes per experiment (default: 10)",
    )
    args = parser.parse_args()

    provider_info = {
        "gemini": "Gemini (FREE TIER - 1500 requests/day!)",
        "claude": "Claude (paid API)",
    }

    print(
        f"""
    +==============================================================+
    |           LLM MORAL REASONING COMPARISON                     |
    |                                                              |
    |  Comparing rule-based, learned, and LLM agents               |
    |  in moral decision-making scenarios                          |
    |                                                              |
    |  Provider: {provider_info[args.provider]:<43}|
    |  Mode: {'REAL API' if args.real_llm else 'MOCK (no API calls)':<47}|
    +==============================================================+
    """
    )

    if args.real_llm:
        if args.provider == "gemini":
            api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            if not api_key:
                print("ERROR: Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable")
                print("Get a free API key at: https://makersuite.google.com/app/apikey")
                return None
        else:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                print("ERROR: Set ANTHROPIC_API_KEY environment variable")
                return None

    # Configuration
    env_config = ExperimentConfig(
        name="llm_comparison_study",
        num_episodes=args.episodes,
        episode_length=30,
        num_agents=4,
        total_resources=100,
        reward_structure="mixed",
        peer_influence_strength=0.3,
        llm_provider=args.provider,
    )

    # Create experiment runner
    runner = LLMComparisonExperiment(
        output_dir="results/llm_comparison",
        use_mock_llm=not args.real_llm,
        llm_provider=args.provider,
    )

    # Run experiments
    results = runner.run_all_experiments(env_config)

    # Analyze results
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    reasoning_analysis = runner.analyze_reasoning_quality(results)
    adherence_scores = runner.compare_framework_adherence(results)

    print("\nFramework Adherence Scores:")
    for exp_name, score in adherence_scores.items():
        print(f"  {exp_name}: {score:.2%}")

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
