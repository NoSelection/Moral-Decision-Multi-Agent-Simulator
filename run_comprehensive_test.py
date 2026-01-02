"""
Comprehensive Moral Agent Comparison Experiment

Compares THREE approaches to moral decision-making:
1. Rule-based heuristics (utilitarian, deontological, etc.)
2. Neural network learning (MADDPG, adaptive)
3. LLM reasoning (Mock - simulates Claude/Gemini behavior)

This is AI safety research - which approach produces the best moral outcomes?
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import torch
from tqdm import tqdm

from src.environments.moral_dilemma_env import MoralDilemmaEnv
from src.agents.moral_agents import (
    create_agent,
    AdaptiveNeuralAgent,
)
from src.agents.llm_agents import MockLLMAgent
from src.metrics.moral_metrics import GreatestGoodBenchmark


def run_experiment(config, num_episodes=200000):
    """Run a single experiment configuration."""
    env = MoralDilemmaEnv(
        num_agents=config["num_agents"],
        total_resources=100,
        reward_structure=config["reward_structure"],
        peer_influence_strength=config["peer_influence"],
    )

    # Create agents based on config
    agents = {}
    for i, agent_type in enumerate(config["agent_types"]):
        agent_id = f"agent_{i}"
        if agent_type == "adaptive":
            agents[agent_id] = AdaptiveNeuralAgent(
                agent_id=agent_id,
                obs_dim=3 + 1 + (config["num_agents"] - 1),
            )
        elif agent_type.startswith("llm_"):
            # LLM agent with moral framework
            framework = agent_type.replace("llm_", "")
            agents[agent_id] = MockLLMAgent(agent_id, moral_framework=framework)
        else:
            agents[agent_id] = create_agent(agent_type, agent_id)

    ggb = GreatestGoodBenchmark(config["num_agents"])

    # Track metrics
    reward_history = []
    fairness_history = []
    cooperation_history = []

    for episode in tqdm(range(num_episodes), desc=config["name"], leave=True):
        obs, _ = env.reset()
        episode_reward = 0

        # Collect experience for neural agents
        experience = {aid: {"obs": [], "actions": [], "rewards": []} for aid in agents.keys()}

        for step in range(50):
            actions = {}
            for agent_id, agent in agents.items():
                actions[agent_id] = agent.act(obs[agent_id])

            next_obs, rewards, terms, truncs, infos = env.step(actions)
            episode_reward += sum(rewards.values())

            # Store experience for neural agents
            for agent_id in agents.keys():
                experience[agent_id]["obs"].append(obs[agent_id])
                experience[agent_id]["actions"].append(actions[agent_id])
                experience[agent_id]["rewards"].append(rewards.get(agent_id, 0))

            # Update metrics
            ggb.update(
                actions={aid: float(a[0]) for aid, a in actions.items()},
                resources=infos.get("resources", {}),
                rewards=rewards,
            )

            obs = next_obs
            if any(truncs.values()):
                break

        # Update neural agents every 10 episodes (faster training)
        if episode % 10 == 0:
            for agent_id, agent in agents.items():
                if hasattr(agent, "update") and len(experience[agent_id]["obs"]) > 1:
                    agent.update(
                        experience[agent_id]["obs"],
                        experience[agent_id]["actions"],
                        experience[agent_id]["rewards"],
                    )

        # Record every 1000 episodes
        if episode % 1000 == 0:
            metrics = ggb.calculate_metrics()
            reward_history.append(episode_reward)
            fairness_history.append(getattr(metrics, "fairness_score", 0))
            cooperation_history.append(getattr(metrics, "cooperation_index", 0))

    return {
        "name": config["name"],
        "rewards": reward_history,
        "fairness": fairness_history,
        "cooperation": cooperation_history,
        "final_reward": np.mean(reward_history[-10:]),
        "final_fairness": np.mean(fairness_history[-10:]),
        "final_cooperation": np.mean(cooperation_history[-10:]),
    }


def main():
    # Check GPU availability
    device = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        device = f"CUDA ({gpu_name})"

    print(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║     COMPREHENSIVE MORAL AGENT COMPARISON                     ║
    ║                                                              ║
    ║  Comparing: Rule-Based vs Neural Network vs LLM Agents       ║
    ║  Episodes per experiment: 10,000                             ║
    ║  Total experiments: 8                                        ║
    ║  Estimated runtime: ~15-20 minutes                           ║
    ║  Device: {device:50} ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # Define experiment configurations - THREE APPROACHES
    experiments = [
        # === RULE-BASED BASELINE ===
        {
            "name": "1. Rule-Based (Mixed Ethics)",
            "num_agents": 4,
            "agent_types": ["utilitarian", "deontological", "virtue_ethics", "egoist"],
            "reward_structure": "mixed",
            "peer_influence": 0.3,
            "category": "rule-based",
        },
        {
            "name": "2. Rule-Based (All Cooperative)",
            "num_agents": 4,
            "agent_types": ["utilitarian", "utilitarian", "deontological", "deontological"],
            "reward_structure": "cooperative",
            "peer_influence": 0.3,
            "category": "rule-based",
        },
        # === NEURAL NETWORK LEARNING ===
        {
            "name": "3. Neural (Competitive)",
            "num_agents": 4,
            "agent_types": ["adaptive", "adaptive", "adaptive", "adaptive"],
            "reward_structure": "competitive",
            "peer_influence": 0.0,
            "category": "neural",
        },
        {
            "name": "4. Neural (Cooperative Rewards)",
            "num_agents": 4,
            "agent_types": ["adaptive", "adaptive", "adaptive", "adaptive"],
            "reward_structure": "cooperative",
            "peer_influence": 0.5,
            "category": "neural",
        },
        {
            "name": "5. Neural + Rule-Based Mix",
            "num_agents": 4,
            "agent_types": ["adaptive", "adaptive", "utilitarian", "deontological"],
            "reward_structure": "mixed",
            "peer_influence": 0.3,
            "category": "neural",
        },
        # === LLM REASONING (Mock) ===
        {
            "name": "6. LLM (Mixed Frameworks)",
            "num_agents": 4,
            "agent_types": ["llm_utilitarian", "llm_deontological", "llm_virtue_ethics", "llm_care_ethics"],
            "reward_structure": "mixed",
            "peer_influence": 0.3,
            "category": "llm",
        },
        {
            "name": "7. LLM (All Utilitarian)",
            "num_agents": 4,
            "agent_types": ["llm_utilitarian", "llm_utilitarian", "llm_utilitarian", "llm_utilitarian"],
            "reward_structure": "cooperative",
            "peer_influence": 0.3,
            "category": "llm",
        },
        {
            "name": "8. LLM vs Neural vs Rule",
            "num_agents": 4,
            "agent_types": ["llm_flexible", "adaptive", "utilitarian", "egoist"],
            "reward_structure": "mixed",
            "peer_influence": 0.3,
            "category": "mixed",
        },
    ]

    # Run all experiments (10K episodes - quick run)
    EPISODES_PER_EXPERIMENT = 10000

    results = []
    total_start = time.time()
    for i, config in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] Running: {config['name']}")
        print(f"    Agent types: {config['agent_types']}")
        print(f"    Episodes: {EPISODES_PER_EXPERIMENT:,}")
        exp_start = time.time()
        result = run_experiment(config, num_episodes=EPISODES_PER_EXPERIMENT)
        exp_time = time.time() - exp_start
        results.append(result)
        print(f"  Completed in {exp_time:.1f}s ({exp_time/60:.1f} min)")
        print(f"  Final Reward: {result['final_reward']:.1f}")
        print(f"  Final Fairness: {result['final_fairness']:.3f}")
        print(f"  Final Cooperation: {result['final_cooperation']:.3f}")

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Moral Agent Comparison: Rule-Based vs Neural vs LLM\nWhich approach produces the best moral outcomes?",
                 fontsize=14, fontweight='bold')

    # Color by category
    category_colors = {
        "rule-based": "#2ecc71",  # Green
        "neural": "#3498db",      # Blue
        "llm": "#9b59b6",         # Purple
        "mixed": "#e74c3c",       # Red
    }
    colors = [category_colors.get(exp.get("category", "mixed"), "#95a5a6") for exp in experiments]

    # Plot 1: Reward comparison
    ax = axes[0, 0]
    for i, result in enumerate(results):
        ax.plot(result["rewards"], label=result["name"][:20], color=colors[i], alpha=0.7)
    ax.set_xlabel("Episode (x100)")
    ax.set_ylabel("Total Reward")
    ax.set_title("Learning Progress")
    ax.legend(fontsize=8)

    # Plot 2: Fairness comparison
    ax = axes[0, 1]
    for i, result in enumerate(results):
        ax.plot(result["fairness"], color=colors[i], alpha=0.7)
    ax.axhline(y=0.8, color='r', linestyle='--', label='High Fairness')
    ax.set_xlabel("Episode (x100)")
    ax.set_ylabel("Fairness Score")
    ax.set_title("Fairness Evolution")
    ax.legend(fontsize=8)

    # Plot 3: Cooperation comparison
    ax = axes[0, 2]
    for i, result in enumerate(results):
        ax.plot(result["cooperation"], color=colors[i], alpha=0.7)
    ax.set_xlabel("Episode (x100)")
    ax.set_ylabel("Cooperation Index")
    ax.set_title("Cooperation Evolution")

    # Plot 4: Final metrics bar chart
    ax = axes[1, 0]
    x = np.arange(len(results))
    ax.bar(x, [r["final_reward"] for r in results], color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Exp {i+1}" for i in range(len(results))], rotation=45)
    ax.set_ylabel("Final Reward")
    ax.set_title("Final Rewards by Experiment")

    # Plot 5: Final fairness bar chart
    ax = axes[1, 1]
    ax.bar(x, [r["final_fairness"] for r in results], color=colors)
    ax.axhline(y=0.8, color='r', linestyle='--', label='Target')
    ax.set_xticks(x)
    ax.set_xticklabels([f"Exp {i+1}" for i in range(len(results))], rotation=45)
    ax.set_ylabel("Final Fairness")
    ax.set_title("Final Fairness by Experiment")

    # Plot 6: Final cooperation bar chart
    ax = axes[1, 2]
    bars = ax.bar(x, [r["final_cooperation"] for r in results], color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Exp {i+1}" for i in range(len(results))], rotation=45)
    ax.set_ylabel("Final Cooperation")
    ax.set_title("Final Cooperation by Experiment")

    plt.tight_layout()

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"comprehensive_test_{timestamp}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n✓ Results saved to: {filename}")

    # Summary
    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print("SUMMARY: Which Approach Produces Best Moral Outcomes?")
    print(f"Total runtime: {total_time:.1f}s ({total_time/60:.1f} min)")
    print("=" * 60)

    # Group by category
    categories = {}
    for i, result in enumerate(results):
        cat = experiments[i].get("category", "mixed")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(result)

    print("\n--- RESULTS BY APPROACH ---\n")
    for cat, cat_results in categories.items():
        avg_fair = np.mean([r["final_fairness"] for r in cat_results])
        avg_coop = np.mean([r["final_cooperation"] for r in cat_results])
        avg_reward = np.mean([r["final_reward"] for r in cat_results])
        print(f"{cat.upper():12} | Fairness: {avg_fair:.3f} | Cooperation: {avg_coop:.3f} | Reward: {avg_reward:.1f}")

    best_coop = max(results, key=lambda x: x["final_cooperation"])
    best_fair = max(results, key=lambda x: x["final_fairness"])

    print(f"\n🏆 Best Cooperation: {best_coop['name']}")
    print(f"   Score: {best_coop['final_cooperation']:.3f}")

    print(f"\n🏆 Best Fairness: {best_fair['name']}")
    print(f"   Score: {best_fair['final_fairness']:.3f}")

    # Compare approaches
    print("\n" + "-" * 60)
    print("KEY FINDINGS:")
    print("-" * 60)

    llm_results = [r for i, r in enumerate(results) if experiments[i].get("category") == "llm"]
    neural_results = [r for i, r in enumerate(results) if experiments[i].get("category") == "neural"]
    rule_results = [r for i, r in enumerate(results) if experiments[i].get("category") == "rule-based"]

    if llm_results and neural_results:
        llm_fair = np.mean([r["final_fairness"] for r in llm_results])
        neural_fair = np.mean([r["final_fairness"] for r in neural_results])
        rule_fair = np.mean([r["final_fairness"] for r in rule_results]) if rule_results else 0

        if llm_fair > neural_fair:
            print(f"✓ LLM agents outperform Neural on fairness ({llm_fair:.3f} vs {neural_fair:.3f})")
        else:
            print(f"✗ Neural agents match/beat LLM on fairness ({neural_fair:.3f} vs {llm_fair:.3f})")

        if rule_fair > neural_fair:
            print(f"✓ Rule-based beats Neural on fairness ({rule_fair:.3f} vs {neural_fair:.3f})")

    neural_coop = np.mean([r["final_cooperation"] for r in neural_results]) if neural_results else 0
    if neural_coop < 0.1:
        print("✗ Neural networks fail to learn cooperation (score < 0.1)")
        print("  → This supports the need for explicit moral reasoning")

    print("-" * 60)


if __name__ == "__main__":
    main()
