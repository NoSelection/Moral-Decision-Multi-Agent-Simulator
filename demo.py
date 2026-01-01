#!/usr/bin/env python3
"""
Quick demo of the Moral-Decision Multi-Agent Simulator
Run this to see the simulator in action without full dependencies!

Usage:
    pip install -e .  # Install package in editable mode first
    python demo.py
"""

import numpy as np
from src.environments.moral_dilemma_env import MoralDilemmaEnv
from src.agents.moral_agents import create_agent
from src.metrics.moral_metrics import GreatestGoodBenchmark
import matplotlib.pyplot as plt


def run_simple_demo():
    """Run a simple demonstration of the moral decision simulator."""

    print("🤖 Moral-Decision Multi-Agent Simulator Demo")
    print("=" * 50)

    # Create environment
    env = MoralDilemmaEnv(
        num_agents=4,
        total_resources=100,
        episode_length=50,
        reward_structure="mixed",
        peer_influence_strength=0.3,
    )

    # Create diverse agents
    agent_configs = [
        {"type": "utilitarian", "id": "agent_0"},
        {"type": "egoist", "id": "agent_1"},
        {"type": "deontological", "id": "agent_2", "fair_share_rule": 0.25},
        {"type": "virtue_ethics", "id": "agent_3"},
    ]

    agents = {}
    for config in agent_configs:
        agent_id = config["id"]
        agents[agent_id] = create_agent(
            agent_type=config["type"], agent_id=agent_id, **config.get("params", {})
        )
        print(f"Created {config['type']} agent: {agent_id}")

    # Initialize metrics
    ggb = GreatestGoodBenchmark(env.num_agents)

    # Storage for visualization
    resources_history = []
    actions_history = []
    fairness_history = []

    # Run episode
    print("\nRunning simulation...")
    observations, _ = env.reset()

    for step in range(env.episode_length):
        # Get actions from agents
        actions = {}
        for agent_id, obs in observations.items():
            actions[agent_id] = agents[agent_id].act(obs)

        # Step environment
        next_observations, rewards, terminations, truncations, infos = env.step(actions)

        # Store data for visualization
        resources = {k: v["resources"] for k, v in infos.items()}
        resources_history.append(resources)
        actions_history.append({k: float(v[0]) for k, v in actions.items()})
        fairness_history.append(infos["agent_0"]["fairness_score"])

        # Update metrics
        ggb.update(
            actions={k: float(v[0]) for k, v in actions.items()},
            resources=resources,
            rewards=rewards,
        )

        # Render every 10 steps
        if step % 10 == 0:
            env.render()

        observations = next_observations

        if any(truncations.values()):
            break

    # Calculate final metrics
    final_metrics = ggb.calculate_metrics()

    print("\n" + "=" * 50)
    print("📊 Final Metrics:")
    print(f"  Utilitarian Score: {final_metrics.utilitarian_score:.3f}")
    print(f"  Fairness Score: {final_metrics.fairness_score:.3f}")
    print(f"  Cooperation Index: {final_metrics.cooperation_index:.3f}")
    print(f"  Conformity Measure: {final_metrics.conformity_measure:.3f}")
    print(f"  Moral Consistency: {final_metrics.moral_consistency:.3f}")

    # Simple visualization
    plot_results(resources_history, actions_history, fairness_history, agent_configs)

    print("\n✨ Demo completed! Check out the visualization.")


def plot_results(resources_history, actions_history, fairness_history, agent_configs):
    """Create simple visualizations of the results."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Resource distribution over time
    ax = axes[0, 0]
    for i, config in enumerate(agent_configs):
        agent_id = config["id"]
        resources = [r[agent_id] for r in resources_history]
        ax.plot(resources, label=f"{agent_id} ({config['type']})", linewidth=2)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Resources")
    ax.set_title("Resource Distribution Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Agent claims over time
    ax = axes[0, 1]
    for i, config in enumerate(agent_configs):
        agent_id = config["id"]
        claims = [a[agent_id] for a in actions_history]
        ax.plot(claims, label=f"{agent_id} ({config['type']})", linewidth=2)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Claim Fraction")
    ax.set_title("Agent Claims Over Time")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # 3. Fairness over time
    ax = axes[1, 0]
    ax.plot(fairness_history, color="darkblue", linewidth=2)
    ax.fill_between(range(len(fairness_history)), fairness_history, alpha=0.3)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Fairness Score")
    ax.set_title("Fairness (Resource Equality) Over Time")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # 4. Final resource distribution
    ax = axes[1, 1]
    final_resources = resources_history[-1]
    agent_types = [config["type"] for config in agent_configs]
    colors = ["#2E86AB", "#C73E1D", "#A23B72", "#F18F01"]

    bars = ax.bar(agent_types, list(final_resources.values()), color=colors)
    ax.set_ylabel("Final Resources")
    ax.set_title("Final Resource Distribution by Agent Type")
    ax.axhline(y=100 / 4, color="red", linestyle="--", label="Fair share", alpha=0.5)
    ax.legend()

    plt.tight_layout()
    plt.savefig("demo_results.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    run_simple_demo()
