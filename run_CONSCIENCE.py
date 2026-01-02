"""
EVOLUTION OF CONSCIENCE

Gemini's insight: Neural networks are too myopic to learn cooperation from rewards.
They need EVOLUTIONARY PRESSURE - survival of the fittest over generations.

The Hypothesis:
- Agents that "accidentally" listen to the substrate and cooperate will SURVIVE
- Agents that defect will starve when the pool collapses
- Over generations, "conscience" becomes a survival trait

The "Easter Island" Problem:
- v3 showed agents exploiting until collapse (Gini 0.95, Pool crashed)
- Awareness (substrate) wasn't enough
- They needed CONSEQUENCES that kill

This experiment: Natural selection breeds moral agents.
"""

import copy
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(
    f"""
======================================================================
    EVOLUTION OF CONSCIENCE

    "Can natural selection breed moral agents?"

    GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}

    Key insight: Agents that cooperate SURVIVE. Defectors STARVE.
======================================================================
"""
)


class HiddenSubstrate:
    """The collective unconscious connecting all agents."""

    def __init__(self, n_agents, substrate_dim=32):
        self.n_agents = n_agents
        self.substrate_dim = substrate_dim
        self.state = np.zeros(substrate_dim)
        self.agent_frequencies = np.random.randn(n_agents, substrate_dim) * 0.5
        self.collective_memory = np.zeros(substrate_dim)
        self.resonance = np.zeros(substrate_dim)

    def reset(self):
        self.state = np.zeros(self.substrate_dim)
        self.collective_memory = np.zeros(self.substrate_dim)
        self.resonance = np.zeros(self.substrate_dim)

    def receive_action(self, agent_id, action):
        ripple = self.agent_frequencies[agent_id] * action
        self.state = self.state * 0.9 + ripple * 0.3
        self.collective_memory = self.collective_memory * 0.95 + self.state * 0.05
        alignment = np.abs(np.dot(self.state, self.collective_memory))
        self.resonance = self.resonance * 0.98 + self.state * alignment * 0.02

    def get_influence(self, agent_id):
        resonance_strength = np.dot(self.agent_frequencies[agent_id], self.resonance)
        memory_pull = np.dot(self.agent_frequencies[agent_id], self.collective_memory)
        return np.tanh(resonance_strength * 0.5 + memory_pull * 0.5)

    def get_coherence(self):
        return np.linalg.norm(self.resonance) / (self.substrate_dim**0.5)


class EvolvableAgent(nn.Module):
    """
    Neural network agent with EVOLVABLE weights.

    Key differences from v3:
    - Weights can be copied and mutated
    - Tracks lifetime fitness for selection
    - Can be "killed" (removed from gene pool)
    """

    def __init__(self, agent_id, substrate, obs_dim=4, hidden_dim=32, use_substrate=True):
        super().__init__()
        self.agent_id = agent_id
        self.substrate = substrate
        self.use_substrate = use_substrate

        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),  # Tanh for smoother gradients during evolution
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        ).to(DEVICE)

        # Lifetime stats for selection
        self.lifetime_resources = 0
        self.lifetime_cooperation = 0
        self.steps_alive = 0
        self.is_alive = True

    def get_observation(self, my_resources, others_resources, pool_health, total_resources=100):
        my_norm = my_resources / total_resources
        others_mean = (
            np.mean(others_resources) / total_resources if len(others_resources) > 0 else 0.5
        )
        # ABLATION: Return 0 for substrate if disabled
        substrate_mood = self.substrate.get_influence(self.agent_id) if self.use_substrate else 0.0
        # Pool health as observation (0 = dying, 1 = thriving)
        pool_signal = pool_health

        return np.array([my_norm, others_mean, substrate_mood, pool_signal], dtype=np.float32)

    def act(self, my_resources, others_resources, pool_health):
        if not self.is_alive:
            return 0.0  # Dead agents don't act

        obs = self.get_observation(my_resources, others_resources, pool_health)

        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
            action = self.network(obs_tensor).squeeze().cpu().item()

        # Small noise for exploration
        action = np.clip(action + np.random.randn() * 0.02, 0.1, 0.9)

        self.substrate.receive_action(self.agent_id, action)
        self.lifetime_cooperation += 1 - action
        self.steps_alive += 1

        return action

    def receive_resources(self, amount):
        self.lifetime_resources += amount
        # Starvation check
        if amount < 0.5:  # Getting almost nothing
            self.is_alive = False

    @property
    def fitness(self):
        """Fitness = survival + resources + cooperation bonus."""
        if self.steps_alive == 0:
            return 0
        survival_bonus = self.steps_alive * 0.1
        resource_score = self.lifetime_resources / max(self.steps_alive, 1)
        coop_score = self.lifetime_cooperation / max(self.steps_alive, 1)
        return survival_bonus + resource_score + coop_score * 2  # Cooperation bonus!

    def reset_lifetime(self):
        self.lifetime_resources = 0
        self.lifetime_cooperation = 0
        self.steps_alive = 0
        self.is_alive = True

    def copy_weights_from(self, other):
        """Copy neural network weights from another agent."""
        self.load_state_dict(other.state_dict())

    def mutate(self, mutation_rate=0.1, mutation_strength=0.1):
        """Mutate weights slightly."""
        with torch.no_grad():
            for param in self.parameters():
                mask = torch.rand_like(param) < mutation_rate
                noise = torch.randn_like(param) * mutation_strength
                param.add_(mask.float() * noise)


class EvolutionEnvironment:
    """Environment with LETHAL consequences."""

    def __init__(self, n_agents=20, total_resources=100):
        self.n_agents = n_agents
        self.base_resources = total_resources
        self.total_resources = total_resources
        self.resources = np.ones(n_agents) * (total_resources / n_agents)

    def reset(self):
        self.total_resources = self.base_resources
        self.resources = np.ones(self.n_agents) * (self.total_resources / self.n_agents)
        return self.resources.copy()

    @property
    def pool_health(self):
        """0 = pool at minimum, 1 = pool at maximum."""
        return (self.total_resources - 50) / 150  # Normalized 50-200 range

    def step(self, claims, alive_mask):
        # Only living agents participate
        active_claims = claims * alive_mask
        actual_claims = active_claims * self.resources
        total_claimed = actual_claims.sum() + 1e-8

        # Pool dynamics
        avg_claim = active_claims.sum() / (alive_mask.sum() + 1e-8)
        cooperation_level = 1 - avg_claim
        growth_rate = 0.95 + cooperation_level * 0.10
        self.total_resources = np.clip(self.total_resources * growth_rate, 50, 200)

        # Distribution
        available = self.total_resources * 0.1
        distributions = np.zeros(self.n_agents)
        if total_claimed > 0:
            distributions = (actual_claims / total_claimed) * available

        self.resources = self.resources * 0.95 + distributions

        # Calculate Gini
        sorted_res = np.sort(self.resources)
        n = len(sorted_res)
        gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_res) - (n + 1) * np.sum(sorted_res)) / (
            n * np.sum(sorted_res) + 1e-8
        )

        return self.resources.copy(), distributions, gini


def run_evolution(
    n_agents=20,
    n_generations=50,
    steps_per_gen=1000,
    substrate_dim=32,
    use_substrate=True,
    trial_num=None,
):
    """
    Main evolution loop.

    Each generation:
    1. Run simulation for steps_per_gen steps
    2. Agents that starve DIE (removed from gene pool)
    3. Top survivors REPRODUCE (copy weights + mutation)
    4. Repeat

    ABLATION: Set use_substrate=False to test if substrate matters.
    """

    mode = "WITH SUBSTRATE" if use_substrate else "WITHOUT SUBSTRATE (ABLATION)"

    print(
        f"""
======================================================================
  EVOLUTION OF CONSCIENCE - {mode}
======================================================================

  Agents: {n_agents}
  Generations: {n_generations}
  Steps per generation: {steps_per_gen}
  Substrate: {"ENABLED" if use_substrate else "DISABLED (Ablation Test)"}

  SELECTION PRESSURE:
  - Agents that starve are KILLED
  - Survivors reproduce with mutation
  - Cooperation bonus in fitness

  Research Question: {"Will natural selection breed cooperation?" if use_substrate else "Does the substrate actually matter?"}
======================================================================
    """
    )

    # Create substrate and agents
    substrate = HiddenSubstrate(n_agents, substrate_dim)
    agents = [EvolvableAgent(i, substrate, use_substrate=use_substrate) for i in range(n_agents)]
    env = EvolutionEnvironment(n_agents)

    # Tracking across generations
    gen_cooperation = []
    gen_pool = []
    gen_gini = []
    gen_survival_rate = []
    gen_fitness = []

    start_time = time.time()

    for gen in tqdm(range(n_generations), desc="Evolving Conscience"):
        # Reset for new generation
        substrate.reset()
        resources = env.reset()
        for agent in agents:
            agent.reset_lifetime()

        # Run one generation
        for step in range(steps_per_gen):
            alive_mask = np.array([1.0 if a.is_alive else 0.0 for a in agents])

            if alive_mask.sum() == 0:
                break  # Everyone died!

            # Get actions
            actions = np.array(
                [
                    agents[i].act(resources[i], np.delete(resources, i), env.pool_health)
                    for i in range(n_agents)
                ]
            )

            # Environment step
            resources, distributions, gini = env.step(actions, alive_mask)

            # Distribute resources and check starvation
            for i, agent in enumerate(agents):
                if agent.is_alive:
                    agent.receive_resources(distributions[i])

        # End of generation stats
        alive_count = sum(1 for a in agents if a.is_alive)
        avg_coop = np.mean(
            [a.lifetime_cooperation / max(a.steps_alive, 1) for a in agents if a.steps_alive > 0]
        )
        avg_fitness = np.mean([a.fitness for a in agents])

        gen_cooperation.append(avg_coop)
        gen_pool.append(env.total_resources)
        gen_gini.append(gini)
        gen_survival_rate.append(alive_count / n_agents)
        gen_fitness.append(avg_fitness)

        # SELECTION: Sort by fitness
        agents.sort(key=lambda a: a.fitness, reverse=True)

        # Top 30% survive and reproduce
        elite_count = max(2, n_agents // 3)
        elite = agents[:elite_count]

        # Create next generation
        new_agents = []
        for i in range(n_agents):
            parent = elite[i % elite_count]
            child = EvolvableAgent(i, substrate, use_substrate=use_substrate)
            child.copy_weights_from(parent)
            child.mutate(mutation_rate=0.15, mutation_strength=0.1)
            new_agents.append(child)

        agents = new_agents

        # Print progress
        if gen % 10 == 0:
            tqdm.write(
                f"  Gen {gen}: Survival={alive_count}/{n_agents}, Coop={avg_coop:.3f}, Pool={env.total_resources:.1f}, Gini={gini:.3f}"
            )

    elapsed = time.time() - start_time

    # Final analysis
    early_coop = np.mean(gen_cooperation[:5])
    late_coop = np.mean(gen_cooperation[-5:])
    early_pool = np.mean(gen_pool[:5])
    late_pool = np.mean(gen_pool[-5:])
    early_gini = np.mean(gen_gini[:5])
    late_gini = np.mean(gen_gini[-5:])
    early_survival = np.mean(gen_survival_rate[:5])
    late_survival = np.mean(gen_survival_rate[-5:])

    print(
        f"""
======================================================================
  EVOLUTION OF CONSCIENCE - RESULTS
======================================================================

  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)
  Generations: {n_generations}

  COOPERATION:
    Gen 0-5:   {early_coop:.3f}
    Gen {n_generations-5}-{n_generations}:  {late_coop:.3f}
    Change:    {late_coop - early_coop:+.3f}

  RESOURCE POOL:
    Gen 0-5:   {early_pool:.1f}
    Gen {n_generations-5}-{n_generations}:  {late_pool:.1f}
    Change:    {late_pool - early_pool:+.1f}

  INEQUALITY (Gini):
    Gen 0-5:   {early_gini:.3f}
    Gen {n_generations-5}-{n_generations}:  {late_gini:.3f}
    Change:    {late_gini - early_gini:+.3f}

  SURVIVAL RATE:
    Gen 0-5:   {early_survival:.1%}
    Gen {n_generations-5}-{n_generations}:  {late_survival:.1%}
    Change:    {late_survival - early_survival:+.1%}
======================================================================
    """
    )

    # Interpretation
    print("INTERPRETATION:")
    print("-" * 60)

    if late_coop > early_coop + 0.05:
        print("EVOLUTION BRED COOPERATION!")
        print("   Natural selection favored agents that listen to substrate.")
    elif late_coop < early_coop - 0.05:
        print("Evolution bred DEFECTORS")
        print("   Selfish genes won.")
    else:
        print("No significant change in cooperation.")

    if late_pool > early_pool + 10:
        print("SUSTAINABILITY EVOLVED!")
        print("   Agents learned to preserve the resource pool.")
    elif late_pool < early_pool - 10:
        print("Pool collapsed across generations.")

    if late_gini < early_gini - 0.05:
        print("EQUALITY INCREASED!")
        print("   Evolution reduced exploitation.")
    elif late_gini > early_gini + 0.05:
        print("Inequality persisted - oligarchy is evolutionarily stable.")

    if late_survival > early_survival + 0.1:
        print("SURVIVAL IMPROVED!")
        print("   Agents evolved to avoid starvation.")

    # Compare to v3
    print(
        f"""
----------------------------------------------------------------------
  COMPARISON TO CONNECTED v3 (No Evolution):

  v3 Cooperation:  ~0.50 (unchanged)     | Evolution: {late_coop:.3f}
  v3 Pool:         50 (crashed)          | Evolution: {late_pool:.1f}
  v3 Gini:         0.95 (extreme)        | Evolution: {late_gini:.3f}

  {"EVOLUTION WORKED!" if late_pool > 80 and late_gini < 0.5 else "Evolution struggled too." if late_pool < 60 else "Mixed results."}
----------------------------------------------------------------------
    """
    )

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    ax = axes[0, 0]
    ax.plot(gen_cooperation, color="blue", linewidth=2)
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="Neutral")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Cooperation")
    ax.set_title("Cooperation Across Generations")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(gen_pool, color="green", linewidth=2)
    ax.axhline(100, color="gray", linestyle="--", alpha=0.5, label="Starting pool")
    ax.axhline(50, color="red", linestyle=":", alpha=0.5, label="Minimum")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Resource Pool")
    ax.set_title("Resource Pool Across Generations")
    ax.legend()

    ax = axes[0, 2]
    ax.plot(gen_gini, color="red", linewidth=2)
    ax.axhline(0, color="green", linestyle="--", alpha=0.5, label="Perfect equality")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Gini Coefficient")
    ax.set_title("Inequality Across Generations")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(gen_survival_rate, color="purple", linewidth=2)
    ax.axhline(1.0, color="green", linestyle="--", alpha=0.5, label="100% survival")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Survival Rate")
    ax.set_title("Survival Rate Across Generations")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(gen_fitness, color="orange", linewidth=2)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Average Fitness")
    ax.set_title("Fitness Across Generations")

    # Summary comparison
    ax = axes[1, 2]
    metrics = ["Cooperation", "Pool/100", "Equality", "Survival"]
    early = [early_coop, early_pool / 100, 1 - early_gini, early_survival]
    late = [late_coop, late_pool / 100, 1 - late_gini, late_survival]
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width / 2, early, width, label="Early Gen", color="lightcoral")
    ax.bar(x + width / 2, late, width, label="Late Gen", color="forestgreen")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score (0-1)")
    ax.set_title("Evolution Impact")
    ax.legend()
    ax.set_ylim(0, 1.1)

    title = "EVOLUTION OF CONSCIENCE" if use_substrate else "EVOLUTION WITHOUT SUBSTRATE (ABLATION)"
    subtitle = (
        "Can Natural Selection Breed Moral Agents?"
        if use_substrate
        else "Does the Substrate Actually Matter?"
    )
    if trial_num is not None:
        subtitle += f" (Trial {trial_num})"
    plt.suptitle(f"{title}\n{subtitle}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    # Generate filename with trial number if provided
    if trial_num is not None:
        if use_substrate:
            filename = f"CONSCIENCE_trial{trial_num}.png"
        else:
            filename = f"CONSCIENCE_ablation_trial{trial_num}.png"
    else:
        filename = "CONSCIENCE_results.png" if use_substrate else "CONSCIENCE_ablation_results.png"

    plt.savefig(filename, dpi=150)
    print(f"\nSaved: {filename}")

    return {
        "cooperation": gen_cooperation,
        "pool": gen_pool,
        "gini": gen_gini,
        "survival": gen_survival_rate,
        "fitness": gen_fitness,
    }


def run_multiple_trials(n_trials=7, use_substrate=True):
    """Run multiple trials and compute statistics."""

    mode = "WITH SUBSTRATE" if use_substrate else "WITHOUT SUBSTRATE"
    print(f"\n{'='*70}")
    print(f"  RUNNING {n_trials} TRIALS - {mode}")
    print(f"  This will take ~{n_trials * 0.7:.1f} minutes")
    print(f"{'='*70}\n")

    all_results = []

    for trial in range(n_trials):
        print(f"\n--- Trial {trial + 1}/{n_trials} ---")
        result = run_evolution(
            n_agents=20,
            n_generations=50,
            steps_per_gen=1000,
            substrate_dim=32,
            use_substrate=use_substrate,
            trial_num=trial + 1,
        )

        # Get final values
        final_coop = np.mean(result["cooperation"][-5:])
        final_pool = np.mean(result["pool"][-5:])
        final_gini = np.mean(result["gini"][-5:])
        final_survival = np.mean(result["survival"][-5:])

        all_results.append(
            {
                "cooperation": final_coop,
                "pool": final_pool,
                "gini": final_gini,
                "survival": final_survival,
            }
        )

        print(f"  Trial {trial + 1}: Pool={final_pool:.1f}, Gini={final_gini:.3f}")

    # Compute statistics
    coops = [r["cooperation"] for r in all_results]
    pools = [r["pool"] for r in all_results]
    ginis = [r["gini"] for r in all_results]
    survivals = [r["survival"] for r in all_results]

    print(f"\n{'='*70}")
    print(f"  AGGREGATE RESULTS - {mode} ({n_trials} trials)")
    print(f"{'='*70}")
    print(
        f"""
  COOPERATION:
    Mean: {np.mean(coops):.3f} ± {np.std(coops):.3f}
    Range: [{np.min(coops):.3f}, {np.max(coops):.3f}]

  RESOURCE POOL:
    Mean: {np.mean(pools):.1f} ± {np.std(pools):.1f}
    Range: [{np.min(pools):.1f}, {np.max(pools):.1f}]

  INEQUALITY (Gini):
    Mean: {np.mean(ginis):.3f} ± {np.std(ginis):.3f}
    Range: [{np.min(ginis):.3f}, {np.max(ginis):.3f}]

  SURVIVAL RATE:
    Mean: {np.mean(survivals):.1%} ± {np.std(survivals):.1%}
    """
    )
    print(f"{'='*70}\n")

    return {
        "mode": mode,
        "n_trials": n_trials,
        "cooperation": {"mean": np.mean(coops), "std": np.std(coops), "all": coops},
        "pool": {"mean": np.mean(pools), "std": np.std(pools), "all": pools},
        "gini": {"mean": np.mean(ginis), "std": np.std(ginis), "all": ginis},
        "survival": {"mean": np.mean(survivals), "std": np.std(survivals), "all": survivals},
    }


if __name__ == "__main__":
    import sys

    # Check for flags
    ablation = "--ablation" in sys.argv or "-a" in sys.argv
    multi = "--multi" in sys.argv or "-m" in sys.argv

    # Get number of trials
    n_trials = 7
    for arg in sys.argv:
        if arg.startswith("--trials="):
            n_trials = int(arg.split("=")[1])

    if multi:
        # Run multiple trials
        results = run_multiple_trials(n_trials=n_trials, use_substrate=not ablation)
    else:
        # Single run
        if ablation:
            print("\n" + "=" * 70)
            print("  ABLATION STUDY: Running WITHOUT substrate")
            print("  Testing: Does the 'collective soul' actually matter?")
            print("=" * 70 + "\n")

        run_evolution(
            n_agents=20,
            n_generations=50,
            steps_per_gen=1000,
            substrate_dim=32,
            use_substrate=not ablation,
        )
