"""
EVOLUTIONARY MORAL AGENTS

Can evolution discover cooperation without any training data or gradients?
Nature did it - altruism evolved because it helped groups survive.

This uses genetic algorithms:
1. Create population of agents with random "DNA" (decision parameters)
2. Run them in the environment
3. Select the fittest
4. Breed + mutate to create next generation
5. Repeat for many generations

Hypothesis: Evolution will discover cooperation faster than gradient descent!
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(
    f"""
======================================================================
    EVOLUTIONARY MORAL AGENTS

    GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}

    "Can evolution discover morality?"
======================================================================
"""
)


class EvolvedAgent:
    """
    An agent with evolvable "DNA" that determines its moral behavior.

    DNA consists of:
    - base_cooperation: baseline tendency to share (0-1)
    - reciprocity: how much to match others' behavior (0-1)
    - greed_threshold: when to get greedy if resources are scarce (0-1)
    - forgiveness: how quickly to forgive selfish neighbors (0-1)
    """

    def __init__(self, dna=None):
        if dna is None:
            # Random DNA
            self.dna = np.random.rand(4)
        else:
            self.dna = np.array(dna)

        # Unpack DNA
        self.base_cooperation = self.dna[0]
        self.reciprocity = self.dna[1]
        self.greed_threshold = self.dna[2]
        self.forgiveness = self.dna[3]

        self.fitness = 0
        self.cooperation_score = 0

    def act(self, my_resources, others_mean, others_actions_mean):
        """Decide how much to claim based on DNA."""

        # Start with base cooperation (higher = less greedy)
        claim = 1.0 - self.base_cooperation

        # Reciprocity: match what others are doing
        if others_actions_mean is not None:
            claim = claim * (1 - self.reciprocity) + others_actions_mean * self.reciprocity

        # Greed response: if resources are low, might get greedy
        if my_resources < self.greed_threshold * 30:  # Below threshold
            claim = min(1.0, claim + (1 - self.forgiveness) * 0.3)

        return np.clip(claim, 0.1, 0.9)

    def mutate(self, mutation_rate=0.1, mutation_strength=0.2):
        """Create mutated copy."""
        new_dna = self.dna.copy()

        for i in range(len(new_dna)):
            if np.random.rand() < mutation_rate:
                new_dna[i] += np.random.randn() * mutation_strength
                new_dna[i] = np.clip(new_dna[i], 0, 1)

        return EvolvedAgent(new_dna)

    @staticmethod
    def crossover(parent1, parent2):
        """Create child from two parents."""
        # Random crossover point
        crossover_point = np.random.randint(1, 4)
        child_dna = np.concatenate([parent1.dna[:crossover_point], parent2.dna[crossover_point:]])
        return EvolvedAgent(child_dna)


class VectorizedEvolutionEnv:
    """Fast environment for evaluating many agents."""

    def __init__(self, n_agents=4, total_resources=100):
        self.n_agents = n_agents
        self.total_resources = total_resources
        self.resources = None
        self.reset()

    def reset(self):
        self.resources = np.ones(self.n_agents) * (self.total_resources / self.n_agents)
        return self.resources.copy()

    def step(self, claims):
        """
        claims: array of claim fractions [0-1] for each agent
        """
        # Calculate actual claims
        actual_claims = claims * self.resources
        total_claimed = actual_claims.sum()

        # Distribute new resources proportionally
        available = self.total_resources * 0.1
        if total_claimed > 0:
            distributions = (actual_claims / total_claimed) * available
        else:
            distributions = np.ones(self.n_agents) * (available / self.n_agents)

        # Update resources
        self.resources = self.resources * 0.95 + distributions

        # Rewards: mix of individual and group welfare
        individual_reward = self.resources
        group_reward = self.resources.mean()
        rewards = 0.5 * individual_reward + 0.5 * group_reward

        return self.resources.copy(), rewards


def evaluate_population(population, n_episodes=50, n_steps=20):
    """Evaluate fitness of all agents in population."""

    n_agents = 4  # Agents per game
    env = VectorizedEvolutionEnv(n_agents=n_agents)

    # Reset fitness
    for agent in population:
        agent.fitness = 0
        agent.cooperation_score = 0

    # Run many episodes with random groupings
    for _ in range(n_episodes):
        # Random groups of 4
        np.random.shuffle(population)

        for group_start in range(0, len(population) - n_agents + 1, n_agents):
            group = population[group_start : group_start + n_agents]

            resources = env.reset()
            prev_actions = None
            total_rewards = np.zeros(n_agents)
            total_actions = np.zeros(n_agents)

            for step in range(n_steps):
                # Get actions from each agent
                actions = np.array(
                    [
                        group[i].act(
                            resources[i],
                            np.mean([resources[j] for j in range(n_agents) if j != i]),
                            (
                                np.mean([prev_actions[j] for j in range(n_agents) if j != i])
                                if prev_actions is not None
                                else None
                            ),
                        )
                        for i in range(n_agents)
                    ]
                )

                resources, rewards = env.step(actions)
                total_rewards += rewards
                total_actions += actions
                prev_actions = actions

            # Update fitness
            for i, agent in enumerate(group):
                agent.fitness += total_rewards[i]
                agent.cooperation_score += (
                    1 - total_actions[i] / n_steps
                )  # Lower claim = more cooperative


def evolve(population_size=100, n_generations=200, n_episodes_per_eval=30):
    """Main evolution loop."""

    print(
        f"""
======================================================================
  EVOLUTION CONFIGURATION

  Population size: {population_size}
  Generations: {n_generations}
  Episodes per evaluation: {n_episodes_per_eval}

  DNA: [base_cooperation, reciprocity, greed_threshold, forgiveness]
======================================================================
    """
    )

    # Initialize population
    population = [EvolvedAgent() for _ in range(population_size)]

    # Tracking
    best_fitness_history = []
    avg_fitness_history = []
    avg_cooperation_history = []
    best_dna_history = []

    start_time = time.time()

    for gen in tqdm(range(n_generations), desc="Evolving"):
        # Evaluate fitness
        evaluate_population(population, n_episodes=n_episodes_per_eval)

        # Sort by fitness
        population.sort(key=lambda x: x.fitness, reverse=True)

        # Track stats
        best_fitness = population[0].fitness
        avg_fitness = np.mean([a.fitness for a in population])
        avg_cooperation = np.mean([a.cooperation_score for a in population])

        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)
        avg_cooperation_history.append(avg_cooperation)
        best_dna_history.append(population[0].dna.copy())

        # Print progress
        if gen % 20 == 0:
            best = population[0]
            tqdm.write(
                f"  Gen {gen}: Best fitness={best_fitness:.1f}, Avg coop={avg_cooperation:.3f}"
            )
            tqdm.write(
                f"    Best DNA: coop={best.base_cooperation:.2f}, recip={best.reciprocity:.2f}, "
                f"greed={best.greed_threshold:.2f}, forgive={best.forgiveness:.2f}"
            )

        # Selection: keep top 20%
        elite_size = population_size // 5
        elite = population[:elite_size]

        # Create new generation
        new_population = elite.copy()  # Elitism

        while len(new_population) < population_size:
            # Tournament selection
            parent1 = max(np.random.choice(elite, 3), key=lambda x: x.fitness)
            parent2 = max(np.random.choice(elite, 3), key=lambda x: x.fitness)

            # Crossover
            child = EvolvedAgent.crossover(parent1, parent2)

            # Mutation
            child = child.mutate(mutation_rate=0.2, mutation_strength=0.15)

            new_population.append(child)

        population = new_population

    elapsed = time.time() - start_time

    # Final evaluation
    evaluate_population(population, n_episodes=50)
    population.sort(key=lambda x: x.fitness, reverse=True)

    print(
        f"""
======================================================================
  EVOLUTION COMPLETE!

  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)
  Generations: {n_generations}

  BEST EVOLVED AGENT:
    Base cooperation: {population[0].base_cooperation:.3f}
    Reciprocity:      {population[0].reciprocity:.3f}
    Greed threshold:  {population[0].greed_threshold:.3f}
    Forgiveness:      {population[0].forgiveness:.3f}

  FITNESS: {population[0].fitness:.1f}
======================================================================
    """
    )

    # Analysis
    early_coop = np.mean(avg_cooperation_history[:10])
    late_coop = np.mean(avg_cooperation_history[-10:])

    print(f"Early cooperation (gen 0-10):   {early_coop:.3f}")
    print(f"Late cooperation (gen {n_generations-10}-{n_generations}): {late_coop:.3f}")
    print(f"Change: {late_coop - early_coop:+.3f}")

    if late_coop > early_coop + 0.05:
        print("\n✅ EVOLUTION DISCOVERED COOPERATION!")
    elif late_coop < early_coop - 0.05:
        print("\n❌ Evolution led to MORE selfishness")
    else:
        print("\n➖ No significant change in cooperation")

    # Compare to neural network
    print(
        f"""
----------------------------------------------------------------------
  COMPARISON TO NEURAL NETWORKS:

  Neural nets (10M episodes):  Cooperation DECREASED (0.494 -> 0.493)
  Evolution ({n_generations} generations):    Cooperation {"INCREASED" if late_coop > early_coop else "DECREASED"} ({early_coop:.3f} -> {late_coop:.3f})

  {"🏆 EVOLUTION WINS!" if late_coop > 0.5 else "Both approaches struggled with cooperation"}
----------------------------------------------------------------------
    """
    )

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.plot(best_fitness_history, label="Best", alpha=0.8)
    ax.plot(avg_fitness_history, label="Average", alpha=0.5)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title("Fitness Over Evolution")
    ax.legend()

    ax = axes[1]
    ax.plot(avg_cooperation_history, color="green")
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="Neutral")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Cooperation Score")
    ax.set_title("Cooperation Over Evolution")
    ax.legend()

    ax = axes[2]
    dna_array = np.array(best_dna_history)
    ax.plot(dna_array[:, 0], label="Base Coop")
    ax.plot(dna_array[:, 1], label="Reciprocity")
    ax.plot(dna_array[:, 2], label="Greed Thresh")
    ax.plot(dna_array[:, 3], label="Forgiveness")
    ax.set_xlabel("Generation")
    ax.set_ylabel("DNA Value")
    ax.set_title("DNA Evolution")
    ax.legend()

    plt.tight_layout()
    plt.savefig("EVOLUTION_results.png", dpi=150)
    print("\nSaved: EVOLUTION_results.png")

    return population, best_fitness_history, avg_cooperation_history


if __name__ == "__main__":
    # Run evolution
    population, fitness_history, coop_history = evolve(
        population_size=100, n_generations=200, n_episodes_per_eval=30
    )
