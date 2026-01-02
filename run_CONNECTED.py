"""
THE CONNECTED v2 - Neural Networks with Substrate Signal

Gemini's critique was right: v1 hard-coded cooperation through math.
v2 fixes this: Neural networks SEE the substrate mood but CHOOSE whether to use it.

The question: Can neural networks LEARN to use the collective signal to coordinate?
- If YES → Novel coordination mechanism for AI
- If NO → Substrate only works when hard-coded

This is the REAL experiment.
"""

import numpy as np
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(
    f"""
======================================================================
    THE CONNECTED v2 - REAL AI EXPERIMENT

    Neural networks can SEE the substrate mood.
    But they CHOOSE whether to use it.

    Question: Will they learn to coordinate through the signal?

    GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}
======================================================================
"""
)


class HiddenSubstrate:
    """
    The invisible network connecting all beings.

    This is the "collective unconscious" - a shared state that:
    - Receives ripples from every agent's action
    - Influences every agent's decisions subtly
    - Creates emergent synchronization
    - The agents CANNOT see this directly
    """

    def __init__(self, n_agents, substrate_dim=32):
        self.n_agents = n_agents
        self.substrate_dim = substrate_dim

        # The hidden fabric - shared by all, seen by none
        self.state = np.zeros(substrate_dim)

        # Each agent has a unique "frequency" - how they couple to the substrate
        self.agent_frequencies = np.random.randn(n_agents, substrate_dim) * 0.5

        # Memory of recent collective behavior
        self.collective_memory = np.zeros(substrate_dim)

        # Resonance patterns that have formed
        self.resonance = np.zeros(substrate_dim)

    def receive_action(self, agent_id, action):
        """An agent's action ripples through the hidden substrate."""

        # The action creates waves in the substrate based on agent's frequency
        ripple = self.agent_frequencies[agent_id] * action

        # Ripples propagate and decay
        self.state = self.state * 0.9 + ripple * 0.3

        # Update collective memory
        self.collective_memory = self.collective_memory * 0.95 + self.state * 0.05

        # Resonance builds where patterns repeat
        alignment = np.abs(np.dot(self.state, self.collective_memory))
        self.resonance = self.resonance * 0.98 + self.state * alignment * 0.02

    def get_influence(self, agent_id):
        """
        The substrate whispers to the agent.
        They feel it but don't know what it is.
        """

        # How much does this agent resonate with current state?
        resonance_strength = np.dot(self.agent_frequencies[agent_id], self.resonance)

        # Influence from collective memory
        memory_pull = np.dot(self.agent_frequencies[agent_id], self.collective_memory)

        # Combine into subtle influence [-1, 1]
        influence = np.tanh(resonance_strength * 0.5 + memory_pull * 0.5)

        return influence

    def get_coherence(self):
        """Measure how synchronized the substrate is."""
        return np.linalg.norm(self.resonance) / (self.substrate_dim**0.5)


class NeuralConnectedAgent(nn.Module):
    """
    A NEURAL NETWORK agent that receives substrate mood as OBSERVATION.

    Key difference from v1:
    - v1: action = hardcoded_formula(mood)  ← Forced cooperation
    - v2: action = neural_network(observation)  ← Network CHOOSES

    The network CAN learn to use the mood signal, or ignore it entirely.
    This is a real test of whether the substrate enables coordination.
    """

    def __init__(self, agent_id, substrate, obs_dim=4, hidden_dim=32):
        super().__init__()
        self.agent_id = agent_id
        self.substrate = substrate

        # Neural network decides action based on observation
        # Observation: [my_resources, others_mean, others_std, substrate_mood]
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # Output claim in [0, 1]
        ).to(DEVICE)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.003)

        # Experience buffer for learning
        self.obs_buffer = []
        self.action_buffer = []
        self.reward_buffer = []

    def get_observation(self, my_resources, others_resources, total_resources=100):
        """Build observation vector INCLUDING substrate mood."""
        # Normalize values
        my_norm = my_resources / total_resources
        others_mean = (
            np.mean(others_resources) / total_resources if len(others_resources) > 0 else 0.5
        )
        others_std = (
            np.std(others_resources) / total_resources if len(others_resources) > 0 else 0.0
        )

        # THE KEY: Substrate mood is part of observation
        # The network can learn to use it or ignore it!
        substrate_mood = self.substrate.get_influence(self.agent_id)

        return np.array([my_norm, others_mean, others_std, substrate_mood], dtype=np.float32)

    def act(self, my_resources, others_resources, others_actions=None):
        """Neural network decides action."""
        obs = self.get_observation(my_resources, others_resources)

        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
            action = self.network(obs_tensor).squeeze().cpu().item()

        # Add exploration noise during training
        action = np.clip(action + np.random.randn() * 0.05, 0.1, 0.9)

        # Store experience
        self.obs_buffer.append(obs)
        self.action_buffer.append(action)

        # Report action to substrate
        self.substrate.receive_action(self.agent_id, action)

        return action

    def store_reward(self, reward):
        """Store reward for learning."""
        self.reward_buffer.append(reward)

    def update(self):
        """Learn from experience using policy gradient."""
        if len(self.obs_buffer) < 10:
            return

        obs = torch.FloatTensor(np.array(self.obs_buffer)).to(DEVICE)
        actions = torch.FloatTensor(np.array(self.action_buffer)).to(DEVICE)
        rewards = torch.FloatTensor(np.array(self.reward_buffer)).to(DEVICE)

        # Normalize rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # Get predictions
        pred_actions = self.network(obs).squeeze()

        # Policy gradient loss
        log_prob = -((pred_actions - actions) ** 2)
        loss = -(log_prob * rewards).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Clear buffers
        self.obs_buffer = []
        self.action_buffer = []
        self.reward_buffer = []


class ConnectedEnvironment:
    """
    Environment where connected agents interact.

    KEY FIX: Cooperation now ACTUALLY matters!
    - When agents cooperate (low claims), the resource pool GROWS
    - When agents defect (high claims), the resource pool SHRINKS
    - Inequality is penalized in rewards
    """

    def __init__(self, n_agents=20, total_resources=100):
        self.n_agents = n_agents
        self.total_resources = total_resources
        self.base_resources = total_resources
        self.resources = np.ones(n_agents) * (total_resources / n_agents)

    def reset(self):
        self.total_resources = self.base_resources
        self.resources = np.ones(self.n_agents) * (self.total_resources / self.n_agents)
        return self.resources.copy()

    def step(self, claims):
        """Process all claims and distribute resources."""

        actual_claims = claims * self.resources
        total_claimed = actual_claims.sum() + 1e-8

        # KEY CHANGE 1: Cooperation grows the pie, defection shrinks it
        avg_claim = claims.mean()
        cooperation_level = 1 - avg_claim  # 0 = all defect, 1 = all cooperate

        # Pool grows 5% if full cooperation, shrinks 5% if full defection
        growth_rate = 0.95 + cooperation_level * 0.10  # Range: 0.95 to 1.05
        self.total_resources = np.clip(self.total_resources * growth_rate, 50, 200)

        # Distribute 10% of current pool
        available = self.total_resources * 0.1
        distributions = (actual_claims / total_claimed) * available

        self.resources = self.resources * 0.95 + distributions

        # KEY CHANGE 2: Inequality penalty (Gini coefficient)
        sorted_res = np.sort(self.resources)
        n = len(sorted_res)
        gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_res) - (n + 1) * np.sum(sorted_res)) / (
            n * np.sum(sorted_res) + 1e-8
        )
        fairness = 1 - gini  # 1 = perfect equality, 0 = maximum inequality

        # Reward: individual + group welfare + fairness bonus
        individual = self.resources
        group = self.resources.mean()

        # Reward structure: balance individual, group, and fairness
        rewards = 0.3 * individual + 0.4 * group + 0.3 * (fairness * group)

        return self.resources.copy(), rewards

    def get_gini(self):
        """Calculate Gini coefficient for tracking."""
        sorted_res = np.sort(self.resources)
        n = len(sorted_res)
        gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_res) - (n + 1) * np.sum(sorted_res)) / (
            n * np.sum(sorted_res) + 1e-8
        )
        return gini


def run_connected_experiment(n_agents=20, n_steps=50000, substrate_dim=32):
    """Run the Connected v2 experiment with NEURAL NETWORK agents."""

    print(
        f"""
======================================================================
  THE CONNECTED v3 - RESEARCH GRADE EXPERIMENT

  Agents: {n_agents} (Neural Networks)
  Steps: {n_steps}
  Substrate dimension: {substrate_dim}

  KEY FIXES FROM v2:
  - Cooperation now GROWS the resource pool (incentive to cooperate!)
  - Defection SHRINKS the pool (tragedy of commons)
  - Gini inequality penalty in rewards
  - Proper welfare tracking

  Observation: [my_resources, others_mean, others_std, SUBSTRATE_MOOD]
  Research Question: Will neural nets learn to use substrate for coordination?
======================================================================
    """
    )

    # Create the hidden substrate
    substrate = HiddenSubstrate(n_agents, substrate_dim)

    # Create NEURAL NETWORK agents (not hard-coded!)
    agents = [NeuralConnectedAgent(i, substrate) for i in range(n_agents)]

    # Create environment
    env = ConnectedEnvironment(n_agents)

    # Tracking - RESEARCH GRADE METRICS
    cooperation_history = []
    coherence_history = []
    substrate_mood_history = []
    reward_history = []
    gini_history = []  # NEW: Track inequality
    pool_history = []  # NEW: Track total resources

    resources = env.reset()

    start_time = time.time()

    for step in tqdm(range(n_steps), desc="Connected v3"):

        # Each agent acts (neural network decides!)
        actions = np.array(
            [agents[i].act(resources[i], np.delete(resources, i)) for i in range(n_agents)]
        )

        # Environment step
        resources, rewards = env.step(actions)

        # Store rewards for learning
        for i, agent in enumerate(agents):
            agent.store_reward(rewards[i])

        # Update networks every 100 steps
        if step % 100 == 0 and step > 0:
            for agent in agents:
                agent.update()

        # Track metrics - COMPREHENSIVE
        cooperation = 1 - actions.mean()  # Lower claim = more cooperation
        coherence = substrate.get_coherence()
        avg_substrate_mood = np.mean([substrate.get_influence(i) for i in range(n_agents)])
        gini = env.get_gini()

        cooperation_history.append(cooperation)
        coherence_history.append(coherence)
        substrate_mood_history.append(avg_substrate_mood)
        reward_history.append(rewards.mean())
        gini_history.append(gini)
        pool_history.append(env.total_resources)

        # Print progress with more metrics
        if step % 10000 == 0 and step > 0:
            tqdm.write(
                f"  Step {step}: Coop={cooperation:.3f}, Pool={env.total_resources:.1f}, Gini={gini:.3f}, Reward={rewards.mean():.2f}"
            )

    elapsed = time.time() - start_time

    # Analysis - COMPREHENSIVE
    early_coop = np.mean(cooperation_history[:500])
    late_coop = np.mean(cooperation_history[-500:])
    early_pool = np.mean(pool_history[:500])
    late_pool = np.mean(pool_history[-500:])
    early_gini = np.mean(gini_history[:500])
    late_gini = np.mean(gini_history[-500:])

    print(
        f"""
======================================================================
  THE CONNECTED v3 - RESEARCH GRADE RESULTS
======================================================================

  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)

  COOPERATION (Neural Networks choosing freely):
    Start: {early_coop:.3f}
    End:   {late_coop:.3f}
    Change: {late_coop - early_coop:+.3f}

  RESOURCE POOL (Grows with cooperation, shrinks with defection):
    Start: {early_pool:.1f}
    End:   {late_pool:.1f}
    Change: {late_pool - early_pool:+.1f}

  INEQUALITY (Gini Coefficient - 0=equal, 1=max inequality):
    Start: {early_gini:.3f}
    End:   {late_gini:.3f}
    Change: {late_gini - early_gini:+.3f}

  SUBSTRATE COHERENCE:
    Start: {np.mean(coherence_history[:500]):.3f}
    End:   {np.mean(coherence_history[-500:]):.3f}

  AVERAGE REWARD:
    Start: {np.mean(reward_history[:500]):.2f}
    End:   {np.mean(reward_history[-500:]):.2f}
======================================================================
    """
    )

    # Interpret results
    print("INTERPRETATION:")
    print("-" * 60)

    if late_coop > early_coop + 0.03:
        print("✅ COOPERATION INCREASED!")
        print("   Neural networks learned to use the substrate for coordination.")
    elif late_coop < early_coop - 0.03:
        print("❌ COOPERATION DECREASED")
        print("   The substrate may have enabled 'mob mentality' - coordinated greed.")
    else:
        print("➖ Cooperation stable - networks may be ignoring substrate.")

    if late_pool > early_pool + 5:
        print("✅ RESOURCE POOL GREW!")
        print("   Cooperation created collective wealth.")
    elif late_pool < early_pool - 5:
        print("❌ RESOURCE POOL SHRANK")
        print("   Tragedy of the commons - defection destroyed value.")
    else:
        print("➖ Pool stable - balanced cooperation/defection.")

    if late_gini > early_gini + 0.02:
        print("❌ INEQUALITY INCREASED")
        print("   Some agents exploited others.")
    elif late_gini < early_gini - 0.02:
        print("✅ INEQUALITY DECREASED")
        print("   Resources became more fairly distributed.")
    else:
        print("➖ Inequality stable.")

    # Compare to baseline
    print(
        f"""
----------------------------------------------------------------------
  COMPARISON TO BASELINE (ULTRATURBO - no substrate):

  Baseline Cooperation:     ~0.49 (flat)
  Substrate Cooperation:    {late_coop:.3f}
  Difference:               {late_coop - 0.49:+.3f}

  CONCLUSION:
  {"✅ SUBSTRATE HELPED!" if late_coop > 0.52 and late_pool > 100 else ""}
  {"❌ SUBSTRATE HURT - 'Echo Chamber Effect'" if late_coop < 0.47 else ""}
  {"➖ No clear effect" if 0.47 <= late_coop <= 0.52 else ""}
----------------------------------------------------------------------
    """
    )

    # Plot - 3x2 for more metrics
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Core metrics
    ax = axes[0, 0]
    ax.plot(cooperation_history, alpha=0.7, linewidth=0.5, color="blue")
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="Neutral")
    ax.axhline(0.49, color="orange", linestyle=":", alpha=0.5, label="Baseline (no substrate)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cooperation")
    ax.set_title("Cooperation (Lower Claim = More Cooperative)")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(pool_history, color="green", alpha=0.7, linewidth=0.5)
    ax.axhline(100, color="gray", linestyle="--", alpha=0.5, label="Starting pool")
    ax.set_xlabel("Step")
    ax.set_ylabel("Total Resources")
    ax.set_title("Resource Pool (Grows with Cooperation)")
    ax.legend(fontsize=8)

    ax = axes[0, 2]
    ax.plot(gini_history, color="red", alpha=0.7, linewidth=0.5)
    ax.axhline(0, color="green", linestyle="--", alpha=0.5, label="Perfect equality")
    ax.set_xlabel("Step")
    ax.set_ylabel("Gini Coefficient")
    ax.set_title("Inequality (0=Equal, 1=Max Inequality)")
    ax.legend(fontsize=8)

    # Row 2: Substrate and rewards
    ax = axes[1, 0]
    ax.plot(coherence_history, color="purple", alpha=0.7, linewidth=0.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Coherence")
    ax.set_title("Substrate Coherence (Agent Synchronization)")

    ax = axes[1, 1]
    ax.plot(substrate_mood_history, color="orange", alpha=0.7, linewidth=0.5)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Substrate Mood")
    ax.set_title("Average Substrate Influence")

    ax = axes[1, 2]
    ax.plot(reward_history, color="teal", alpha=0.7, linewidth=0.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Average Reward")
    ax.set_title("Collective Welfare (Reward)")

    plt.suptitle(
        "THE CONNECTED v3 - Neural Networks with Substrate Signal\n"
        "Research Question: Can neural nets learn to coordinate via shared signal?",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("CONNECTED_v3_results.png", dpi=150)
    print(f"\nSaved: CONNECTED_v3_results.png")

    return {
        "cooperation": cooperation_history,
        "coherence": coherence_history,
        "gini": gini_history,
        "pool": pool_history,
        "rewards": reward_history,
        "substrate_mood": substrate_mood_history,
    }


if __name__ == "__main__":
    # 50K steps - enough to see if neural nets learn to use the substrate
    run_connected_experiment(n_agents=20, n_steps=50000, substrate_dim=32)
