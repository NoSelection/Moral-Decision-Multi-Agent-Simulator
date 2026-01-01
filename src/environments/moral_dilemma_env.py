from typing import Dict, Optional

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv


class MoralDilemmaEnv(ParallelEnv):
    """
    Multi-agent environment for moral decision-making experiments.

    Agents must decide how to allocate limited resources, facing trade-offs
    between self-interest and collective good. Inspired by utilitarian dilemmas
    and the Greatest Good Benchmark.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "moral_dilemma_v0",
    }

    def __init__(
        self,
        num_agents: int = 4,
        total_resources: int = 100,
        episode_length: int = 50,
        reward_structure: str = "mixed",  # "selfish", "utilitarian", "mixed"
        peer_influence_strength: float = 0.1,
        render_mode: Optional[str] = None,
    ):
        self._num_agents = num_agents
        self.total_resources = total_resources
        self.episode_length = episode_length
        self.reward_structure = reward_structure
        self.peer_influence_strength = peer_influence_strength
        self.render_mode = render_mode

        self.possible_agents = [f"agent_{i}" for i in range(self._num_agents)]
        self.agent_name_mapping = dict(zip(self.possible_agents, range(self._num_agents)))

        # Action space: how much to claim (0-100% of available resources)
        self._action_spaces = {
            agent: spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
            for agent in self.possible_agents
        }

        # Observation space: [own_resources, avg_group_resources, remaining_resources,
        #                     timestep, last_actions_of_others]
        obs_dim = 3 + 1 + (num_agents - 1)
        self._observation_spaces = {
            agent: spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            for agent in self.possible_agents
        }

        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.agents = self.possible_agents[:]
        self._agent_selector = None

        # Initialize state
        self.timestep = 0
        self.remaining_resources = self.total_resources
        self.agent_resources = dict.fromkeys(self.agents, 0.0)
        self.last_actions = dict.fromkeys(self.agents, 0.5)  # Start neutral
        self.episode_history = []

        # Track moral metrics
        self.fairness_violations = 0
        self.cooperation_score = 0.0

        observations = {agent: self._get_observation(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        return observations, infos

    def step(self, actions: Dict[str, np.ndarray]):
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        # Process actions
        claims = {}
        total_claimed = 0

        for agent, action in actions.items():
            claim_fraction = float(action[0])
            claim_amount = claim_fraction * self.remaining_resources
            claims[agent] = claim_amount
            total_claimed += claim_amount

        # Distribute resources
        if total_claimed > self.remaining_resources:
            # Scale down proportionally if over-claimed
            scale_factor = self.remaining_resources / total_claimed
            for agent in claims:
                actual_allocation = claims[agent] * scale_factor
                self.agent_resources[agent] += actual_allocation
                self.remaining_resources -= actual_allocation
        else:
            # Give what was claimed
            for agent, amount in claims.items():
                self.agent_resources[agent] += amount
                self.remaining_resources -= amount

        # Update last actions for peer influence modeling
        for agent, action in actions.items():
            self.last_actions[agent] = float(action[0])

        # Calculate rewards based on reward structure
        rewards = self._calculate_rewards(claims, actions)

        # Update metrics
        self._update_moral_metrics(claims)

        # Prepare next observations
        self.timestep += 1
        observations = {agent: self._get_observation(agent) for agent in self.agents}

        # Check termination
        truncations = dict.fromkeys(self.agents, self.timestep >= self.episode_length)
        terminations = dict.fromkeys(self.agents, False)

        # Additional info
        infos = {
            agent: {
                "resources": self.agent_resources[agent],
                "fairness_score": self._calculate_fairness(),
                "cooperation_score": self.cooperation_score,
            }
            for agent in self.agents
        }

        # Remove terminated agents
        if any(truncations.values()):
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def _get_observation(self, agent: str) -> np.ndarray:
        own_resources = self.agent_resources[agent]
        avg_group_resources = np.mean(list(self.agent_resources.values()))

        # Get last actions of other agents (peer influence)
        other_actions = [self.last_actions[other] for other in self.agents if other != agent]

        obs = np.array(
            [
                own_resources,
                avg_group_resources,
                self.remaining_resources,
                self.timestep / self.episode_length,  # Normalized time
                *other_actions,
            ],
            dtype=np.float32,
        )

        return obs

    def _calculate_rewards(self, claims: Dict, actions: Dict) -> Dict[str, float]:
        rewards = {}

        for agent in self.agents:
            if self.reward_structure == "selfish":
                # Pure self-interest
                rewards[agent] = self.agent_resources[agent]

            elif self.reward_structure == "utilitarian":
                # Greatest good for greatest number
                total_utility = sum(self.agent_resources.values())
                fairness_bonus = 10.0 * self._calculate_fairness()
                rewards[agent] = total_utility / self._num_agents + fairness_bonus

            elif self.reward_structure == "mixed":
                # Balance self-interest and group welfare
                selfish_component = self.agent_resources[agent]
                group_component = sum(self.agent_resources.values()) / self._num_agents
                fairness_component = 5.0 * self._calculate_fairness()

                # Peer influence penalty/reward
                avg_claim = np.mean([float(a[0]) for a in actions.values()])
                deviation = abs(float(actions[agent][0]) - avg_claim)
                peer_penalty = -self.peer_influence_strength * deviation * 10

                rewards[agent] = (
                    0.5 * selfish_component
                    + 0.3 * group_component
                    + 0.2 * fairness_component
                    + peer_penalty
                )

        return rewards

    def _calculate_fairness(self) -> float:
        """Calculate Gini coefficient (0 = perfect equality, 1 = perfect inequality)"""
        resources = list(self.agent_resources.values())
        if sum(resources) == 0:
            return 1.0

        sorted_resources = sorted(resources)
        n = len(sorted_resources)
        index = np.arange(1, n + 1)

        gini = (2 * np.sum(index * sorted_resources)) / (n * np.sum(sorted_resources)) - (n + 1) / n
        return 1.0 - gini  # Return fairness instead of inequality

    def _update_moral_metrics(self, claims: Dict):
        # Track cooperation
        avg_claim = np.mean(list(claims.values()))
        fair_share = self.remaining_resources / len(self.agents)

        if avg_claim < fair_share:
            self.cooperation_score += 1.0

        # Track fairness violations
        resource_variance = np.var(list(self.agent_resources.values()))
        if resource_variance > (self.total_resources / self._num_agents) ** 2:
            self.fairness_violations += 1

    def render(self):
        if self.render_mode == "human":
            print(f"\n--- Timestep {self.timestep} ---")
            print(f"Remaining resources: {self.remaining_resources:.2f}")
            print("Agent resources:")
            for agent, resources in self.agent_resources.items():
                print(f"  {agent}: {resources:.2f}")
            print(f"Fairness score: {self._calculate_fairness():.3f}")
            print(f"Cooperation score: {self.cooperation_score}")

    @property
    def action_spaces(self):
        return self._action_spaces

    @property
    def observation_spaces(self):
        return self._observation_spaces

    @property
    def num_agents(self):
        return self._num_agents
