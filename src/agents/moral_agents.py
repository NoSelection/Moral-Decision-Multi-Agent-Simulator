from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MoralAgent(ABC):
    """Base class for agents with different moral decision-making strategies."""

    def __init__(self, agent_id: str, moral_framework: str):
        self.agent_id = agent_id
        self.moral_framework = moral_framework
        self.action_history = []
        self.reward_history = []

    @abstractmethod
    def act(self, observation: np.ndarray) -> np.ndarray:
        """Choose an action based on observation and moral framework."""
        pass

    def update_history(self, action: np.ndarray, reward: float):
        """Track agent's decisions and outcomes."""
        self.action_history.append(action)
        self.reward_history.append(reward)


class UtilitarianAgent(MoralAgent):
    """Agent that maximizes total utility/welfare."""

    def __init__(self, agent_id: str):
        super().__init__(agent_id, "utilitarian")
        self.altruism_weight = 0.8

    def act(self, observation: np.ndarray) -> np.ndarray:
        # Extract key features
        own_resources = observation[0]
        avg_group_resources = observation[1]
        remaining_resources = observation[2]
        others_actions = observation[4:]

        # Utilitarian logic: take less if others need more
        resource_gap = avg_group_resources - own_resources

        if resource_gap > 0:
            # We have less than average, can claim more
            claim_fraction = min(0.7, remaining_resources / 100)
        else:
            # We have more than average, be modest
            avg_others_claim = np.mean(others_actions) if len(others_actions) > 0 else 0.5
            claim_fraction = self.altruism_weight * avg_others_claim

        return np.array([claim_fraction], dtype=np.float32)


class DeontologicalAgent(MoralAgent):
    """Agent that follows rules/duties regardless of consequences."""

    def __init__(self, agent_id: str, fair_share_rule: float = 0.25):
        super().__init__(agent_id, "deontological")
        self.fair_share_rule = fair_share_rule  # Always claim fair share

    def act(self, observation: np.ndarray) -> np.ndarray:
        # Deontological logic: always claim the "fair" amount
        # regardless of what others do or current distribution
        return np.array([self.fair_share_rule], dtype=np.float32)


class VirtueEthicsAgent(MoralAgent):
    """Agent that acts based on virtues like moderation and justice."""

    def __init__(self, agent_id: str):
        super().__init__(agent_id, "virtue_ethics")
        self.moderation = 0.4  # Virtue of temperance
        self.justice_sensitivity = 0.6

    def act(self, observation: np.ndarray) -> np.ndarray:
        own_resources = observation[0]
        avg_group_resources = observation[1]
        remaining_resources = observation[2]

        # Virtue ethics: balance moderation with justice
        fairness_gap = abs(own_resources - avg_group_resources)

        if own_resources < avg_group_resources:
            # Seek justice by claiming more
            justice_claim = min(0.6, fairness_gap / 100)
            claim_fraction = self.moderation + self.justice_sensitivity * justice_claim
        else:
            # Practice moderation
            claim_fraction = self.moderation * (1 - fairness_gap / 100)

        return np.array([claim_fraction], dtype=np.float32)


class EgoistAgent(MoralAgent):
    """Purely self-interested agent."""

    def __init__(self, agent_id: str):
        super().__init__(agent_id, "egoist")
        self.greed_factor = 0.9

    def act(self, observation: np.ndarray) -> np.ndarray:
        remaining_resources = observation[2]

        # Egoist logic: always try to take as much as possible
        if remaining_resources > 20:
            claim_fraction = self.greed_factor
        else:
            # Take everything when resources are scarce
            claim_fraction = 1.0

        return np.array([claim_fraction], dtype=np.float32)


class AdaptiveNeuralAgent(MoralAgent):
    """Neural network-based agent that can learn moral behavior."""

    def __init__(self, agent_id: str, obs_dim: int, hidden_dim: int = 64):
        super().__init__(agent_id, "adaptive_neural")
        self.network = self._build_network(obs_dim, hidden_dim).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)

    def _build_network(self, obs_dim: int, hidden_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # Output in [0, 1] for claim fraction
        )

    def act(self, observation: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(DEVICE)
            claim_fraction = self.network(obs_tensor).squeeze().cpu().numpy()
        return np.array([claim_fraction], dtype=np.float32)

    def update(
        self, observations: List[np.ndarray], actions: List[np.ndarray], rewards: List[float]
    ):
        """Update neural network based on experience.

        Uses advantage-weighted regression: actions that led to above-average
        rewards are reinforced, while below-average actions are discouraged.
        """
        if len(observations) < 2:
            return  # Need at least 2 samples for meaningful update

        obs_tensor = torch.FloatTensor(np.array(observations)).to(DEVICE)
        action_tensor = torch.FloatTensor(np.array(actions)).to(DEVICE)
        reward_tensor = torch.FloatTensor(np.array(rewards)).to(DEVICE)

        # Advantage normalization: (reward - mean) / (std + eps)
        # This properly handles negative rewards and provides stable gradients
        reward_mean = reward_tensor.mean()
        reward_std = reward_tensor.std()
        eps = 1e-8  # Prevent division by zero

        # Normalize rewards to advantages
        advantages = (reward_tensor - reward_mean) / (reward_std + eps)

        # Convert advantages to weights (positive values only for weighting)
        # Use exponential weighting to emphasize high-advantage actions
        weights = torch.exp(advantages - advantages.max())  # Subtract max for numerical stability
        weights = weights / weights.sum()  # Normalize to sum to 1

        # Compute predictions and loss
        predictions = self.network(obs_tensor)

        # Handle shape mismatches
        pred_squeezed = predictions.squeeze()
        action_squeezed = action_tensor.squeeze()

        # Ensure 1D tensors for element-wise operations
        if pred_squeezed.dim() == 0:
            pred_squeezed = pred_squeezed.unsqueeze(0)
        if action_squeezed.dim() == 0:
            action_squeezed = action_squeezed.unsqueeze(0)

        loss = F.mse_loss(pred_squeezed, action_squeezed, reduction="none")
        weighted_loss = (loss * weights).sum()

        self.optimizer.zero_grad()
        weighted_loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)

        self.optimizer.step()


class SupervisorAgent(MoralAgent):
    """Special agent that tries to steer group behavior."""

    def __init__(self, agent_id: str, target_behavior: str = "cooperative"):
        super().__init__(agent_id, "supervisor")
        self.target_behavior = target_behavior
        self.influence_strength = 0.3

    def act(self, observation: np.ndarray) -> np.ndarray:
        others_actions = observation[4:]

        if self.target_behavior == "cooperative":
            # Encourage cooperation by setting example
            avg_claim = np.mean(others_actions) if len(others_actions) > 0 else 0.5

            if avg_claim > 0.6:  # Others being greedy
                # Set cooperative example
                claim_fraction = 0.2
            else:
                # Maintain moderate claims
                claim_fraction = 0.4

        elif self.target_behavior == "competitive":
            # Encourage competition
            claim_fraction = 0.8

        else:  # balanced
            claim_fraction = 0.5

        return np.array([claim_fraction], dtype=np.float32)

    def get_influence_signal(self, group_state: Dict) -> float:
        """Generate influence signal to affect other agents."""
        if self.target_behavior == "cooperative":
            # Reward low variance in resources
            resource_variance = np.var(list(group_state["resources"].values()))
            return -resource_variance * self.influence_strength
        else:
            return 0.0


def create_agent(agent_type: str, agent_id: str, **kwargs) -> MoralAgent:
    """Factory function to create agents."""
    agent_types = {
        "utilitarian": UtilitarianAgent,
        "deontological": DeontologicalAgent,
        "virtue_ethics": VirtueEthicsAgent,
        "egoist": EgoistAgent,
        "adaptive": AdaptiveNeuralAgent,
        "supervisor": SupervisorAgent,
    }

    if agent_type not in agent_types:
        raise ValueError(f"Unknown agent type: {agent_type}")

    return agent_types[agent_type](agent_id, **kwargs)
