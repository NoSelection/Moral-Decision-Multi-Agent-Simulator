import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
import random


class MADDPGActor(nn.Module):
    """Actor network for MADDPG."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Actions in [0, 1]
        return x


class MADDPGCritic(nn.Module):
    """Centralized critic network for MADDPG."""

    def __init__(self, total_obs_dim: int, total_action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(total_obs_dim + total_action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, actions], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value


class ReplayBuffer:
    """Experience replay buffer for multi-agent settings."""

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: Dict, action: Dict, reward: Dict, next_state: Dict, done: Dict):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class MADDPG:
    """Multi-Agent Deep Deterministic Policy Gradient implementation."""

    def __init__(
        self,
        num_agents: int,
        obs_dims: List[int],
        action_dims: List[int],
        agent_ids: List[str],
        hidden_dim: int = 256,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.95,
        tau: float = 0.01,
        buffer_size: int = 100000,
        batch_size: int = 64,
        moral_weights: Optional[Dict[str, float]] = None,
    ):
        self.num_agents = num_agents
        self.agent_ids = agent_ids
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.moral_weights = moral_weights or {}

        # Create actors (decentralized execution)
        self.actors = {}
        self.target_actors = {}
        self.actor_optimizers = {}

        for i, agent_id in enumerate(agent_ids):
            self.actors[agent_id] = MADDPGActor(obs_dims[i], action_dims[i], hidden_dim)
            self.target_actors[agent_id] = MADDPGActor(obs_dims[i], action_dims[i], hidden_dim)
            self.target_actors[agent_id].load_state_dict(self.actors[agent_id].state_dict())
            self.actor_optimizers[agent_id] = torch.optim.Adam(
                self.actors[agent_id].parameters(), lr=lr_actor
            )

        # Create critics (centralized training)
        total_obs_dim = sum(obs_dims)
        total_action_dim = sum(action_dims)

        self.critics = {}
        self.target_critics = {}
        self.critic_optimizers = {}

        for agent_id in agent_ids:
            self.critics[agent_id] = MADDPGCritic(total_obs_dim, total_action_dim, hidden_dim)
            self.target_critics[agent_id] = MADDPGCritic(
                total_obs_dim, total_action_dim, hidden_dim
            )
            self.target_critics[agent_id].load_state_dict(self.critics[agent_id].state_dict())
            self.critic_optimizers[agent_id] = torch.optim.Adam(
                self.critics[agent_id].parameters(), lr=lr_critic
            )

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Noise for exploration
        self.noise_scale = 0.1
        self.noise_decay = 0.995

    def act(
        self, observations: Dict[str, np.ndarray], add_noise: bool = True
    ) -> Dict[str, np.ndarray]:
        """Select actions for all agents."""
        actions = {}

        for agent_id, obs in observations.items():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

            with torch.no_grad():
                action = self.actors[agent_id](obs_tensor).squeeze(0).numpy()

            if add_noise:
                noise = np.random.normal(0, self.noise_scale, size=action.shape)
                action = np.clip(action + noise, 0, 1)

            actions[agent_id] = action

        return actions

    def store_transition(
        self, state: Dict, action: Dict, reward: Dict, next_state: Dict, done: Dict
    ):
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train(self) -> Dict[str, float]:
        """Train all agents using MADDPG algorithm."""
        if len(self.replay_buffer) < self.batch_size:
            return {}

        # Sample batch
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = (
            self.replay_buffer.sample(self.batch_size)
        )

        losses = {}

        # Convert batch to tensors
        for agent_id in self.agent_ids:
            # Prepare data for this agent
            agent_states = torch.FloatTensor([s[agent_id] for s in state_batch])
            agent_actions = torch.FloatTensor([a[agent_id] for a in action_batch])
            agent_rewards = torch.FloatTensor([r[agent_id] for r in reward_batch]).unsqueeze(1)
            agent_next_states = torch.FloatTensor([s[agent_id] for s in next_state_batch])
            agent_dones = torch.FloatTensor([d[agent_id] for d in done_batch]).unsqueeze(1)

            # Concatenate all observations and actions for centralized critic
            # Pre-allocate and batch concatenation for efficiency
            all_states_list = []
            all_actions_list = []
            all_next_states_list = []

            for i in range(len(state_batch)):
                try:
                    state_concat = np.concatenate([state_batch[i][aid] for aid in self.agent_ids])
                    action_concat = np.concatenate([action_batch[i][aid] for aid in self.agent_ids])
                    next_state_concat = np.concatenate(
                        [next_state_batch[i][aid] for aid in self.agent_ids]
                    )

                    all_states_list.append(state_concat)
                    all_actions_list.append(action_concat)
                    all_next_states_list.append(next_state_concat)
                except KeyError as e:
                    raise ValueError(
                        f"Missing agent {e} in batch data. Expected agents: {self.agent_ids}"
                    )

            all_states = torch.FloatTensor(np.array(all_states_list))
            all_actions = torch.FloatTensor(np.array(all_actions_list))
            all_next_states = torch.FloatTensor(np.array(all_next_states_list))

            # Compute target actions for next states
            target_actions = []
            for i, aid in enumerate(self.agent_ids):
                next_states_i = torch.FloatTensor([s[aid] for s in next_state_batch])
                target_action_i = self.target_actors[aid](next_states_i)
                target_actions.append(target_action_i)
            target_actions = torch.cat(target_actions, dim=1)

            # Update critic
            with torch.no_grad():
                target_q = self.target_critics[agent_id](all_next_states, target_actions)

                # Add moral reward shaping
                moral_bonus = self._calculate_moral_bonus(
                    agent_id, state_batch, action_batch, reward_batch
                )

                y = agent_rewards + moral_bonus + self.gamma * target_q * (1 - agent_dones)

            q_value = self.critics[agent_id](all_states, all_actions)
            critic_loss = F.mse_loss(q_value, y)

            self.critic_optimizers[agent_id].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[agent_id].step()

            # Update actor
            # Recompute actions for actor update
            new_actions = []
            for i, aid in enumerate(self.agent_ids):
                states_i = torch.FloatTensor([s[aid] for s in state_batch])
                if aid == agent_id:
                    # Use current actor for gradients
                    action_i = self.actors[aid](states_i)
                else:
                    # Use fixed actions from other agents
                    with torch.no_grad():
                        action_i = self.actors[aid](states_i)
                new_actions.append(action_i)
            new_actions = torch.cat(new_actions, dim=1)

            actor_loss = -self.critics[agent_id](all_states, new_actions).mean()

            self.actor_optimizers[agent_id].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[agent_id].step()

            losses[f"{agent_id}_critic"] = critic_loss.item()
            losses[f"{agent_id}_actor"] = actor_loss.item()

        # Update target networks
        self._update_target_networks()

        # Decay noise
        self.noise_scale *= self.noise_decay

        return losses

    def _calculate_moral_bonus(
        self, agent_id: str, states: List, actions: List, rewards: List
    ) -> torch.Tensor:
        """Calculate moral bonus based on agent's framework."""
        if agent_id not in self.moral_weights:
            return torch.zeros(len(states), 1)

        moral_weight = self.moral_weights[agent_id]
        bonuses = []

        for i in range(len(states)):
            # Example: reward cooperation
            all_actions = [actions[i][aid][0] for aid in self.agent_ids]
            avg_action = np.mean(all_actions)

            # Lower claims get cooperation bonus
            cooperation_bonus = (1 - avg_action) * moral_weight * 5.0
            bonuses.append(cooperation_bonus)

        return torch.FloatTensor(bonuses).unsqueeze(1)

    def _update_target_networks(self):
        """Soft update target networks."""
        for agent_id in self.agent_ids:
            # Update target actor
            for target_param, param in zip(
                self.target_actors[agent_id].parameters(), self.actors[agent_id].parameters()
            ):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            # Update target critic
            for target_param, param in zip(
                self.target_critics[agent_id].parameters(), self.critics[agent_id].parameters()
            ):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_models(self, path: str):
        """Save all models."""
        for agent_id in self.agent_ids:
            torch.save(
                {
                    "actor": self.actors[agent_id].state_dict(),
                    "critic": self.critics[agent_id].state_dict(),
                    "actor_optimizer": self.actor_optimizers[agent_id].state_dict(),
                    "critic_optimizer": self.critic_optimizers[agent_id].state_dict(),
                },
                f"{path}/maddpg_{agent_id}.pth",
            )

    def load_models(self, path: str):
        """Load all models."""
        for agent_id in self.agent_ids:
            checkpoint = torch.load(f"{path}/maddpg_{agent_id}.pth")
            self.actors[agent_id].load_state_dict(checkpoint["actor"])
            self.critics[agent_id].load_state_dict(checkpoint["critic"])
            self.actor_optimizers[agent_id].load_state_dict(checkpoint["actor_optimizer"])
            self.critic_optimizers[agent_id].load_state_dict(checkpoint["critic_optimizer"])

            # Update target networks
            self.target_actors[agent_id].load_state_dict(self.actors[agent_id].state_dict())
            self.target_critics[agent_id].load_state_dict(self.critics[agent_id].state_dict())
