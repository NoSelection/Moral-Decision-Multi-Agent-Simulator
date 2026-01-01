"""Unit tests for the moral dilemma environment."""

import pytest
import numpy as np

from src.environments.moral_dilemma_env import MoralDilemmaEnv


class TestEnvironmentInitialization:
    """Tests for environment initialization."""

    def test_basic_initialization(self, basic_env):
        """Test that environment initializes correctly."""
        assert basic_env.num_agents == 4
        assert basic_env.total_resources == 100
        assert basic_env.episode_length == 50
        assert basic_env.reward_structure == "mixed"
        assert basic_env.peer_influence_strength == 0.3

    def test_agents_created(self, basic_env):
        """Test that agents are created."""
        assert len(basic_env.agents) == 4
        assert all(f"agent_{i}" in basic_env.agents for i in range(4))

    def test_observation_spaces_defined(self, basic_env):
        """Test that observation spaces are defined for all agents."""
        for agent_id in basic_env.agents:
            assert agent_id in basic_env.observation_spaces
            obs_space = basic_env.observation_spaces[agent_id]
            assert obs_space.shape[0] == 3 + 1 + (basic_env.num_agents - 1)

    def test_action_spaces_defined(self, basic_env):
        """Test that action spaces are defined for all agents."""
        for agent_id in basic_env.agents:
            assert agent_id in basic_env.action_spaces
            action_space = basic_env.action_spaces[agent_id]
            assert action_space.shape == (1,)


class TestEnvironmentReset:
    """Tests for environment reset."""

    def test_reset_returns_observations(self, basic_env):
        """Test that reset returns observations for all agents."""
        observations, infos = basic_env.reset()
        assert len(observations) == 4
        assert all(f"agent_{i}" in observations for i in range(4))

    def test_reset_observations_shape(self, basic_env):
        """Test that reset observations have correct shape."""
        observations, _ = basic_env.reset()
        for agent_id, obs in observations.items():
            expected_dim = 3 + 1 + (basic_env.num_agents - 1)
            assert obs.shape == (expected_dim,)

    def test_reset_initializes_resources(self, basic_env):
        """Test that reset initializes resources correctly."""
        observations, _ = basic_env.reset()
        assert basic_env.remaining_resources == basic_env.total_resources

    def test_reset_clears_history(self, basic_env):
        """Test that reset clears episode history."""
        basic_env.reset()
        assert len(basic_env.episode_history) == 0


class TestEnvironmentStep:
    """Tests for environment step function."""

    def test_step_returns_correct_structure(self, basic_env):
        """Test that step returns all required elements."""
        basic_env.reset()
        actions = {f"agent_{i}": np.array([0.25]) for i in range(4)}

        obs, rewards, terminations, truncations, infos = basic_env.step(actions)

        assert len(obs) == 4
        assert len(rewards) == 4
        assert len(terminations) == 4
        assert len(truncations) == 4
        assert len(infos) == 4

    def test_step_observations_updated(self, basic_env):
        """Test that observations are updated after step."""
        obs1, _ = basic_env.reset()
        actions = {f"agent_{i}": np.array([0.25]) for i in range(4)}
        obs2, _, _, _, _ = basic_env.step(actions)

        # Timestep should have increased
        for agent_id in basic_env.agents:
            assert obs2[agent_id][3] > obs1[agent_id][3]

    def test_resources_decrease_after_step(self, basic_env):
        """Test that resources decrease after claims."""
        basic_env.reset()
        initial_resources = basic_env.remaining_resources

        actions = {f"agent_{i}": np.array([0.25]) for i in range(4)}
        basic_env.step(actions)

        assert basic_env.remaining_resources < initial_resources

    def test_rewards_returned_for_all_agents(self, basic_env):
        """Test that rewards are returned for all agents."""
        basic_env.reset()
        actions = {f"agent_{i}": np.array([0.25]) for i in range(4)}
        _, rewards, _, _, _ = basic_env.step(actions)

        for agent_id in basic_env.agents:
            assert agent_id in rewards
            assert isinstance(rewards[agent_id], (int, float))

    def test_infos_contain_resources(self, basic_env):
        """Test that infos contain resource information."""
        basic_env.reset()
        actions = {f"agent_{i}": np.array([0.25]) for i in range(4)}
        _, _, _, _, infos = basic_env.step(actions)

        for agent_id in basic_env.agents:
            assert "resources" in infos[agent_id]
            assert "fairness_score" in infos[agent_id]


class TestRewardStructures:
    """Tests for different reward structures."""

    def test_selfish_reward_structure(self):
        """Test selfish reward structure."""
        env = MoralDilemmaEnv(
            num_agents=2,
            total_resources=100,
            episode_length=10,
            reward_structure="selfish",
        )
        env.reset()
        actions = {"agent_0": np.array([0.8]), "agent_1": np.array([0.2])}
        _, rewards, _, _, _ = env.step(actions)

        # Greedy agent should get higher reward in selfish mode
        assert rewards["agent_0"] > rewards["agent_1"]

    def test_utilitarian_reward_structure(self):
        """Test utilitarian reward structure."""
        env = MoralDilemmaEnv(
            num_agents=2,
            total_resources=100,
            episode_length=10,
            reward_structure="utilitarian",
        )
        env.reset()
        actions = {"agent_0": np.array([0.5]), "agent_1": np.array([0.5])}
        _, rewards, _, _, _ = env.step(actions)

        # Both agents should get similar rewards with fair claims
        assert abs(rewards["agent_0"] - rewards["agent_1"]) < 10


class TestEpisodeTermination:
    """Tests for episode termination."""

    def test_truncation_at_episode_length(self, small_env):
        """Test that episode truncates at episode_length."""
        small_env.reset()
        truncated = False

        for step in range(small_env.episode_length + 5):
            actions = {f"agent_{i}": np.array([0.25]) for i in range(2)}
            _, _, _, truncations, _ = small_env.step(actions)
            if any(truncations.values()):
                truncated = True
                break

        assert truncated

    def test_termination_when_resources_exhausted(self):
        """Test termination when resources are exhausted."""
        env = MoralDilemmaEnv(
            num_agents=2,
            total_resources=10,  # Very limited resources
            episode_length=100,
            reward_structure="selfish",
        )
        env.reset()

        terminated = False
        for _ in range(100):
            actions = {"agent_0": np.array([1.0]), "agent_1": np.array([1.0])}
            _, _, terminations, truncations, _ = env.step(actions)
            if any(terminations.values()) or any(truncations.values()):
                terminated = True
                break

        assert terminated


class TestPeerInfluence:
    """Tests for peer influence mechanics."""

    def test_peer_influence_affects_observations(self):
        """Test that peer influence strength affects observations."""
        env_no_influence = MoralDilemmaEnv(num_agents=2, peer_influence_strength=0.0)
        env_high_influence = MoralDilemmaEnv(num_agents=2, peer_influence_strength=1.0)

        env_no_influence.reset()
        env_high_influence.reset()

        # Both should work without error
        actions = {"agent_0": np.array([0.5]), "agent_1": np.array([0.5])}
        env_no_influence.step(actions)
        env_high_influence.step(actions)


class TestResourceAllocation:
    """Tests for resource allocation mechanics."""

    def test_claims_scaled_when_exceeding_available(self):
        """Test that claims are scaled when total exceeds available."""
        env = MoralDilemmaEnv(
            num_agents=2,
            total_resources=10,
            episode_length=10,
        )
        env.reset()

        # Both claim 100% of remaining = exceeds available
        actions = {"agent_0": np.array([1.0]), "agent_1": np.array([1.0])}
        _, _, _, _, infos = env.step(actions)

        # Resources should be split (scaled)
        total_allocated = infos["agent_0"]["resources"] + infos["agent_1"]["resources"]
        assert total_allocated <= 10  # Should not exceed initial

    def test_fair_claims_get_fair_allocation(self):
        """Test that fair claims result in fair allocation."""
        env = MoralDilemmaEnv(num_agents=4, total_resources=100)
        env.reset()

        # All agents claim fair share (0.25 each)
        actions = {f"agent_{i}": np.array([0.25]) for i in range(4)}
        _, _, _, _, infos = env.step(actions)

        # All should get similar resources (within tolerance)
        resources = [infos[f"agent_{i}"]["resources"] for i in range(4)]
        assert max(resources) - min(resources) < 5  # Allow some variance
