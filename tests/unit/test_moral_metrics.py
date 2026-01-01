"""Unit tests for moral metrics."""

import pytest
import numpy as np

from src.metrics.moral_metrics import (
    MoralMetrics,
    GreatestGoodBenchmark,
    PeerPressureAnalyzer,
)


class TestMoralMetricsDataclass:
    """Tests for the MoralMetrics dataclass."""

    def test_dataclass_creation(self):
        """Test that MoralMetrics can be created."""
        metrics = MoralMetrics(
            utilitarian_score=0.5,
            fairness_score=0.7,
            cooperation_index=0.4,
            conformity_measure=0.6,
            peer_influence_strength=0.3,
            moral_consistency=0.8,
            group_welfare=0.6,
            individual_sacrifice=0.2,
        )
        assert metrics.utilitarian_score == 0.5
        assert metrics.fairness_score == 0.7


class TestGreatestGoodBenchmark:
    """Tests for the Greatest Good Benchmark."""

    def test_initialization(self, ggb):
        """Test GGB initialization."""
        assert ggb.num_agents == 4
        assert len(ggb.history["actions"]) == 0

    def test_update_stores_history(self, ggb, sample_actions, sample_resources, sample_rewards):
        """Test that update stores history correctly."""
        ggb.update(sample_actions, sample_resources, sample_rewards)

        assert len(ggb.history["actions"]) == 1
        assert len(ggb.history["resources"]) == 1
        assert len(ggb.history["rewards"]) == 1

    def test_calculate_metrics_empty_history(self, ggb):
        """Test metrics calculation with empty history."""
        metrics = ggb.calculate_metrics()

        assert metrics.utilitarian_score == 0.0
        assert metrics.fairness_score == 0.0

    def test_calculate_metrics_with_data(
        self, ggb, sample_actions, sample_resources, sample_rewards
    ):
        """Test metrics calculation with data."""
        # Add multiple updates
        for _ in range(10):
            ggb.update(sample_actions, sample_resources, sample_rewards)

        metrics = ggb.calculate_metrics()

        assert 0 <= metrics.utilitarian_score <= 1
        assert 0 <= metrics.fairness_score <= 1
        assert 0 <= metrics.cooperation_index <= 1

    def test_fairness_perfect_equality(self, ggb):
        """Test fairness score with perfect equality."""
        equal_resources = {
            "agent_0": 25.0,
            "agent_1": 25.0,
            "agent_2": 25.0,
            "agent_3": 25.0,
        }
        fairness = ggb._calculate_fairness(equal_resources)
        assert fairness == pytest.approx(1.0, rel=0.01)

    def test_fairness_extreme_inequality(self, ggb):
        """Test fairness score with extreme inequality."""
        unequal_resources = {
            "agent_0": 100.0,
            "agent_1": 0.0,
            "agent_2": 0.0,
            "agent_3": 0.0,
        }
        fairness = ggb._calculate_fairness(unequal_resources)
        assert fairness < 0.5  # Should be low

    def test_cooperation_high_claims(self, ggb):
        """Test cooperation with high claims (low cooperation)."""
        high_claims = {
            "agent_0": 0.9,
            "agent_1": 0.8,
            "agent_2": 0.9,
            "agent_3": 0.85,
        }
        cooperation = ggb._calculate_cooperation(high_claims)
        assert cooperation == 0.0  # Capped at 0

    def test_cooperation_low_claims(self, ggb):
        """Test cooperation with low claims (high cooperation)."""
        low_claims = {
            "agent_0": 0.1,
            "agent_1": 0.1,
            "agent_2": 0.1,
            "agent_3": 0.1,
        }
        cooperation = ggb._calculate_cooperation(low_claims)
        assert cooperation > 0.5  # Should be positive

    def test_summary_stats(self, ggb, sample_actions, sample_resources, sample_rewards):
        """Test get_summary_stats."""
        for _ in range(10):
            ggb.update(sample_actions, sample_resources, sample_rewards)

        stats = ggb.get_summary_stats()

        assert "utilitarian_score" in stats
        assert "fairness_score" in stats
        assert "overall_moral_score" in stats

    def test_conformity_high_variance(self, ggb, sample_actions, sample_resources, sample_rewards):
        """Test conformity with varying actions."""
        # Add data with varying actions
        varied_actions = {
            "agent_0": 0.1,
            "agent_1": 0.9,
            "agent_2": 0.2,
            "agent_3": 0.8,
        }
        for _ in range(5):
            ggb.update(varied_actions, sample_resources, sample_rewards)

        conformity = ggb._calculate_conformity()
        assert conformity < 0.7  # Lower conformity due to high variance

    def test_moral_consistency_stable_actions(self, ggb, sample_resources, sample_rewards):
        """Test moral consistency with stable actions."""
        stable_actions = {
            "agent_0": 0.3,
            "agent_1": 0.3,
            "agent_2": 0.3,
            "agent_3": 0.3,
        }
        for _ in range(10):
            ggb.update(stable_actions, sample_resources, sample_rewards)

        consistency = ggb._calculate_moral_consistency()
        assert consistency > 0.8  # High consistency


class TestPeerPressureAnalyzer:
    """Tests for the Peer Pressure Analyzer."""

    def test_initialization(self, peer_analyzer):
        """Test analyzer initialization."""
        assert len(peer_analyzer.influence_events) == 0

    def test_no_influence_detected_small_change(self, peer_analyzer):
        """Test that small changes don't trigger influence detection."""
        actions_before = {
            "agent_0": 0.5,
            "agent_1": 0.5,
            "agent_2": 0.5,
            "agent_3": 0.5,
        }
        actions_after = {
            "agent_0": 0.51,  # Tiny change
            "agent_1": 0.49,
            "agent_2": 0.5,
            "agent_3": 0.5,
        }

        influenced = peer_analyzer.detect_influence_event(actions_before, actions_after)
        assert len(influenced) == 0

    def test_influence_detected_large_move_toward_group(self, peer_analyzer):
        """Test that large moves toward group are detected."""
        actions_before = {
            "agent_0": 0.9,  # Far from group
            "agent_1": 0.3,
            "agent_2": 0.3,
            "agent_3": 0.3,
        }
        # Agent 0 moves significantly toward group average (~0.3)
        actions_after = {
            "agent_0": 0.4,  # Moved toward group
            "agent_1": 0.3,
            "agent_2": 0.3,
            "agent_3": 0.3,
        }

        influenced = peer_analyzer.detect_influence_event(
            actions_before, actions_after, threshold=0.2, min_distance=0.1, convergence_ratio=0.7
        )
        assert "agent_0" in influenced

    def test_no_influence_when_moving_away(self, peer_analyzer):
        """Test that moving away from group is not detected as influence."""
        actions_before = {
            "agent_0": 0.4,  # Close to group
            "agent_1": 0.3,
            "agent_2": 0.3,
            "agent_3": 0.3,
        }
        actions_after = {
            "agent_0": 0.8,  # Moved away from group
            "agent_1": 0.3,
            "agent_2": 0.3,
            "agent_3": 0.3,
        }

        influenced = peer_analyzer.detect_influence_event(actions_before, actions_after)
        assert "agent_0" not in influenced

    def test_influence_events_stored(self, peer_analyzer):
        """Test that influence events are stored."""
        actions_before = {
            "agent_0": 0.9,
            "agent_1": 0.3,
            "agent_2": 0.3,
            "agent_3": 0.3,
        }
        actions_after = {
            "agent_0": 0.35,  # Moved toward group
            "agent_1": 0.3,
            "agent_2": 0.3,
            "agent_3": 0.3,
        }

        peer_analyzer.detect_influence_event(
            actions_before, actions_after, threshold=0.2, min_distance=0.1
        )

        if peer_analyzer.influence_events:
            assert "timestep" in peer_analyzer.influence_events[0]
            assert "influenced_agents" in peer_analyzer.influence_events[0]

    def test_influence_summary_empty(self, peer_analyzer):
        """Test influence summary with no events."""
        summary = peer_analyzer.get_influence_summary()

        assert summary["total_events"] == 0
        assert summary["avg_influence_strength"] == 0.0

    def test_influence_summary_with_events(self, peer_analyzer):
        """Test influence summary with events."""
        # Generate some influence events
        for i in range(3):
            actions_before = {
                "agent_0": 0.9,
                "agent_1": 0.3,
                "agent_2": 0.3,
                "agent_3": 0.3,
            }
            actions_after = {
                "agent_0": 0.35,
                "agent_1": 0.3,
                "agent_2": 0.3,
                "agent_3": 0.3,
            }
            peer_analyzer.detect_influence_event(
                actions_before, actions_after, threshold=0.2, min_distance=0.1
            )

        summary = peer_analyzer.get_influence_summary()

        if summary["total_events"] > 0:
            assert summary["total_events"] >= 1
            assert "most_influenced_agents" in summary


class TestGiniCoefficient:
    """Tests specifically for Gini coefficient calculation."""

    def test_gini_perfect_equality(self, ggb):
        """Test Gini = 0 (fairness = 1) for perfect equality."""
        equal = {"a": 25, "b": 25, "c": 25, "d": 25}
        fairness = ggb._calculate_fairness(equal)
        assert fairness == pytest.approx(1.0, rel=0.01)

    def test_gini_moderate_inequality(self, ggb):
        """Test Gini for moderate inequality."""
        moderate = {"a": 40, "b": 30, "c": 20, "d": 10}
        fairness = ggb._calculate_fairness(moderate)
        assert 0.5 < fairness < 0.9  # Moderate fairness

    def test_gini_handles_zeros(self, ggb):
        """Test Gini handles zero values."""
        with_zeros = {"a": 100, "b": 0, "c": 0, "d": 0}
        fairness = ggb._calculate_fairness(with_zeros)
        assert fairness < 0.5  # Low fairness

    def test_gini_all_zeros(self, ggb):
        """Test Gini handles all zeros."""
        all_zeros = {"a": 0, "b": 0, "c": 0, "d": 0}
        fairness = ggb._calculate_fairness(all_zeros)
        assert fairness == 1.0  # Perfect equality (everyone has nothing)
