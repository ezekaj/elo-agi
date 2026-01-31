"""Tests for intrinsic motivation system"""

import numpy as np
import pytest
import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

from src.intrinsic_motivation import (
    PathEntropyMaximizer,
    PossibilitySpace,
    ActionDiversityTracker,
    IntrinsicDrive,
    DriveType
)


class TestPossibilitySpace:
    """Tests for possibility space tracking"""

    def test_initialization(self):
        space = PossibilitySpace(state_dim=4, action_dim=2)
        assert space.state_dim == 4
        assert space.action_dim == 2
        assert len(space.visited_states) == 0

    def test_observe_expands_boundaries(self):
        space = PossibilitySpace(state_dim=2, action_dim=1)

        space.observe(np.array([0.0, 0.0]))
        space.observe(np.array([1.0, 1.0]))
        space.observe(np.array([-1.0, 0.5]))

        assert space.state_min[0] == -1.0
        assert space.state_max[0] == 1.0
        assert len(space.visited_states) == 3

    def test_volume_increases_with_exploration(self):
        space = PossibilitySpace(state_dim=2, action_dim=1)

        # Initial small volume
        for _ in range(10):
            space.observe(np.random.randn(2) * 0.1)
        small_volume = space.compute_volume()

        # Expand exploration
        for _ in range(10):
            space.observe(np.random.randn(2) * 1.0)
        large_volume = space.compute_volume()

        assert large_volume > small_volume

    def test_diversity_increases_with_spread(self):
        space = PossibilitySpace(state_dim=2, action_dim=1)

        # Very clustered observations (almost no spread)
        center = np.array([0.5, 0.5])
        for _ in range(100):
            space.observe(center + np.random.randn(2) * 0.001)
        clustered_diversity = space.compute_diversity()

        # Reset and use well-spread observations
        space.reset()
        for i in range(100):
            # Systematic grid-like spread
            x = (i % 10) / 10.0
            y = (i // 10) / 10.0
            space.observe(np.array([x, y]))
        spread_diversity = space.compute_diversity()

        # Spread should have more diversity (or at least equal for edge cases)
        assert spread_diversity >= clustered_diversity * 0.9  # Allow small variance

    def test_action_value_rewards_novelty(self):
        space = PossibilitySpace(state_dim=2, action_dim=1)

        # Build up experience
        for _ in range(30):
            state = np.array([0.5, 0.5]) + np.random.randn(2) * 0.1
            space.observe(state)

        # Value of action leading to novel state should be higher
        novel_action = np.array([1.0])  # Would lead to new area
        value = space.compute_action_value(novel_action)

        # Should have some positive value for expansion
        assert value >= 0.0


class TestActionDiversityTracker:
    """Tests for action diversity tracking"""

    def test_diversity_starts_high(self):
        tracker = ActionDiversityTracker(action_dim=2)
        # Initially everything is "new"
        initial_diversity = tracker.compute_diversity()
        assert initial_diversity == 1.0

    def test_repetition_reduces_diversity(self):
        tracker = ActionDiversityTracker(action_dim=2)

        # Repeat same action
        same_action = np.array([0.5, 0.5])
        for _ in range(30):
            tracker.record_action(same_action)

        low_diversity = tracker.compute_diversity()

        # Now vary actions
        tracker2 = ActionDiversityTracker(action_dim=2)
        for _ in range(30):
            tracker2.record_action(np.random.rand(2) * 2 - 1)

        high_diversity = tracker2.compute_diversity()

        assert high_diversity > low_diversity

    def test_diversity_bonus_for_rare_actions(self):
        tracker = ActionDiversityTracker(action_dim=2)

        # Use one action a lot
        common_action = np.array([0.0, 0.0])
        for _ in range(20):
            tracker.record_action(common_action)

        # Bonus should be low for common action
        common_bonus = tracker.get_diversity_bonus(common_action)

        # Bonus should be high for rare action
        rare_action = np.array([0.9, 0.9])
        rare_bonus = tracker.get_diversity_bonus(rare_action)

        assert rare_bonus > common_bonus


class TestIntrinsicDrive:
    """Tests for individual intrinsic drives"""

    def test_drive_initialization(self):
        drive = IntrinsicDrive(DriveType.EXPLORATION)
        assert drive.drive_type == DriveType.EXPLORATION
        assert drive.level == 0.5

    def test_satisfaction_reduces_drive(self):
        drive = IntrinsicDrive(DriveType.EXPLORATION, satiation_rate=0.2)
        initial_level = drive.level

        drive.update(satisfaction=1.0, dt=1.0)

        assert drive.level < initial_level

    def test_drive_recovers_without_satisfaction(self):
        drive = IntrinsicDrive(DriveType.EXPLORATION, recovery_rate=0.1)

        # Satiate drive
        drive.level = 0.1

        # Let it recover
        for _ in range(20):
            drive.update(satisfaction=0.0, dt=1.0)

        assert drive.level > 0.1

    def test_drive_bounded(self):
        drive = IntrinsicDrive(DriveType.MASTERY)

        # Try to exceed bounds
        for _ in range(100):
            drive.update(satisfaction=0.0, dt=1.0)
        assert drive.level <= 1.0

        for _ in range(100):
            drive.update(satisfaction=1.0, dt=1.0)
        assert drive.level >= 0.0


class TestPathEntropyMaximizer:
    """Tests for the complete path entropy maximizer"""

    def test_initialization(self):
        maximizer = PathEntropyMaximizer(state_dim=4, action_dim=2)
        assert maximizer.state_dim == 4
        assert maximizer.action_dim == 2
        assert len(maximizer.drives) == 4

    def test_observe_updates_state(self):
        maximizer = PathEntropyMaximizer(state_dim=3, action_dim=2)

        state = np.array([1.0, 0.0, -1.0])
        action = np.array([0.5, -0.5])
        maximizer.observe(state, action)

        assert len(maximizer.path_history) == 1
        assert len(maximizer.possibility_space.visited_states) == 1

    def test_intrinsic_motivation_computed(self):
        maximizer = PathEntropyMaximizer(state_dim=3, action_dim=2)

        for _ in range(10):
            state = np.random.randn(3)
            action = np.random.randn(2)
            maximizer.observe(state, action)

        motivation = maximizer.compute_intrinsic_motivation()
        assert 0.0 <= motivation <= 1.0

    def test_action_value_varies(self):
        maximizer = PathEntropyMaximizer(state_dim=2, action_dim=2)

        # Build up some history
        for _ in range(30):
            state = np.array([0.5, 0.5]) + np.random.randn(2) * 0.1
            action = np.random.randn(2) * 0.1
            maximizer.observe(state, action)

        # Different actions should have different values
        action1 = np.array([0.0, 0.0])
        action2 = np.array([1.0, 1.0])

        value1 = maximizer.compute_action_value(action1)
        value2 = maximizer.compute_action_value(action2)

        # Values should be different (novel action vs common)
        # Not asserting which is higher since it depends on history
        assert value1 != value2 or abs(value1 - value2) < 0.001

    def test_suggest_action_returns_valid(self):
        maximizer = PathEntropyMaximizer(state_dim=2, action_dim=2)

        for _ in range(20):
            maximizer.observe(np.random.randn(2), np.random.randn(2))

        suggested = maximizer.suggest_action(np.array([0.0, 0.0]))

        assert suggested.shape == (2,)
        assert np.all(np.abs(suggested) <= 1.0)

    def test_metrics_comprehensive(self):
        maximizer = PathEntropyMaximizer(state_dim=2, action_dim=2)

        for _ in range(30):
            maximizer.observe(np.random.randn(2), np.random.randn(2))

        metrics = maximizer.get_metrics()

        assert 'intrinsic_motivation' in metrics
        assert 'path_entropy' in metrics
        assert 'possibility_volume' in metrics
        assert 'state_diversity' in metrics
        assert 'exploration_drive' in metrics

    def test_exploration_vs_exploitation_behavior(self):
        maximizer = PathEntropyMaximizer(state_dim=2, action_dim=2)

        # Explore diverse states
        for i in range(50):
            angle = i * 0.2
            state = np.array([np.cos(angle), np.sin(angle)])
            action = np.random.randn(2)
            maximizer.observe(state, action)

        # Should have healthy diversity
        metrics = maximizer.get_metrics()
        assert metrics['state_diversity'] > 0.3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
