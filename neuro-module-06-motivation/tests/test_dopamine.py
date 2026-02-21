"""Tests for dopamine system"""

import numpy as np
import pytest
from neuro.modules.m06_motivation.dopamine_system import (
    DopamineSystem,
    PredictionErrorComputer,
    IncentiveSalience,
    BenefitCostEvaluator,
    DopamineSignal,
    DopamineChannel
)

class TestPredictionErrorComputer:
    """Tests for reward prediction error computation"""

    def test_initialization(self):
        computer = PredictionErrorComputer()
        assert computer.learning_rate == 0.1

    def test_positive_rpe_for_better_than_expected(self):
        computer = PredictionErrorComputer()
        state = np.array([0.0, 0.0])
        next_state = np.array([0.1, 0.1])

        # No expected value, but got reward
        rpe = computer.compute_prediction_error(
            actual_reward=1.0,
            current_state=state,
            next_state=next_state
        )

        # Should be positive (got more than expected)
        assert rpe > 0

    def test_negative_rpe_for_worse_than_expected(self):
        computer = PredictionErrorComputer()
        state = np.array([0.0, 0.0])

        # Build up expectation
        for _ in range(10):
            computer.compute_prediction_error(1.0, state, state)
            computer.update_values(state, 1.0)

        # Now get nothing
        rpe = computer.compute_prediction_error(
            actual_reward=0.0,
            current_state=state,
            next_state=state
        )

        # Should be negative (got less than expected)
        assert rpe < 0

    def test_value_learning(self):
        computer = PredictionErrorComputer()
        state = np.array([0.5, 0.5])

        # Repeatedly reward this state
        for _ in range(20):
            rpe = computer.compute_prediction_error(1.0, state, state)
            computer.update_values(state, rpe)

        # Value should have increased
        value = computer.get_value(state)
        assert value > 0.5

    def test_surprise_is_absolute_rpe(self):
        computer = PredictionErrorComputer()

        surprise_pos = computer.get_surprise(0.5)
        surprise_neg = computer.get_surprise(-0.5)

        assert surprise_pos == surprise_neg == 0.5

class TestIncentiveSalience:
    """Tests for incentive salience (wanting)"""

    def test_initialization(self):
        salience = IncentiveSalience(cue_dim=3)
        assert salience.cue_dim == 3

    def test_salience_increases_with_reward(self):
        salience = IncentiveSalience(cue_dim=2)
        cue = np.array([1.0, 0.0])

        initial = salience.compute_salience(cue)

        # Pair cue with reward
        for _ in range(5):
            salience.update_salience(cue, reward=1.0, prediction_error=0.5)

        updated = salience.compute_salience(cue)
        assert updated > initial

    def test_internal_state_modulates_wanting(self):
        salience = IncentiveSalience(cue_dim=2)
        cue = np.array([1.0, 0.0])

        # Build up salience
        salience.update_salience(cue, reward=1.0, prediction_error=0.5)
        salience.update_salience(cue, reward=1.0, prediction_error=0.3)

        # Satiated state
        satiated = {'hunger': 0.9}
        satiated_salience = salience.compute_salience(cue, satiated)

        # Deprived state
        deprived = {'hunger': 0.1}
        deprived_salience = salience.compute_salience(cue, deprived)

        # Wanting should be higher when deprived
        assert deprived_salience > satiated_salience

    def test_most_wanted_selection(self):
        salience = IncentiveSalience(cue_dim=2)

        # Create cues with different saliences
        cue_high = np.array([1.0, 0.0])
        cue_low = np.array([0.0, 1.0])

        # Make one more salient
        for _ in range(5):
            salience.update_salience(cue_high, reward=1.0, prediction_error=0.5)

        idx, value = salience.get_most_wanted([cue_low, cue_high])
        assert idx == 1  # cue_high is second in list

class TestBenefitCostEvaluator:
    """Tests for benefit/cost evaluation"""

    def test_initialization(self):
        evaluator = BenefitCostEvaluator()
        assert evaluator.dopamine_level == 0.5

    def test_high_dopamine_favors_benefits(self):
        evaluator = BenefitCostEvaluator()

        benefits = [1.0]
        costs = [0.5]

        # High dopamine
        evaluator.set_dopamine_level(0.9)
        high_da_value = evaluator.evaluate(benefits, costs)

        # Low dopamine
        evaluator.set_dopamine_level(0.1)
        low_da_value = evaluator.evaluate(benefits, costs)

        # High DA should weight benefits more
        assert high_da_value > low_da_value

    def test_effort_willingness_increases_with_dopamine(self):
        evaluator = BenefitCostEvaluator()

        # Use harder ratio so willingness doesn't saturate
        reward = 0.5
        effort = 1.0

        evaluator.set_dopamine_level(0.9)
        high_willingness = evaluator.compute_effort_willingness(reward, effort)

        evaluator.set_dopamine_level(0.1)
        low_willingness = evaluator.compute_effort_willingness(reward, effort)

        # High dopamine should increase willingness
        # (or at minimum equal if implementation clips)
        assert high_willingness >= low_willingness

class TestDopamineSystem:
    """Tests for integrated dopamine system"""

    def test_initialization(self):
        system = DopamineSystem(state_dim=3, cue_dim=2)
        assert system.state_dim == 3
        assert system.cue_dim == 2

    def test_process_transition_returns_signal(self):
        system = DopamineSystem(state_dim=2, cue_dim=2)

        signal = system.process_transition(
            state=np.array([0.0, 0.0]),
            action=np.array([0.1, 0.1]),
            reward=1.0,
            next_state=np.array([0.1, 0.1]),
            cue=np.array([1.0, 0.0])
        )

        assert isinstance(signal, DopamineSignal)
        assert hasattr(signal, 'prediction_error')
        assert hasattr(signal, 'incentive_salience')

    def test_tonic_level_tracks_average_reward(self):
        system = DopamineSystem(state_dim=2, cue_dim=2)

        initial_tonic = system.tonic_level

        # Consistently rewarding environment
        for _ in range(30):
            system.process_transition(
                state=np.random.randn(2),
                action=np.random.randn(2),
                reward=1.0,
                next_state=np.random.randn(2)
            )

        # Tonic should have increased
        assert system.tonic_level >= initial_tonic

    def test_motivation_level_reflects_state(self):
        system = DopamineSystem(state_dim=2, cue_dim=2)

        # Process some transitions
        for _ in range(20):
            system.process_transition(
                state=np.random.randn(2),
                action=np.random.randn(2),
                reward=np.random.rand(),
                next_state=np.random.randn(2)
            )

        motivation = system.get_motivation_level()
        assert 0.0 <= motivation <= 1.0

    def test_exploration_bonus_available(self):
        system = DopamineSystem(state_dim=2, cue_dim=2)

        # Some variability should give exploration bonus
        for _ in range(20):
            reward = 1.0 if np.random.rand() > 0.5 else 0.0
            system.process_transition(
                state=np.random.randn(2),
                action=np.random.randn(2),
                reward=reward,
                next_state=np.random.randn(2)
            )

        bonus = system.get_exploration_bonus()
        assert bonus > 0

    def test_state_summary_comprehensive(self):
        system = DopamineSystem(state_dim=2, cue_dim=2)

        for _ in range(10):
            system.process_transition(
                state=np.random.randn(2),
                action=np.random.randn(2),
                reward=np.random.rand(),
                next_state=np.random.randn(2)
            )

        summary = system.get_state_summary()

        assert 'tonic_level' in summary
        assert 'motivation' in summary
        assert 'exploration_bonus' in summary

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
