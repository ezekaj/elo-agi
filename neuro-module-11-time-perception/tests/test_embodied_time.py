"""Tests for embodied time perception"""

import numpy as np
import pytest
import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

from src.embodied_time import (
    InteroceptiveTimer,
    MotorTimer,
    BodyEnvironmentCoupler,
    EmbodiedTimeSystem,
    BodyState
)


class TestInteroceptiveTimer:
    """Tests for interoceptive timing"""

    def test_initialization(self):
        """Test timer initialization"""
        timer = InteroceptiveTimer(interoceptive_accuracy=0.8)

        assert timer.interoceptive_accuracy == 0.8
        assert timer.heartbeat_count == 0

    def test_count_heartbeats(self):
        """Test heartbeat counting"""
        timer = InteroceptiveTimer()
        timer.body_state.heart_rate = 60  # 1 beat per second

        beats = timer.count_heartbeats(duration=10.0)

        # Should be approximately 10 beats
        assert 7 < beats < 14

    def test_count_breaths(self):
        """Test breath counting"""
        timer = InteroceptiveTimer()
        timer.body_state.breathing_rate = 12  # 12 per minute = 0.2 per second

        breaths = timer.count_breaths(duration=30.0)

        # Should be approximately 6 breaths
        assert 3 < breaths < 10

    def test_duration_from_heartbeats(self):
        """Test duration estimation from heartbeats"""
        timer = InteroceptiveTimer()

        estimate = timer.estimate_duration_from_heartbeats(10.0)

        # Should be reasonable estimate
        assert 5 < estimate < 20

    def test_combined_estimate(self):
        """Test combined heartbeat/breathing estimate"""
        timer = InteroceptiveTimer()

        estimate = timer.combined_estimate(10.0)

        assert estimate > 0

    def test_interoceptive_accuracy_matters(self):
        """Test that accuracy affects reliability"""
        high_accuracy = InteroceptiveTimer(interoceptive_accuracy=0.95)
        low_accuracy = InteroceptiveTimer(interoceptive_accuracy=0.5)

        high_estimates = [high_accuracy.count_heartbeats(10.0) for _ in range(10)]
        high_accuracy.reset()
        low_estimates = [low_accuracy.count_heartbeats(10.0) for _ in range(10)]

        # High accuracy should have lower variance
        # (reset between each to avoid accumulation issues)
        assert np.std(high_estimates) <= np.std(low_estimates) + 1

    def test_arousal_affects_body_state(self):
        """Test that arousal changes heart rate"""
        timer = InteroceptiveTimer()
        baseline_hr = timer.body_state.heart_rate

        timer.set_arousal(0.9)

        assert timer.body_state.heart_rate > baseline_hr


class TestMotorTimer:
    """Tests for motor timing"""

    def test_initialization(self):
        """Test timer initialization"""
        timer = MotorTimer(motor_precision=0.9)

        assert timer.motor_precision == 0.9
        assert timer.action_count == 0

    def test_time_through_action(self):
        """Test timing through actions"""
        timer = MotorTimer(base_tempo=2.0)

        estimate = timer.time_through_action(5.0)

        # Should be reasonable
        assert 3 < estimate < 8

    def test_movement_sequence(self):
        """Test timing a movement sequence"""
        timer = MotorTimer()

        movements = [1.0, 2.0, 1.5, 0.5]
        estimated, actual = timer.time_movement_sequence(movements)

        assert actual == 5.0
        assert 3 < estimated < 8

    def test_tap_rhythm(self):
        """Test producing a rhythm"""
        timer = MotorTimer(motor_precision=0.9)

        intervals = timer.tap_rhythm(target_interval=0.5, n_taps=10)

        assert len(intervals) == 10
        # All should be close to target
        for interval in intervals:
            assert 0.3 < interval < 0.7

    def test_synchronization(self):
        """Test synchronization to external beat"""
        timer = MotorTimer()

        taps, asynchrony = timer.synchronize_to_beat(
            beat_interval=0.5,
            n_beats=10
        )

        assert len(taps) == 10
        assert asynchrony < 0.2  # Reasonable synchronization


class TestBodyEnvironmentCoupler:
    """Tests for body-environment coupling"""

    def test_initialization(self):
        """Test coupler initialization"""
        coupler = BodyEnvironmentCoupler()

        assert coupler.body_rhythm == 1.0
        assert coupler.environmental_rhythm is None

    def test_entrainment_calculation(self):
        """Test entrainment factor calculation"""
        coupler = BodyEnvironmentCoupler()

        coupler.set_body_rhythm(1.0)
        coupler.set_environmental_rhythm(1.0)

        entrainment = coupler.compute_entrainment()

        # Perfect 1:1 should have high entrainment
        assert entrainment > 0.8

    def test_harmonic_entrainment(self):
        """Test entrainment for harmonic relationships"""
        coupler = BodyEnvironmentCoupler()

        coupler.set_body_rhythm(2.0)
        coupler.set_environmental_rhythm(1.0)

        entrainment = coupler.compute_entrainment()

        # 2:1 harmonic should have good entrainment
        assert entrainment > 0.6

    def test_non_harmonic_low_entrainment(self):
        """Test low entrainment for non-harmonic"""
        coupler = BodyEnvironmentCoupler()

        coupler.set_body_rhythm(1.0)
        coupler.set_environmental_rhythm(1.4)  # Not harmonic

        entrainment = coupler.compute_entrainment()

        # Should be lower than harmonic relationships
        assert entrainment < 0.7

    def test_timing_modulation(self):
        """Test timing modulation by coupling"""
        coupler = BodyEnvironmentCoupler()

        estimate, entrainment = coupler.modulate_timing(
            duration=10.0,
            body_rhythm=1.0,
            env_rhythm=1.0
        )

        assert estimate > 0
        assert entrainment > 0


class TestEmbodiedTimeSystem:
    """Tests for integrated embodied time system"""

    def test_initialization(self):
        """Test system initialization"""
        system = EmbodiedTimeSystem()

        assert system.interoceptive is not None
        assert system.motor is not None
        assert system.coupler is not None

    def test_duration_estimation(self):
        """Test duration estimation"""
        system = EmbodiedTimeSystem()

        estimate, components = system.estimate_duration(
            actual_duration=10.0,
            movement_present=False
        )

        assert estimate > 0
        assert 'interoceptive' in components

    def test_with_movement(self):
        """Test estimation with movement"""
        system = EmbodiedTimeSystem()

        estimate, components = system.estimate_duration(
            actual_duration=10.0,
            movement_present=True
        )

        assert 'motor' in components

    def test_with_external_rhythm(self):
        """Test estimation with external rhythm"""
        system = EmbodiedTimeSystem()

        estimate, components = system.estimate_duration(
            actual_duration=10.0,
            external_rhythm=1.0
        )

        assert 'coupling' in components
        assert 'entrainment' in components

    def test_body_state_update(self):
        """Test updating body state"""
        system = EmbodiedTimeSystem()

        state = BodyState(
            heart_rate=90,
            breathing_rate=20,
            metabolic_rate=1.2
        )

        system.set_body_state(state)

        assert system.interoceptive.body_state.heart_rate == 90

    def test_statistics(self):
        """Test statistics collection"""
        system = EmbodiedTimeSystem()

        system.estimate_duration(10.0)
        system.estimate_duration(5.0, movement_present=True)

        stats = system.get_statistics()

        assert stats['heartbeat_count'] > 0
        assert stats['action_count'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
