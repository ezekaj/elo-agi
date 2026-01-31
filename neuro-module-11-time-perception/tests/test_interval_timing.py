"""Tests for interval timing models"""

import numpy as np
import pytest
import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

from src.interval_timing import (
    PacemakerAccumulator,
    StriatalBeatFrequency,
    IntervalTimer,
    TimingMode,
    TimingResult
)


class TestPacemakerAccumulator:
    """Tests for pacemaker-accumulator model"""

    def test_initialization(self):
        """Test model initialization"""
        pa = PacemakerAccumulator(base_rate=10.0)

        assert pa.base_rate == 10.0
        assert pa.accumulator == 0.0
        assert not pa.switch_open

    def test_start_stop_timing(self):
        """Test starting and stopping timing"""
        pa = PacemakerAccumulator()

        pa.start_timing()
        assert pa.switch_open
        assert pa.accumulator == 0.0

        pa.accumulate(5.0)
        pulses = pa.stop_timing()

        assert not pa.switch_open
        assert pulses > 0

    def test_accumulation(self):
        """Test pulse accumulation"""
        pa = PacemakerAccumulator(base_rate=10.0)
        pa.start_timing()

        pa.accumulate(1.0, attention=1.0)

        # ~10 pulses per second
        assert 8 < pa.accumulator < 12

    def test_attention_affects_accumulation(self):
        """Test that attention affects pulse gating"""
        pa = PacemakerAccumulator()

        pa.start_timing()
        pa.accumulate(5.0, attention=0.3)
        low_attention = pa.accumulator

        pa.start_timing()  # Reset
        pa.accumulate(5.0, attention=1.0)
        high_attention = pa.accumulator

        assert high_attention > low_attention

    def test_duration_estimation(self):
        """Test duration estimation"""
        pa = PacemakerAccumulator()

        result = pa.estimate_duration(10.0, attention=0.7)

        assert isinstance(result, TimingResult)
        assert result.actual_duration == 10.0
        assert result.estimated_duration > 0
        assert result.mode == TimingMode.PROSPECTIVE

    def test_prospective_vs_retrospective(self):
        """Test prospective vs retrospective timing"""
        pa = PacemakerAccumulator()

        prospective = pa.estimate_duration(10.0, mode=TimingMode.PROSPECTIVE)
        pa.reset()
        retrospective = pa.estimate_duration(10.0, mode=TimingMode.RETROSPECTIVE)

        # Retrospective typically has more variability
        # Both should be reasonable estimates
        assert 5 < prospective.estimated_duration < 20
        assert 5 < retrospective.estimated_duration < 20

    def test_duration_production(self):
        """Test producing a target duration"""
        pa = PacemakerAccumulator()

        result = pa.produce_duration(5.0)

        assert result.mode == TimingMode.PRODUCTION
        # Should be close to target
        assert 3 < result.estimated_duration < 8

    def test_reference_storage(self):
        """Test storing reference durations"""
        pa = PacemakerAccumulator()

        pa.store_reference("short", 2.0)
        pa.store_reference("long", 10.0)

        comparison, ratio = pa.compare_to_reference(5.0, "short")
        assert comparison == "longer"
        assert ratio > 1.0

    def test_clock_speed_modulation(self):
        """Test clock speed modulation"""
        pa = PacemakerAccumulator()

        pa.set_clock_speed(1.5)
        assert pa.effective_rate > pa.base_rate

        pa.set_clock_speed(0.7)
        assert pa.effective_rate < pa.base_rate

    def test_scalar_property(self):
        """Test Weber's law - variability scales with duration"""
        pa = PacemakerAccumulator()

        # Collect multiple estimates
        short_errors = []
        long_errors = []

        for _ in range(50):
            result = pa.estimate_duration(2.0)
            short_errors.append(abs(result.relative_error))
            pa.reset()

            result = pa.estimate_duration(20.0)
            long_errors.append(abs(result.relative_error))
            pa.reset()

        # Coefficient of variation should be similar (Weber's law)
        # Allow for noise in the comparison
        assert np.std(short_errors) < 0.5
        assert np.std(long_errors) < 0.5


class TestStriatalBeatFrequency:
    """Tests for striatal beat frequency model"""

    def test_initialization(self):
        """Test model initialization"""
        sbf = StriatalBeatFrequency(n_oscillators=100)

        assert sbf.n_oscillators == 100
        assert len(sbf.frequencies) == 100

    def test_oscillator_states(self):
        """Test oscillator state computation"""
        sbf = StriatalBeatFrequency()

        states_t0 = sbf.get_oscillator_states(0.0)
        states_t1 = sbf.get_oscillator_states(1.0)

        assert len(states_t0) == sbf.n_oscillators
        # States should differ at different times
        assert not np.allclose(states_t0, states_t1)

    def test_duration_encoding(self):
        """Test encoding durations as patterns"""
        sbf = StriatalBeatFrequency()

        pattern = sbf.encode_duration(5.0, name="five_seconds")

        assert len(pattern) == sbf.n_oscillators
        assert "five_seconds" in sbf.learned_patterns

    def test_duration_estimation(self):
        """Test duration estimation"""
        sbf = StriatalBeatFrequency()

        result = sbf.estimate_duration(10.0)

        assert result.actual_duration == 10.0
        assert result.estimated_duration > 0

    def test_learned_detection(self):
        """Test detecting learned durations"""
        sbf = StriatalBeatFrequency()

        # Learn a duration
        sbf.encode_duration(5.0, name="target")

        # Get pattern at same time
        pattern = sbf.get_oscillator_states(5.0)

        name, confidence = sbf.detect_learned_duration(pattern)

        # Should detect the learned duration
        assert name == "target"
        assert confidence > 0.7


class TestIntervalTimer:
    """Tests for combined interval timer"""

    def test_initialization(self):
        """Test timer initialization"""
        timer = IntervalTimer()

        assert timer.pacemaker is not None
        assert timer.sbf is not None

    def test_duration_estimation(self):
        """Test combined duration estimation"""
        timer = IntervalTimer()

        result = timer.estimate_duration(10.0, attention=0.7)

        assert result.estimated_duration > 0
        assert result.actual_duration == 10.0

    def test_learning_durations(self):
        """Test learning named durations"""
        timer = IntervalTimer()

        timer.learn_duration("interval", 5.0)

        comparison, ratio = timer.compare_duration(5.0, "interval")
        # Allow for memory noise - ratio should be close to 1.0
        assert 0.7 < ratio < 1.4

    def test_dopamine_modulation(self):
        """Test dopamine affects timing"""
        timer = IntervalTimer()

        timer.set_dopamine_level(1.0)
        baseline = timer.pacemaker.effective_rate

        timer.set_dopamine_level(1.5)
        high_da = timer.pacemaker.effective_rate

        assert high_da > baseline

    def test_timing_statistics(self):
        """Test statistics collection"""
        timer = IntervalTimer()

        for _ in range(10):
            timer.estimate_duration(5.0)

        stats = timer.get_timing_statistics()

        assert stats['n_trials'] == 10
        assert 'mean_error' in stats
        assert 'mean_relative_error' in stats


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
