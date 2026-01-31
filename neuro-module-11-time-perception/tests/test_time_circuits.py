"""Tests for neural time circuits"""

import numpy as np
import pytest
import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

from src.time_circuits import (
    Insula,
    SMA,
    BasalGanglia,
    Cerebellum,
    TimeCircuit,
    TimingScale,
    TemporalSignal
)


class TestInsula:
    """Tests for insula interoceptive timing"""

    def test_initialization(self):
        """Test insula initialization"""
        insula = Insula(heartbeat_rate=70.0, breathing_rate=15.0)

        assert insula.heartbeat_rate == 70.0
        assert insula.breathing_rate == 15.0
        assert insula.accumulated_heartbeats == 0

    def test_interoceptive_signal(self):
        """Test interoceptive signal processing"""
        insula = Insula()

        signal = insula.process_interoceptive_signal(duration=10.0, arousal_level=0.5)

        assert signal.source == "insula"
        assert signal.duration_estimate > 0
        assert 0 <= signal.confidence <= 1

    def test_arousal_affects_time(self):
        """Test that high arousal slows time perception"""
        insula = Insula()

        low_arousal = insula.process_interoceptive_signal(10.0, arousal_level=0.2)
        insula.reset()
        high_arousal = insula.process_interoceptive_signal(10.0, arousal_level=0.9)

        # High arousal should produce longer perceived duration
        assert high_arousal.duration_estimate > low_arousal.duration_estimate

    def test_body_rhythm(self):
        """Test body rhythm parameters"""
        insula = Insula(heartbeat_rate=80.0)

        rhythm = insula.get_body_rhythm()

        assert rhythm['heartbeat_rate'] == 80.0
        assert rhythm['heartbeat_period'] == 60.0 / 80.0


class TestSMA:
    """Tests for SMA motor timing"""

    def test_initialization(self):
        """Test SMA initialization"""
        sma = SMA(motor_precision=0.9)

        assert sma.motor_precision == 0.9
        assert sma.action_count == 0

    def test_motor_sequence_timing(self):
        """Test timing through motor sequences"""
        sma = SMA()

        signal = sma.time_motor_sequence(duration=5.0, movement_complexity=0.7)

        assert signal.source == "sma"
        assert signal.duration_estimate > 0
        assert signal.confidence > 0

    def test_complexity_affects_precision(self):
        """Test that more complex movements improve timing"""
        sma = SMA()

        simple = sma.time_motor_sequence(5.0, movement_complexity=0.2)
        sma.reset()
        complex_mov = sma.time_motor_sequence(5.0, movement_complexity=0.9)

        # Complex movements should have higher confidence
        assert complex_mov.confidence > simple.confidence

    def test_environment_synchronization(self):
        """Test synchronization with external rhythm"""
        sma = SMA()

        signal = sma.coordinate_body_environment(
            external_rhythm=1.0,  # 1 Hz
            duration=5.0
        )

        assert signal.confidence > sma.motor_precision  # Sync improves precision


class TestBasalGanglia:
    """Tests for basal ganglia interval timing"""

    def test_initialization(self):
        """Test basal ganglia initialization"""
        bg = BasalGanglia(dopamine_level=1.0)

        assert bg.dopamine_level == 1.0
        assert bg.accumulated_pulses == 0

    def test_interval_timing(self):
        """Test interval timing"""
        bg = BasalGanglia()

        signal = bg.time_interval(duration=10.0, attention_level=0.7)

        assert signal.source == "basal_ganglia"
        assert signal.duration_estimate > 0

    def test_attention_lengthens_duration(self):
        """Test that more attention increases perceived duration"""
        bg = BasalGanglia()

        low_attention = bg.time_interval(10.0, attention_level=0.2)
        bg.accumulated_pulses = 0  # Reset accumulator
        high_attention = bg.time_interval(10.0, attention_level=0.9)

        # Higher attention should lengthen perceived duration
        assert high_attention.duration_estimate > low_attention.duration_estimate

    def test_dopamine_affects_clock_speed(self):
        """Test that dopamine modulates clock speed"""
        bg = BasalGanglia(dopamine_level=1.0)
        baseline_speed = bg.clock_speed

        bg.set_dopamine_level(1.5)
        assert bg.clock_speed > baseline_speed

        bg.set_dopamine_level(0.5)
        assert bg.clock_speed < baseline_speed

    def test_reference_comparison(self):
        """Test comparing to learned reference"""
        bg = BasalGanglia()
        bg.learn_reference("one_second", 1.0)

        comparison = bg.compare_to_reference(1.5, "one_second")
        assert comparison > 1.0  # Longer than reference

    def test_temporal_working_memory(self):
        """Test temporal working memory"""
        bg = BasalGanglia(memory_capacity=5)

        for i in range(7):
            bg.time_interval(float(i + 1))

        memory = bg.get_temporal_memory()
        assert len(memory) <= 5  # Capacity limited


class TestCerebellum:
    """Tests for cerebellar precise timing"""

    def test_initialization(self):
        """Test cerebellum initialization"""
        cerebellum = Cerebellum(precision=0.95)

        assert cerebellum.precision == 0.95

    def test_precise_timing(self):
        """Test sub-second precise timing"""
        cerebellum = Cerebellum()

        signal = cerebellum.time_precise_interval(duration=0.5)

        assert signal.source == "cerebellum"
        assert signal.scale == TimingScale.MILLISECONDS
        # High precision for short intervals
        assert signal.confidence > 0.9

    def test_precision_degrades_with_duration(self):
        """Test that precision degrades for longer durations"""
        cerebellum = Cerebellum(max_duration=1.0)

        short = cerebellum.time_precise_interval(0.5)
        long = cerebellum.time_precise_interval(3.0)

        assert short.confidence > long.confidence

    def test_rhythm_timing(self):
        """Test rhythmic timing"""
        cerebellum = Cerebellum()

        intervals = cerebellum.time_rhythm(beat_interval=0.5, n_beats=10)

        assert len(intervals) == 10
        # All intervals should be close to target
        for interval in intervals:
            assert 0.3 < interval < 0.7


class TestTimeCircuit:
    """Tests for integrated time circuit"""

    def test_initialization(self):
        """Test circuit initialization"""
        circuit = TimeCircuit()

        assert circuit.insula is not None
        assert circuit.sma is not None
        assert circuit.basal_ganglia is not None
        assert circuit.cerebellum is not None

    def test_duration_estimation(self):
        """Test integrated duration estimation"""
        circuit = TimeCircuit()

        signal = circuit.estimate_duration(
            actual_duration=5.0,
            arousal=0.5,
            attention=0.5,
            movement=False
        )

        assert signal.source == "integrated"
        assert signal.duration_estimate > 0
        assert signal.confidence > 0

    def test_movement_involves_sma(self):
        """Test that movement engages SMA"""
        circuit = TimeCircuit()

        no_movement = circuit.estimate_duration(5.0, movement=False)
        circuit.reset()
        with_movement = circuit.estimate_duration(5.0, movement=True)

        # Both should produce valid estimates
        assert no_movement.duration_estimate > 0
        assert with_movement.duration_estimate > 0

    def test_short_duration_uses_cerebellum(self):
        """Test that short durations engage cerebellum"""
        circuit = TimeCircuit()

        signal = circuit.estimate_duration(0.5)

        assert signal.scale == TimingScale.MILLISECONDS

    def test_statistics(self):
        """Test statistics collection"""
        circuit = TimeCircuit()

        circuit.estimate_duration(5.0)
        circuit.estimate_duration(10.0)

        stats = circuit.get_statistics()

        assert stats['current_time'] == 15.0
        assert stats['insula_signals'] >= 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
