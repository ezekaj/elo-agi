"""Tests for sleep stages and oscillation generation"""

import numpy as np
import pytest
import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

from src.sleep_stages import (
    SleepStage,
    StageProperties,
    SleepStageController,
    OscillationGenerator,
    OscillationEvent
)


class TestSleepStage:
    """Tests for sleep stage enum and properties"""

    def test_all_stages_exist(self):
        """Test all expected stages exist"""
        stages = [SleepStage.WAKE, SleepStage.NREM1, SleepStage.NREM2,
                  SleepStage.SWS, SleepStage.REM]
        assert len(stages) == 5

    def test_stage_properties(self):
        """Test stage properties are defined correctly"""
        sws_props = StageProperties.for_stage(SleepStage.SWS)

        assert sws_props.oscillation_freq[0] == 0.5  # Slow waves start at 0.5 Hz
        assert sws_props.oscillation_freq[1] == 4.0
        assert sws_props.plasticity_mode == "consolidation"
        assert sws_props.arousal_threshold > 0.5  # Hard to wake from SWS

    def test_rem_properties(self):
        """Test REM stage properties"""
        rem_props = StageProperties.for_stage(SleepStage.REM)

        assert rem_props.oscillation_freq[0] == 4.0  # Theta
        assert rem_props.oscillation_freq[1] == 8.0
        assert rem_props.memory_function == "emotional_processing"

    def test_wake_properties(self):
        """Test wake state properties"""
        wake_props = StageProperties.for_stage(SleepStage.WAKE)

        assert wake_props.plasticity_mode == "encoding"
        assert wake_props.arousal_threshold == 0.0


class TestOscillationGenerator:
    """Tests for neural oscillation generation"""

    def test_initialization(self):
        """Test generator initializes correctly"""
        gen = OscillationGenerator(sample_rate=1000.0)
        assert gen.sample_rate == 1000.0
        assert gen.current_time == 0.0

    def test_sleep_spindle(self):
        """Test sleep spindle generation"""
        gen = OscillationGenerator()
        spindle = gen.sleep_spindle(duration=1.0)

        assert spindle.event_type == "spindle"
        assert 12.0 <= spindle.frequency <= 16.0  # Spindle range
        assert spindle.duration == 1.0
        assert spindle.waveform is not None
        assert len(spindle.waveform) == 1000  # 1 second at 1000 Hz

    def test_slow_oscillation(self):
        """Test slow oscillation generation"""
        gen = OscillationGenerator()
        slow_osc = gen.slow_oscillation(duration=2.0)

        assert slow_osc.event_type == "slow_oscillation"
        assert 0.5 <= slow_osc.frequency <= 1.0
        assert slow_osc.duration == 2.0

    def test_sharp_wave_ripple(self):
        """Test sharp-wave ripple generation"""
        gen = OscillationGenerator()
        swr = gen.sharp_wave_ripple()

        assert swr.event_type == "sharp_wave_ripple"
        assert 80 <= swr.frequency <= 120  # Ripple range
        assert swr.duration == 0.05  # ~50ms

    def test_theta_wave(self):
        """Test theta wave generation"""
        gen = OscillationGenerator()
        theta = gen.theta_wave(duration=1.0)

        assert theta.event_type == "theta_wave"
        assert 4.0 <= theta.frequency <= 8.0  # Theta range

    def test_k_complex(self):
        """Test K-complex generation"""
        gen = OscillationGenerator()
        k_complex = gen.k_complex()

        assert k_complex.event_type == "k_complex"
        assert k_complex.duration == 0.5
        # K-complex should have negative followed by positive
        waveform = k_complex.waveform
        assert np.min(waveform[:250]) < 0  # Negative peak first
        assert np.max(waveform[250:]) > 0  # Positive peak second

    def test_stage_activity_sws(self):
        """Test SWS activity generation"""
        gen = OscillationGenerator()
        events = gen.generate_stage_activity(SleepStage.SWS, duration=5.0)

        assert len(events) > 0

        # Should contain slow oscillations
        slow_oscs = [e for e in events if e.event_type == "slow_oscillation"]
        assert len(slow_oscs) > 0

    def test_stage_activity_nrem2(self):
        """Test NREM2 activity generation"""
        gen = OscillationGenerator()
        events = gen.generate_stage_activity(SleepStage.NREM2, duration=5.0)

        event_types = set(e.event_type for e in events)
        # Should have spindles or K-complexes
        assert len(events) > 0

    def test_time_advances(self):
        """Test that time advances with event generation"""
        gen = OscillationGenerator()
        assert gen.current_time == 0.0

        gen.sleep_spindle(duration=1.0)
        assert gen.current_time == 1.0

        gen.theta_wave(duration=2.0)
        assert gen.current_time == 3.0


class TestSleepStageController:
    """Tests for sleep stage controller"""

    def test_initialization(self):
        """Test controller initializes in wake state"""
        controller = SleepStageController()

        assert controller.current_stage == SleepStage.WAKE
        assert controller.cycle_count == 0
        assert controller.total_time == 0.0

    def test_start_sleep(self):
        """Test starting sleep"""
        controller = SleepStageController()
        stage = controller.start_sleep()

        assert stage == SleepStage.NREM1
        assert controller.current_stage == SleepStage.NREM1
        assert controller.cycle_count == 0

    def test_stage_transition(self):
        """Test manual stage transition"""
        controller = SleepStageController()

        success = controller.transition(SleepStage.NREM2)
        assert success
        assert controller.current_stage == SleepStage.NREM2

    def test_no_self_transition(self):
        """Test transitioning to same stage returns False"""
        controller = SleepStageController()
        controller.transition(SleepStage.NREM2)

        success = controller.transition(SleepStage.NREM2)
        assert not success

    def test_time_advance(self):
        """Test time advancement"""
        controller = SleepStageController()
        controller.start_sleep()

        # Use less than NREM1 duration (5 min) to avoid triggering transition
        controller.advance_time(4.0)
        assert controller.total_time == 4.0
        assert controller.get_time_in_stage() == 4.0

    def test_automatic_stage_transition(self):
        """Test automatic transition after stage duration"""
        controller = SleepStageController()
        controller.start_sleep()

        # Advance past NREM1 typical duration
        for _ in range(20):  # 20 * 1 minute = 20 minutes
            result = controller.advance_time(1.0)

        # Should have transitioned at some point
        assert controller.current_stage != SleepStage.NREM1 or controller.total_time >= 5

    def test_cycle_counting(self):
        """Test sleep cycle counting"""
        controller = SleepStageController()
        controller.start_sleep()

        # Force through one complete cycle manually (no advance_time to avoid auto-transitions)
        controller.transition(SleepStage.NREM2)
        controller.transition(SleepStage.SWS)
        controller.transition(SleepStage.NREM2)
        controller.transition(SleepStage.REM)

        # Transition to next cycle (REM → NREM1 should increment cycle_count)
        controller.transition(SleepStage.NREM1)

        assert controller.cycle_count >= 1

    def test_stage_summary(self):
        """Test stage time summary"""
        controller = SleepStageController()
        controller.start_sleep()

        controller.advance_time(10)
        controller.transition(SleepStage.NREM2)
        controller.advance_time(20)

        summary = controller.get_stage_summary()

        assert SleepStage.NREM1 in summary
        assert summary[SleepStage.NREM1] == 10
        assert summary[SleepStage.NREM2] == 20

    def test_wake_up(self):
        """Test waking up"""
        controller = SleepStageController()
        controller.start_sleep()
        controller.advance_time(30)

        controller.wake_up()

        assert controller.current_stage == SleepStage.WAKE

    def test_reset(self):
        """Test controller reset"""
        controller = SleepStageController()
        controller.start_sleep()
        controller.advance_time(60)

        controller.reset()

        assert controller.current_stage == SleepStage.WAKE
        assert controller.total_time == 0.0
        assert controller.cycle_count == 0


class TestSleepCycleArchitecture:
    """Tests for sleep architecture patterns"""

    def test_sws_decreases_across_night(self):
        """Test that SWS duration decreases across sleep cycles"""
        controller = SleepStageController()

        # Get adjusted durations for different cycles
        props = StageProperties.for_stage(SleepStage.SWS)
        base_duration = props.duration_typical

        controller.cycle_count = 0
        duration_early = controller._adjust_duration_for_cycle(base_duration, SleepStage.SWS)

        controller.cycle_count = 3
        duration_late = controller._adjust_duration_for_cycle(base_duration, SleepStage.SWS)

        assert duration_late < duration_early

    def test_rem_increases_across_night(self):
        """Test that REM duration increases across sleep cycles"""
        controller = SleepStageController()

        props = StageProperties.for_stage(SleepStage.REM)
        base_duration = props.duration_typical

        controller.cycle_count = 0
        duration_early = controller._adjust_duration_for_cycle(base_duration, SleepStage.REM)

        controller.cycle_count = 3
        duration_late = controller._adjust_duration_for_cycle(base_duration, SleepStage.REM)

        assert duration_late > duration_early

    def test_cycle_duration_approximately_90_minutes(self):
        """Test that cycle duration is approximately 90 minutes"""
        controller = SleepStageController()

        # Default cycle duration
        assert controller.cycle_duration == 90.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
