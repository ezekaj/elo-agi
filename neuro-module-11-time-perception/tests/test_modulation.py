"""Tests for time perception modulation factors"""

import numpy as np
import pytest
import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

from src.time_modulation import (
    EmotionalModulator,
    AttentionalModulator,
    DopamineModulator,
    AgeModulator,
    TimeModulationSystem,
    EmotionalState,
    ModulationEffect
)


class TestEmotionalModulator:
    """Tests for emotional modulation of time"""

    def test_initialization(self):
        """Test modulator initialization"""
        modulator = EmotionalModulator()

        assert modulator.current_state == EmotionalState.NEUTRAL

    def test_fear_slows_time(self):
        """Test that fear makes time feel slower"""
        modulator = EmotionalModulator()

        neutral = modulator.modulate(10.0, EmotionalState.NEUTRAL)
        fear = modulator.modulate(10.0, EmotionalState.FEAR)

        assert fear.distortion_ratio > neutral.distortion_ratio

    def test_boredom_stretches_time(self):
        """Test that boredom makes time drag"""
        modulator = EmotionalModulator()

        neutral = modulator.modulate(10.0, EmotionalState.NEUTRAL)
        boredom = modulator.modulate(10.0, EmotionalState.BOREDOM)

        # Boredom also stretches time
        assert boredom.distortion_ratio > neutral.distortion_ratio

    def test_flow_speeds_time(self):
        """Test that flow state makes time fly"""
        modulator = EmotionalModulator()

        neutral = modulator.modulate(10.0, EmotionalState.NEUTRAL)
        flow = modulator.modulate(10.0, EmotionalState.FLOW)

        # Flow makes time feel shorter
        assert flow.distortion_ratio < neutral.distortion_ratio

    def test_arousal_level(self):
        """Test arousal level extraction"""
        modulator = EmotionalModulator()

        modulator.set_state(EmotionalState.FEAR)
        assert modulator.get_arousal() > 0.8

        modulator.set_state(EmotionalState.BOREDOM)
        assert modulator.get_arousal() < 0.3

    def test_event_processing(self):
        """Test processing emotional events"""
        modulator = EmotionalModulator()

        # Negative intense event should increase arousal
        new_arousal = modulator.process_event(
            event_valence=-0.8,
            event_intensity=0.9
        )

        assert new_arousal > 0.5


class TestAttentionalModulator:
    """Tests for attentional modulation of time"""

    def test_initialization(self):
        """Test modulator initialization"""
        modulator = AttentionalModulator()

        assert modulator.current_attention == modulator.base_attention

    def test_more_attention_longer_time(self):
        """Test that more attention to time = longer perceived duration"""
        modulator = AttentionalModulator()

        low_attention = modulator.modulate(10.0, attention_to_time=0.2)
        high_attention = modulator.modulate(10.0, attention_to_time=0.9)

        assert high_attention.distortion_ratio > low_attention.distortion_ratio

    def test_attention_allocation(self):
        """Test attention allocation with capacity limits"""
        modulator = AttentionalModulator()

        # Request more than capacity
        actual = modulator.allocate_attention(to_time=0.7, to_task=0.6)

        # Should be scaled down
        assert actual < 0.7
        assert modulator.attention_allocated_to_time == actual

    def test_dual_task_effect(self):
        """Test dual-task reduces time attention"""
        modulator = AttentionalModulator()

        # Easy task
        easy = modulator.dual_task_effect(task_difficulty=0.2)

        # Hard task
        hard = modulator.dual_task_effect(task_difficulty=0.9)

        assert hard < easy


class TestDopamineModulator:
    """Tests for dopamine modulation of time"""

    def test_initialization(self):
        """Test modulator initialization"""
        modulator = DopamineModulator(baseline_level=1.0)

        assert modulator.current_level == 1.0

    def test_high_dopamine_speeds_time(self):
        """Test that high dopamine makes time feel faster"""
        modulator = DopamineModulator()

        modulator.set_level(0.7)
        low_da = modulator.modulate(10.0)

        modulator.set_level(1.5)
        high_da = modulator.modulate(10.0)

        # High dopamine = faster clock = shorter perceived duration
        assert high_da.distortion_ratio < low_da.distortion_ratio

    def test_reward_increases_dopamine(self):
        """Test that reward boosts dopamine"""
        modulator = DopamineModulator()
        baseline = modulator.current_level

        modulator.simulate_reward(reward_magnitude=0.8)

        assert modulator.current_level > baseline

    def test_stimulant_effect(self):
        """Test stimulant drug effect"""
        modulator = DopamineModulator()

        modulator.simulate_stimulant(dose=0.8)

        assert modulator.current_level > 1.5

    def test_parkinsons_effect(self):
        """Test Parkinson's dopamine depletion"""
        modulator = DopamineModulator()

        modulator.simulate_parkinsons(severity=0.8)

        assert modulator.current_level < 0.5


class TestAgeModulator:
    """Tests for age modulation of time"""

    def test_initialization(self):
        """Test modulator initialization"""
        modulator = AgeModulator(current_age=30)

        assert modulator.current_age == 30

    def test_aging_speeds_subjective_time(self):
        """Test that aging makes time feel faster"""
        modulator = AgeModulator()

        young = modulator.modulate(10.0, age=20)
        old = modulator.modulate(10.0, age=70)

        # Older age = faster subjective time = shorter perceived
        assert old.distortion_ratio < young.distortion_ratio

    def test_children_slow_time(self):
        """Test that children experience slower time"""
        modulator = AgeModulator()

        child = modulator.modulate(10.0, age=8)
        adult = modulator.modulate(10.0, age=30)

        assert child.distortion_ratio > adult.distortion_ratio

    def test_subjective_year_length(self):
        """Test subjective year length calculation"""
        modulator = AgeModulator(current_age=40)

        # At 40, year feels half as long as at 20
        subjective = modulator.get_subjective_year_length(40)

        assert subjective < 1.0


class TestTimeModulationSystem:
    """Tests for integrated modulation system"""

    def test_initialization(self):
        """Test system initialization"""
        system = TimeModulationSystem()

        assert system.emotional is not None
        assert system.attentional is not None
        assert system.dopamine is not None
        assert system.age is not None

    def test_combined_modulation(self):
        """Test combined modulation effect"""
        system = TimeModulationSystem()

        perceived, effects = system.modulate_duration(
            actual_duration=10.0,
            emotional_state=EmotionalState.FEAR,
            attention=0.8,
            dopamine_level=1.2,
            age=30
        )

        assert perceived > 0
        assert 'emotion' in effects
        assert 'attention' in effects
        assert 'dopamine' in effects
        assert 'age' in effects

    def test_set_state(self):
        """Test setting system state"""
        system = TimeModulationSystem()

        system.set_state(
            emotional_state=EmotionalState.EXCITEMENT,
            attention=0.7,
            dopamine=1.3,
            age=25
        )

        state = system.get_current_state()

        assert state['emotional_state'] == 'excitement'
        assert state['attention'] == 0.7
        assert state['age'] == 25

    def test_fear_lengthens_perception(self):
        """Test that fear lengthens time perception"""
        system = TimeModulationSystem()

        neutral_perceived, _ = system.modulate_duration(
            10.0,
            emotional_state=EmotionalState.NEUTRAL
        )

        fear_perceived, _ = system.modulate_duration(
            10.0,
            emotional_state=EmotionalState.FEAR
        )

        assert fear_perceived > neutral_perceived


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
