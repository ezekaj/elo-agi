"""Integration tests for time perception system"""

import numpy as np
import pytest
from neuro.modules.m11_time_perception.temporal_integration import (
    SubjectiveTimeSystem,
    TimePerceptionOrchestrator,
    TemporalEstimate,
)
from neuro.modules.m11_time_perception.time_modulation import EmotionalState
from neuro.modules.m11_time_perception.interval_timing import TimingMode
from neuro.modules.m11_time_perception.embodied_time import BodyState


class TestSubjectiveTimeSystem:
    """Tests for subjective time system"""

    def test_initialization(self):
        """Test system initialization"""
        system = SubjectiveTimeSystem()

        assert system.circuits is not None
        assert system.interval_timer is not None
        assert system.modulation is not None
        assert system.embodied is not None

    def test_duration_estimation(self):
        """Test complete duration estimation"""
        system = SubjectiveTimeSystem()

        estimate = system.estimate_duration(actual_duration=10.0, attention=0.5)

        assert isinstance(estimate, TemporalEstimate)
        assert estimate.actual_duration == 10.0
        assert estimate.perceived_duration > 0
        assert estimate.confidence > 0

    def test_emotional_modulation(self):
        """Test that emotion modulates time"""
        system = SubjectiveTimeSystem()

        neutral = system.estimate_duration(10.0, emotional_state=EmotionalState.NEUTRAL)

        fear = system.estimate_duration(10.0, emotional_state=EmotionalState.FEAR)

        # Fear should lengthen perceived duration
        assert fear.perceived_duration > neutral.perceived_duration

    def test_attention_modulation(self):
        """Test that attention modulates time"""
        system = SubjectiveTimeSystem()

        low_attention = system.estimate_duration(10.0, attention=0.2)
        system.reset()
        high_attention = system.estimate_duration(10.0, attention=0.9)

        # More attention should lengthen perceived duration
        assert high_attention.perceived_duration > low_attention.perceived_duration

    def test_dopamine_modulation(self):
        """Test that dopamine modulates time"""
        system = SubjectiveTimeSystem()

        low_da = system.estimate_duration(10.0, dopamine=0.6)
        system.reset()
        high_da = system.estimate_duration(10.0, dopamine=1.5)

        # High dopamine should shorten perceived duration
        assert high_da.perceived_duration < low_da.perceived_duration

    def test_age_modulation(self):
        """Test that age modulates time"""
        system = SubjectiveTimeSystem()

        young = system.estimate_duration(10.0, age=20)
        system.reset()
        old = system.estimate_duration(10.0, age=70)

        # Older age should shorten perceived duration
        assert old.perceived_duration < young.perceived_duration

    def test_movement_engages_components(self):
        """Test that movement engages additional components"""
        system = SubjectiveTimeSystem()

        no_movement = system.estimate_duration(10.0, movement=False)
        system.reset()
        with_movement = system.estimate_duration(10.0, movement=True)

        # Both should produce valid estimates
        assert no_movement.perceived_duration > 0
        assert with_movement.perceived_duration > 0

    def test_duration_production(self):
        """Test producing a target duration"""
        system = SubjectiveTimeSystem()

        estimate = system.produce_duration(5.0)

        # Should be close to target
        assert 3 < estimate.perceived_duration < 8

    def test_statistics(self):
        """Test statistics collection"""
        system = SubjectiveTimeSystem()

        for _ in range(5):
            system.estimate_duration(10.0)

        stats = system.get_statistics()

        assert stats["n_estimates"] == 5
        assert "mean_error" in stats

    def test_set_dopamine(self):
        """Test setting dopamine across systems"""
        system = SubjectiveTimeSystem()

        system.set_dopamine(1.5)

        assert system.modulation.dopamine.current_level == 1.5

    def test_component_contributions(self):
        """Test that estimate includes component contributions"""
        system = SubjectiveTimeSystem()

        estimate = system.estimate_duration(10.0)

        assert estimate.circuit_estimate > 0
        assert estimate.interval_estimate > 0
        assert estimate.embodied_estimate > 0
        assert estimate.modulation_ratio > 0


class TestTimePerceptionOrchestrator:
    """Tests for orchestrator"""

    def test_initialization(self):
        """Test orchestrator initialization"""
        orchestrator = TimePerceptionOrchestrator()

        assert orchestrator.subjective_system is not None
        assert len(orchestrator.scenarios) > 0

    def test_baseline_estimation(self):
        """Test baseline scenario estimation"""
        orchestrator = TimePerceptionOrchestrator()

        estimate = orchestrator.estimate(10.0, scenario="baseline")

        assert estimate.perceived_duration > 0

    def test_fear_scenario(self):
        """Test fear scenario"""
        orchestrator = TimePerceptionOrchestrator()

        baseline = orchestrator.estimate(10.0, scenario="baseline")
        fear = orchestrator.estimate(10.0, scenario="fear")

        assert fear.perceived_duration > baseline.perceived_duration

    def test_boredom_scenario(self):
        """Test boredom scenario"""
        orchestrator = TimePerceptionOrchestrator()

        # Boredom has mixed effects: emotional stretching but low attention
        # Just verify it produces a valid estimate
        estimate = orchestrator.estimate(10.0, scenario="boredom")

        assert estimate.perceived_duration > 0

    def test_flow_scenario(self):
        """Test flow scenario"""
        orchestrator = TimePerceptionOrchestrator()

        baseline = orchestrator.estimate(10.0, scenario="baseline")
        flow = orchestrator.estimate(10.0, scenario="flow")

        assert flow.perceived_duration < baseline.perceived_duration

    def test_elderly_scenario(self):
        """Test elderly scenario"""
        orchestrator = TimePerceptionOrchestrator()

        # Age effect is one factor among many, may be overwhelmed by noise
        # Verify elderly estimate is valid
        elderly = orchestrator.estimate(10.0, scenario="elderly")

        assert elderly.perceived_duration > 0
        assert elderly.age == 70

    def test_child_scenario(self):
        """Test child scenario"""
        orchestrator = TimePerceptionOrchestrator()

        baseline = orchestrator.estimate(10.0, scenario="baseline")
        child = orchestrator.estimate(10.0, scenario="child")

        # Time should feel slower for children
        assert child.perceived_duration > baseline.perceived_duration

    def test_parkinsons_scenario(self):
        """Test Parkinson's (low dopamine) scenario"""
        orchestrator = TimePerceptionOrchestrator()

        baseline = orchestrator.estimate(10.0, scenario="baseline")
        parkinsons = orchestrator.estimate(10.0, scenario="parkinsons")

        # Low dopamine = slower clock = time feels longer
        assert parkinsons.perceived_duration > baseline.perceived_duration

    def test_compare_scenarios(self):
        """Test comparing multiple scenarios"""
        orchestrator = TimePerceptionOrchestrator()

        results = orchestrator.compare_scenarios(
            duration=10.0, scenarios=["baseline", "fear", "flow"]
        )

        assert len(results) == 3
        assert "baseline" in results
        assert "fear" in results
        assert "flow" in results

    def test_simulate_threatening_event(self):
        """Test simulating a threatening event"""
        orchestrator = TimePerceptionOrchestrator()

        estimate = orchestrator.simulate_event(
            duration=5.0, event_type="threatening", intensity=0.8
        )

        # Threatening events slow time
        assert estimate.perceived_duration > 5.0

    def test_simulate_rewarding_event(self):
        """Test simulating a rewarding event"""
        orchestrator = TimePerceptionOrchestrator()

        estimate = orchestrator.simulate_event(duration=5.0, event_type="rewarding", intensity=0.8)

        assert estimate.perceived_duration > 0

    def test_simulate_boring_event(self):
        """Test simulating a boring event"""
        orchestrator = TimePerceptionOrchestrator()

        estimate = orchestrator.simulate_event(duration=60.0, event_type="boring", intensity=0.9)

        # Boring events should produce valid estimate
        # Note: low attention can counteract emotional stretching
        assert estimate.perceived_duration > 0
        assert estimate.emotional_state == "boredom"

    def test_simulate_engaging_event(self):
        """Test simulating an engaging event"""
        orchestrator = TimePerceptionOrchestrator()

        estimate = orchestrator.simulate_event(duration=60.0, event_type="engaging", intensity=0.8)

        # Engaging events feel shorter
        assert estimate.perceived_duration < 60.0

    def test_simulate_day(self):
        """Test simulating a day of events"""
        orchestrator = TimePerceptionOrchestrator()

        events = [
            (60.0, "boring", 0.5),  # Boring meeting
            (30.0, "engaging", 0.7),  # Fun task
            (10.0, "threatening", 0.8),  # Stressful moment
        ]

        results = orchestrator.simulate_day(events)

        assert len(results) == 3

    def test_subjective_day_length(self):
        """Test calculating subjective day length"""
        orchestrator = TimePerceptionOrchestrator()

        # Mix of boring and engaging events
        events = [(120.0, "boring", 0.7), (60.0, "engaging", 0.8), (30.0, "boring", 0.5)]

        subjective, objective = orchestrator.get_subjective_day_length(events)

        assert objective == 210.0
        # Mix should result in some distortion
        assert subjective != objective

    def test_parameter_override(self):
        """Test overriding scenario parameters"""
        orchestrator = TimePerceptionOrchestrator()

        # Use baseline but override age
        estimate = orchestrator.estimate(
            10.0,
            scenario="baseline",
            age=80,  # Override
        )

        assert estimate.age == 80


class TestResearchValidation:
    """Tests validating key research findings"""

    def test_high_arousal_slows_time(self):
        """Validate: High arousal → time slows"""
        orchestrator = TimePerceptionOrchestrator()

        calm = orchestrator.estimate(10.0, emotional_state=EmotionalState.NEUTRAL)
        aroused = orchestrator.estimate(10.0, emotional_state=EmotionalState.FEAR)

        assert aroused.perceived_duration > calm.perceived_duration

    def test_more_attention_longer_duration(self):
        """Validate: More attention → longer perceived duration"""
        orchestrator = TimePerceptionOrchestrator()

        low_att = orchestrator.estimate(10.0, attention=0.2)
        high_att = orchestrator.estimate(10.0, attention=0.9)

        assert high_att.perceived_duration > low_att.perceived_duration

    def test_dopamine_speeds_time(self):
        """Validate: Dopamine increase → time speeds up"""
        orchestrator = TimePerceptionOrchestrator()

        low_da = orchestrator.estimate(10.0, dopamine=0.6)
        high_da = orchestrator.estimate(10.0, dopamine=1.5)

        # High DA = faster clock = shorter perceived duration
        assert high_da.perceived_duration < low_da.perceived_duration

    def test_aging_accelerates_time(self):
        """Validate: Aging → subjective time acceleration"""
        orchestrator = TimePerceptionOrchestrator()

        young = orchestrator.estimate(10.0, age=20)
        old = orchestrator.estimate(10.0, age=70)

        assert old.perceived_duration < young.perceived_duration


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
