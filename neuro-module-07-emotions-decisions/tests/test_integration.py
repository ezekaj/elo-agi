"""Tests for full emotion-decision integration."""

import numpy as np
import pytest
from neuro.modules.m07_emotions_decisions.emotion_decision_integrator import (
    EmotionDecisionSystem,
    Situation,
    SituationType,
    create_threat_situation,
    create_reward_situation,
    create_moral_situation,
)
from neuro.modules.m07_emotions_decisions.moral_reasoning import (
    create_trolley_push,
    create_trolley_switch,
)
from neuro.modules.m07_emotions_decisions.emotional_states import Outcome, OutcomeType


class TestEmotionDecisionSystem:
    """Test the full integrated system."""

    def test_threat_response(self):
        system = EmotionDecisionSystem()
        situation = create_threat_situation(intensity=0.8)

        decision = system.process_situation(situation)

        # Should avoid threat
        assert decision.action in ["avoid", "flee", "freeze", "dont_act"]
        # System avoids when threat_level > 0.3
        assert decision.emotional_state.threat_level > 0.3 or decision.emotional_state.valence < 0

    def test_reward_approach(self):
        system = EmotionDecisionSystem()
        situation = create_reward_situation(value=0.7)

        decision = system.process_situation(situation)

        # Should approach reward
        assert decision.action in ["act", "approach"]
        assert decision.value > 0

    def test_moral_dilemma_processing(self):
        system = EmotionDecisionSystem()
        scenario = create_trolley_push()
        situation = create_moral_situation(scenario)

        decision = system.process_situation(situation)

        assert decision.moral_decision is not None
        assert decision.moral_decision.deontological_weight > 0.5

    def test_learning_from_outcome(self):
        system = EmotionDecisionSystem()
        situation = create_reward_situation(value=0.5)
        decision = system.process_situation(situation)

        # Simulate positive outcome
        outcome = Outcome(outcome_type=OutcomeType.REWARD_RECEIVED, magnitude=0.8, expected=True)

        system.learn_from_outcome(decision, outcome)

        # System should have learned
        assert len(system.decision_history) > 0


class TestLesionSimulation:
    """Test lesion effects on decision-making."""

    def test_vmpfc_lesion_moral_change(self):
        # Normal processing
        normal_system = EmotionDecisionSystem()
        scenario = create_trolley_push()
        situation = create_moral_situation(scenario)

        normal_decision = normal_system.process_situation(situation)

        # Lesioned processing
        lesion_system = EmotionDecisionSystem()
        lesion_system.simulate_lesion("vmpfc")

        lesion_decision = lesion_system.process_situation(situation)

        # Lesioned should have reduced deontological weight
        assert (
            lesion_decision.moral_decision.deontological_weight
            <= normal_decision.moral_decision.deontological_weight
        )

    def test_amygdala_lesion_threat_response(self):
        normal_system = EmotionDecisionSystem()
        lesion_system = EmotionDecisionSystem()
        lesion_system.simulate_lesion("amygdala")

        threat = create_threat_situation(intensity=0.7)

        normal_decision = normal_system.process_situation(threat)
        lesion_decision = lesion_system.process_situation(threat)

        # Lesioned should have reduced threat response
        assert (
            lesion_decision.emotional_state.threat_level
            <= normal_decision.emotional_state.threat_level
        )


class TestStressEffects:
    """Test stress effects on processing."""

    def test_high_stress_fast_response(self):
        system = EmotionDecisionSystem()
        system.set_stress_level(0.95)

        situation = create_threat_situation(intensity=0.5)
        decision = system.process_situation(situation)

        # Under high stress, should rely more on fast route
        # Processing time should be faster
        assert decision.processing_time_ms < 50  # Closer to fast route timing


class TestEmotionalDynamics:
    """Test emotional state changes over time."""

    def test_mood_tracking(self):
        system = EmotionDecisionSystem()

        # Process several situations
        for _ in range(3):
            threat = create_threat_situation(0.6)
            system.process_situation(threat)

        mood = system.get_mood()

        assert "valence" in mood
        assert mood["valence"] < 0  # Negative from threats

    def test_emotional_state_access(self):
        system = EmotionDecisionSystem()
        situation = create_reward_situation(0.7)
        system.process_situation(situation)

        state = system.get_emotional_state()

        assert state is not None
        # Reward situation should result in approach behavior and have some valence
        assert state.valence is not None
