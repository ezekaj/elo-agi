"""Tests for emotion circuit components."""

import numpy as np
from neuro.modules.m07_emotions_decisions.emotion_circuit import (
    VMPFC,
    Amygdala,
    ACC,
    Insula,
    EmotionCircuit,
    EmotionType,
)


class TestAmygdala:
    """Test amygdala threat/reward detection."""

    def test_threat_detection(self):
        amygdala = Amygdala()
        # High variance, extreme values = threat
        threatening = np.array([0.9, 0.1, 0.8, 0.2])
        threat_level = amygdala.detect_threat(threatening)
        assert threat_level > 0.3

    def test_safe_stimulus(self):
        amygdala = Amygdala()
        safe = np.array([0.5, 0.5, 0.5, 0.5])
        threat_level = amygdala.detect_threat(safe)
        assert threat_level < 0.5

    def test_fear_conditioning(self):
        amygdala = Amygdala()
        stimulus = np.array([0.3, 0.3, 0.3])

        # Before conditioning
        before = amygdala.detect_threat(stimulus)

        # Condition with threat
        amygdala.fear_conditioning(stimulus, 0.9)

        # After conditioning
        after = amygdala.detect_threat(stimulus)
        assert after > before

    def test_emotional_memory_storage(self):
        amygdala = Amygdala()
        content = np.array([0.5, 0.5])
        amygdala.store_emotional_memory(content, valence=-0.8, arousal=0.9)

        assert len(amygdala.emotional_memories) == 1
        assert amygdala.emotional_memories[0].valence == -0.8


class TestVMPFC:
    """Test VMPFC value computation."""

    def test_value_computation(self):
        vmpfc = VMPFC()
        stimulus = np.array([0.7, 0.7, 0.7])
        value = vmpfc.compute_value(stimulus)
        assert isinstance(value, float)

    def test_lesion_effect(self):
        vmpfc = VMPFC()
        stimulus = np.array([0.5, 0.5])

        before = vmpfc.compute_value(stimulus)
        vmpfc.lesion()
        after = vmpfc.compute_value(stimulus)

        assert after == 0.0  # No value without VMPFC
        vmpfc.restore()
        restored = vmpfc.compute_value(stimulus)
        assert restored == before

    def test_gut_feeling(self):
        vmpfc = VMPFC()
        context = {"familiarity": 0.8, "past_outcomes": [0.5, 0.7, 0.6]}
        valence, confidence = vmpfc.generate_gut_feeling(context)
        assert valence > 0  # Positive past outcomes
        assert confidence > 0


class TestACC:
    """Test ACC conflict monitoring."""

    def test_conflict_detection(self):
        acc = ACC()
        # Two similarly strong responses = conflict
        responses = [("option_a", 0.7), ("option_b", 0.65)]
        conflict = acc.conflict_detection(responses)
        assert conflict > 0.5

    def test_no_conflict(self):
        acc = ACC()
        # One clear winner = no conflict
        responses = [("winner", 0.9), ("loser", 0.2)]
        conflict = acc.conflict_detection(responses)
        assert conflict < 0.3

    def test_outcome_monitoring(self):
        acc = ACC()
        acc.monitor_outcomes("action_a", 0.5)
        acc.monitor_outcomes("action_a", 0.7)

        expected = acc.get_expected_value("action_a")
        assert 0.4 < expected < 0.8


class TestInsula:
    """Test insula body state mapping."""

    def test_body_to_emotion_mapping(self):
        insula = Insula()
        insula.update_body_state(heart_rate=0.9, muscle_tension=0.8)
        emotion = insula.map_to_emotion()
        assert emotion.emotion_type == EmotionType.FEAR

    def test_calm_state(self):
        insula = Insula()
        insula.update_body_state(heart_rate=0.3, muscle_tension=0.2)
        emotion = insula.map_to_emotion()
        assert emotion.arousal < 0.5


class TestEmotionCircuit:
    """Test integrated emotion circuit."""

    def test_full_processing(self):
        circuit = EmotionCircuit()
        stimulus = np.array([0.7, 0.3, 0.8, 0.2])
        result = circuit.process(stimulus)

        assert hasattr(result, "valence")
        assert hasattr(result, "arousal")
        assert hasattr(result, "emotion_type")

    def test_vmpfc_lesion_effect(self):
        circuit = EmotionCircuit()
        stimulus = np.array([0.5, 0.5, 0.5])

        circuit.process(stimulus)
        circuit.lesion_vmpfc()
        lesioned = circuit.process(stimulus)

        # Should still process but with different characteristics
        assert lesioned is not None
