"""Tests for dual emotion routes."""

import numpy as np
from neuro.modules.m07_emotions_decisions.dual_emotion_routes import (
    FastEmotionRoute,
    SlowEmotionRoute,
    DualRouteProcessor,
    ResponseType,
)


class TestFastRoute:
    """Test fast emotion pathway (12ms)."""

    def test_latency(self):
        fast = FastEmotionRoute()
        assert fast.LATENCY_MS == 12.0

    def test_threat_detection(self):
        fast = FastEmotionRoute()
        threatening = np.array([0.9, 0.1, 0.9, 0.1])  # High contrast
        response = fast.process(threatening)

        assert response.latency_ms == 12.0
        assert response.response_type == ResponseType.THREAT
        assert response.intensity > 0.5

    def test_safe_stimulus(self):
        fast = FastEmotionRoute()
        safe = np.array([0.5, 0.5, 0.5, 0.5])
        response = fast.process(safe)

        assert response.response_type in [ResponseType.NEUTRAL, ResponseType.REWARD]

    def test_learned_threat(self):
        fast = FastEmotionRoute()
        stimulus = np.array([0.3, 0.3, 0.3])

        before = fast.process(stimulus)
        fast.condition_threat(stimulus, 0.9)
        after = fast.process(stimulus)

        # Should detect threat after conditioning
        assert after.intensity > before.intensity


class TestSlowRoute:
    """Test slow emotion pathway (100ms)."""

    def test_latency(self):
        slow = SlowEmotionRoute()
        assert slow.LATENCY_MS == 100.0

    def test_higher_confidence(self):
        slow = SlowEmotionRoute()
        stimulus = np.array([0.5, 0.5, 0.5, 0.5])
        response = slow.process(stimulus)

        assert response.confidence > 0.7  # Slow route is more confident

    def test_safety_override(self):
        slow = SlowEmotionRoute()
        stick = np.array([0.8, 0.1, 0.7, 0.2])  # Looks like snake

        # Teach that it's a stick
        slow.learn_safety_override(stick, "It's just a stick")

        response = slow.process(stick)
        assert response.response_type == ResponseType.OVERRIDE

    def test_context_use(self):
        slow = SlowEmotionRoute()
        stimulus = np.array([0.6, 0.6, 0.6])

        # Same stimulus, different context
        safe_context = {"safe_environment": True}
        danger_context = {"known_danger": True}

        safe_response = slow.process(stimulus, safe_context)
        danger_response = slow.process(stimulus, danger_context)

        assert (
            danger_response.details["threat_assessment"]
            > safe_response.details["threat_assessment"]
        )


class TestDualRouteProcessor:
    """Test integrated dual route processing."""

    def test_both_routes_fire(self):
        processor = DualRouteProcessor()
        stimulus = np.array([0.5, 0.5, 0.5])

        fast, slow = processor.process(stimulus)

        assert fast is not None
        assert slow is not None
        assert fast.latency_ms < slow.latency_ms

    def test_fast_only_under_stress(self):
        processor = DualRouteProcessor(stress_level=0.95)
        stimulus = np.array([0.5, 0.5, 0.5])

        fast, slow = processor.process(stimulus)

        assert fast is not None
        assert slow is None  # No slow processing under extreme stress

    def test_reconciliation_agreement(self):
        processor = DualRouteProcessor()
        safe = np.array([0.4, 0.4, 0.4, 0.4])

        final = processor.get_final_response(safe)

        assert final.details.get("reconciled", False)

    def test_safety_learning(self):
        processor = DualRouteProcessor()
        stimulus = np.array([0.8, 0.2, 0.7, 0.3])

        # Before learning
        before = processor.get_final_response(stimulus)

        # Learn it's safe
        processor.learn_safety(stimulus, "Harmless pattern")

        # After learning
        after = processor.get_final_response(stimulus)

        # Should be overridden
        assert "override" in str(after.details).lower() or after.intensity < before.intensity


class TestTimingDifference:
    """Test the critical timing difference between routes."""

    def test_fast_triggers_first(self):
        processor = DualRouteProcessor()
        threat = np.array([0.9, 0.1, 0.9, 0.1])

        fast, slow = processor.process(threat)

        # Fast route should have threat response with lower latency
        assert fast.latency_ms == 12.0
        assert slow.latency_ms == 100.0

        # In real system, fast would trigger before slow completes
        # This represents the ~88ms where only fast response exists
