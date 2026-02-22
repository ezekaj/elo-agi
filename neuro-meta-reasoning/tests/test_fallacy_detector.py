"""Tests for fallacy detector."""

import pytest
import numpy as np

from neuro.modules.meta_reasoning.fallacy_detector import (
    FallacyDetector,
    FallacyDetectorConfig,
    FallacyDetection,
    FallacyType,
    ReasoningStep,
)


class TestFallacyDetectorConfig:
    """Tests for FallacyDetectorConfig class."""

    def test_default_config(self):
        config = FallacyDetectorConfig()
        assert config.confirmation_bias_threshold == 0.7
        assert config.circular_depth_limit == 5
        assert config.enable_all_detectors

    def test_custom_config(self):
        config = FallacyDetectorConfig(
            confirmation_bias_threshold=0.8,
            min_evidence_count=5,
        )
        assert config.confirmation_bias_threshold == 0.8
        assert config.min_evidence_count == 5


class TestFallacyDetector:
    """Tests for FallacyDetector class."""

    def test_creation(self):
        detector = FallacyDetector(random_seed=42)
        assert detector is not None

    def test_detect_fallacies_empty_trace(self):
        detector = FallacyDetector(random_seed=42)

        fallacies = detector.detect_fallacies([])

        assert fallacies == []

    def test_detect_circular_reasoning(self):
        detector = FallacyDetector(random_seed=42)

        step = ReasoningStep(
            step_id="step1",
            content="A implies A",
            premises=["A"],
            conclusion="A",
            evidence_used=[],
            confidence=0.8,
        )

        fallacies = detector.detect_fallacies([step])

        assert len(fallacies) == 1
        assert fallacies[0].fallacy_type == FallacyType.CIRCULAR

    def test_detect_hasty_generalization(self):
        config = FallacyDetectorConfig(min_evidence_count=3)
        detector = FallacyDetector(config=config, random_seed=42)

        step = ReasoningStep(
            step_id="step1",
            content="All swans are white",
            premises=["I saw a white swan"],
            conclusion="All swans are always white",
            evidence_used=["swan1"],
            confidence=0.9,
        )

        fallacies = detector.detect_fallacies([step])

        hasty = [f for f in fallacies if f.fallacy_type == FallacyType.HASTY_GENERALIZATION]
        assert len(hasty) == 1

    def test_detect_confirmation_bias(self):
        detector = FallacyDetector(random_seed=42)

        hypotheses = [{"id": "h1", "content": "Theory A"}]
        evidence = [
            {"id": "e1", "supports": "h1"},
            {"id": "e2", "supports": "h1"},
            {"id": "e3", "supports": "h1"},
            {"id": "e4", "supports": "h1"},
            {"id": "e5", "contradicts": "h1"},
        ]

        fallacy = detector.detect_confirmation_bias(hypotheses, evidence)

        assert fallacy is not None
        assert fallacy.fallacy_type == FallacyType.CONFIRMATION_BIAS

    def test_no_confirmation_bias(self):
        detector = FallacyDetector(random_seed=42)

        hypotheses = [{"id": "h1", "content": "Theory A"}]
        evidence = [
            {"id": "e1", "supports": "h1"},
            {"id": "e2", "supports": "h1"},
            {"id": "e3", "contradicts": "h1"},
            {"id": "e4", "contradicts": "h1"},
        ]

        fallacy = detector.detect_confirmation_bias(hypotheses, evidence)

        assert fallacy is None

    def test_detect_circular_chain(self):
        detector = FallacyDetector(random_seed=42)

        chain = [
            ("A", "B"),
            ("B", "C"),
            ("C", "A"),
        ]

        fallacy = detector.detect_circular_reasoning(chain)

        assert fallacy is not None
        assert fallacy.fallacy_type == FallacyType.CIRCULAR

    def test_no_circular_chain(self):
        detector = FallacyDetector(random_seed=42)

        chain = [
            ("A", "B"),
            ("B", "C"),
            ("C", "D"),
        ]

        fallacy = detector.detect_circular_reasoning(chain)

        assert fallacy is None

    def test_detect_anchoring(self):
        detector = FallacyDetector(random_seed=42)

        fallacy = detector.detect_anchoring(
            initial_estimate=100.0,
            final_estimate=105.0,
            evidence_range=(150.0, 200.0),
        )

        assert fallacy is not None
        assert fallacy.fallacy_type == FallacyType.ANCHORING

    def test_no_anchoring(self):
        detector = FallacyDetector(random_seed=42)

        fallacy = detector.detect_anchoring(
            initial_estimate=100.0,
            final_estimate=175.0,
            evidence_range=(150.0, 200.0),
        )

        assert fallacy is None

    def test_detect_base_rate_neglect(self):
        detector = FallacyDetector(random_seed=42)

        fallacy = detector.detect_base_rate_neglect(
            base_rate=0.01,
            likelihood_ratio=10.0,
            posterior=0.9,
        )

        assert fallacy is not None
        assert fallacy.fallacy_type == FallacyType.BASE_RATE_NEGLECT

    def test_suggest_corrections(self):
        detector = FallacyDetector(random_seed=42)

        fallacies = [
            FallacyDetection(
                fallacy_type=FallacyType.CONFIRMATION_BIAS,
                confidence=0.8,
                location="test",
                evidence=["test"],
                severity=0.7,
            ),
            FallacyDetection(
                fallacy_type=FallacyType.CIRCULAR,
                confidence=0.9,
                location="test",
                evidence=["test"],
                severity=0.8,
            ),
        ]

        corrections = detector.suggest_corrections(fallacies)

        assert len(corrections) >= 2
        assert any("confirmation" in c.lower() for c in corrections)

    def test_get_detection_history(self):
        detector = FallacyDetector(random_seed=42)

        step = ReasoningStep(
            step_id="step1",
            content="A implies A",
            premises=["A"],
            conclusion="A",
            evidence_used=[],
            confidence=0.8,
        )
        detector.detect_fallacies([step])

        history = detector.get_detection_history()
        assert len(history) >= 1

    def test_get_history_by_type(self):
        detector = FallacyDetector(random_seed=42)

        step = ReasoningStep(
            step_id="step1",
            content="A",
            premises=["A"],
            conclusion="A",
            evidence_used=[],
            confidence=0.8,
        )
        detector.detect_fallacies([step])

        circular = detector.get_detection_history(fallacy_type=FallacyType.CIRCULAR)
        assert all(f.fallacy_type == FallacyType.CIRCULAR for f in circular)

    def test_statistics(self):
        detector = FallacyDetector(random_seed=42)

        step = ReasoningStep(
            step_id="step1",
            content="A",
            premises=["A"],
            conclusion="A",
            evidence_used=[],
            confidence=0.8,
        )
        detector.detect_fallacies([step])

        stats = detector.statistics()

        assert "total_detections" in stats
        assert "type_distribution" in stats


class TestFallacyType:
    """Tests for FallacyType enum."""

    def test_types(self):
        assert FallacyType.CONFIRMATION_BIAS.value == "confirmation_bias"
        assert FallacyType.ANCHORING.value == "anchoring"
        assert FallacyType.CIRCULAR.value == "circular"
        assert FallacyType.BASE_RATE_NEGLECT.value == "base_rate_neglect"


class TestReasoningStep:
    """Tests for ReasoningStep dataclass."""

    def test_step_creation(self):
        step = ReasoningStep(
            step_id="step1",
            content="If A then B; A; therefore B",
            premises=["A implies B", "A"],
            conclusion="B",
            evidence_used=["axiom1"],
            confidence=0.9,
        )

        assert step.step_id == "step1"
        assert step.conclusion == "B"
        assert len(step.premises) == 2
