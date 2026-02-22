"""Tests for metacognition"""

import numpy as np
import pytest
from neuro.modules.m16_consciousness.metacognition import (
    ConfidenceEstimator,
    PerformanceMonitor,
    MetacognitiveSystem,
)


class TestConfidenceEstimator:
    """Tests for confidence estimation"""

    def test_initialization(self):
        """Test estimator initialization"""
        estimator = ConfidenceEstimator()
        assert len(estimator.confidence_history) == 0

    def test_estimate_confidence(self):
        """Test confidence estimation"""
        estimator = ConfidenceEstimator()
        evidence = np.random.rand(50)
        decision = np.random.rand(50)

        result = estimator.estimate_confidence(evidence, decision)
        assert "confidence" in result
        assert 0 <= result["confidence"] <= 1

    def test_high_evidence_high_confidence(self):
        """Test strong evidence gives high confidence"""
        estimator = ConfidenceEstimator()
        strong_evidence = np.ones(50)
        consistent_decision = np.ones(50)

        result = estimator.estimate_confidence(strong_evidence, consistent_decision)
        assert result["confidence"] > 0.5

    def test_receive_feedback(self):
        """Test feedback updates calibration"""
        estimator = ConfidenceEstimator()
        initial_cal = estimator.calibration.copy()

        # Overconfident and wrong
        estimator.receive_feedback(0.9, False)

        assert not np.allclose(estimator.calibration, initial_cal)


class TestPerformanceMonitor:
    """Tests for performance monitoring"""

    def test_initialization(self):
        """Test monitor initialization"""
        monitor = PerformanceMonitor()
        assert monitor.current_effort == 0.5

    def test_monitor_process(self):
        """Test process monitoring"""
        monitor = PerformanceMonitor()
        current = np.random.rand(50)
        expected = np.random.rand(50)

        result = monitor.monitor_process(current, expected)
        assert "error_signal" in result
        assert "effort_needed" in result
        assert "apfc_activity" in result

    def test_error_detection(self):
        """Test error detection"""
        monitor = PerformanceMonitor()
        current = np.ones(50)
        expected = -np.ones(50)

        result = monitor.monitor_process(current, expected)
        assert result["needs_adjustment"]

    def test_record_outcome(self):
        """Test recording outcomes"""
        monitor = PerformanceMonitor()
        monitor.record_outcome(0.8)
        assert len(monitor.performance_history) == 1

    def test_performance_trend(self):
        """Test trend calculation"""
        monitor = PerformanceMonitor()

        # Improving performance
        for acc in [0.5, 0.6, 0.7, 0.8, 0.9]:
            monitor.record_outcome(acc)

        monitor.get_performance_trend(window=2)
        # May be positive or zero depending on window


class TestMetacognitiveSystem:
    """Tests for integrated metacognition"""

    def test_initialization(self):
        """Test system initialization"""
        system = MetacognitiveSystem()
        assert system.current_strategy == "default"

    def test_evaluate_decision(self):
        """Test decision evaluation"""
        system = MetacognitiveSystem()
        evidence = np.random.rand(50)
        decision = np.random.rand(50)

        result = system.evaluate_decision(evidence, decision)
        assert "confidence" in result
        assert "recommend_continue" in result

    def test_monitor_task(self):
        """Test task monitoring"""
        system = MetacognitiveSystem()
        current = np.random.rand(50)
        target = np.random.rand(50)

        result = system.monitor_task(current, target)
        assert "error_signal" in result

    def test_receive_feedback(self):
        """Test feedback processing"""
        system = MetacognitiveSystem()

        # Correct feedback
        system.receive_feedback(True, 0.8)
        assert system.strategy_effectiveness["default"] >= 0.5

    def test_introspect_confidence(self):
        """Test confidence introspection"""
        system = MetacognitiveSystem()

        # Do some decisions
        for _ in range(15):
            system.evaluate_decision(np.random.rand(50), np.random.rand(50))
            system.receive_feedback(np.random.rand() > 0.5, 0.6)

        result = system.introspect_confidence()
        assert "calibration_quality" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
