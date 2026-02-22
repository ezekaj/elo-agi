"""Tests for efficiency monitor."""

import time

from neuro.modules.meta_reasoning.efficiency_monitor import (
    EfficiencyMonitor,
    EfficiencyConfig,
    ReasoningMetrics,
    EfficiencyReport,
    TerminationReason,
)
from neuro.modules.meta_reasoning.style_selector import ReasoningStyle


class TestEfficiencyConfig:
    """Tests for EfficiencyConfig class."""

    def test_default_config(self):
        config = EfficiencyConfig()
        assert config.time_limit_seconds == 60.0
        assert config.cost_limit == 100.0
        assert config.enable_early_termination

    def test_custom_config(self):
        config = EfficiencyConfig(time_limit_seconds=30.0, cost_limit=50.0)
        assert config.time_limit_seconds == 30.0
        assert config.cost_limit == 50.0


class TestEfficiencyMonitor:
    """Tests for EfficiencyMonitor class."""

    def test_creation(self):
        monitor = EfficiencyMonitor(random_seed=42)
        assert monitor is not None

    def test_start_monitoring(self):
        monitor = EfficiencyMonitor(random_seed=42)

        monitor.start_monitoring("problem1", ReasoningStyle.DEDUCTIVE)

        assert "problem1" in monitor.get_active_sessions()

    def test_update_progress(self):
        monitor = EfficiencyMonitor(random_seed=42)

        monitor.start_monitoring("problem1", ReasoningStyle.DEDUCTIVE)
        monitor.update_progress("problem1", progress=0.5, confidence=0.7, cost=1.0)

        metrics = monitor.get_session_metrics("problem1")
        assert metrics.progress == 0.5
        assert metrics.confidence == 0.7

    def test_should_terminate_time_limit(self):
        config = EfficiencyConfig(time_limit_seconds=0.01)
        monitor = EfficiencyMonitor(config=config, random_seed=42)

        monitor.start_monitoring("problem1", ReasoningStyle.DEDUCTIVE)
        time.sleep(0.02)
        monitor.update_progress("problem1", 0.1, 0.5)

        should_term, reason = monitor.should_terminate_early("problem1")

        assert should_term
        assert reason == TerminationReason.TIME_LIMIT

    def test_should_terminate_cost_limit(self):
        config = EfficiencyConfig(cost_limit=5.0)
        monitor = EfficiencyMonitor(config=config, random_seed=42)

        monitor.start_monitoring("problem1", ReasoningStyle.DEDUCTIVE)

        for _ in range(10):
            monitor.update_progress("problem1", 0.1, 0.5, cost=1.0)

        should_term, reason = monitor.should_terminate_early("problem1")

        assert should_term
        assert reason == TerminationReason.COST_LIMIT

    def test_should_terminate_confidence(self):
        config = EfficiencyConfig(confidence_threshold=0.9)
        monitor = EfficiencyMonitor(config=config, random_seed=42)

        monitor.start_monitoring("problem1", ReasoningStyle.DEDUCTIVE)
        monitor.update_progress("problem1", 0.95, 0.95, cost=1.0)

        should_term, reason = monitor.should_terminate_early("problem1")

        assert should_term
        assert reason == TerminationReason.CONFIDENCE_SUFFICIENT

    def test_no_termination(self):
        config = EfficiencyConfig(
            time_limit_seconds=100.0,
            cost_limit=100.0,
            confidence_threshold=0.99,
        )
        monitor = EfficiencyMonitor(config=config, random_seed=42)

        monitor.start_monitoring("problem1", ReasoningStyle.DEDUCTIVE)
        monitor.update_progress("problem1", 0.5, 0.7, cost=1.0)

        should_term, reason = monitor.should_terminate_early("problem1")

        assert not should_term
        assert reason is None

    def test_complete_session(self):
        monitor = EfficiencyMonitor(random_seed=42)

        monitor.start_monitoring("problem1", ReasoningStyle.DEDUCTIVE)
        monitor.update_progress("problem1", 1.0, 0.9, cost=5.0)

        metrics = monitor.complete_session("problem1", final_quality=0.85)

        assert metrics is not None
        assert metrics.quality_estimate == 0.85
        assert "problem1" not in monitor.get_active_sessions()

    def test_compute_efficiency_score(self):
        monitor = EfficiencyMonitor(random_seed=42)

        metrics = ReasoningMetrics(
            problem_id="test",
            style=ReasoningStyle.DEDUCTIVE,
            start_time=time.time(),
            elapsed_time=10.0,
            steps_taken=5,
            cost_accumulated=10.0,
            progress=1.0,
            confidence=0.9,
            quality_estimate=0.8,
        )

        score = monitor.compute_efficiency_score(metrics)

        assert 0 <= score <= 1

    def test_get_efficiency_report(self):
        monitor = EfficiencyMonitor(random_seed=42)

        monitor.start_monitoring("problem1", ReasoningStyle.DEDUCTIVE)
        for i in range(5):
            monitor.update_progress("problem1", (i + 1) * 0.2, 0.7, cost=1.0)

        report = monitor.get_efficiency_report("problem1")

        assert isinstance(report, EfficiencyReport)
        assert 0 <= report.efficiency_score <= 1
        assert len(report.recommendations) > 0

    def test_get_nonexistent_report(self):
        monitor = EfficiencyMonitor(random_seed=42)

        report = monitor.get_efficiency_report("nonexistent")

        assert report is None

    def test_abort_session(self):
        monitor = EfficiencyMonitor(random_seed=42)

        monitor.start_monitoring("problem1", ReasoningStyle.DEDUCTIVE)
        monitor.abort_session("problem1")

        assert "problem1" not in monitor.get_active_sessions()

    def test_statistics(self):
        monitor = EfficiencyMonitor(random_seed=42)

        monitor.start_monitoring("problem1", ReasoningStyle.DEDUCTIVE)
        monitor.update_progress("problem1", 1.0, 0.9, cost=5.0)
        monitor.complete_session("problem1", 0.8)

        stats = monitor.statistics()

        assert "active_sessions" in stats
        assert "completed_sessions" in stats
        assert "avg_efficiency" in stats
        assert stats["completed_sessions"] == 1


class TestTerminationReason:
    """Tests for TerminationReason enum."""

    def test_reasons(self):
        assert TerminationReason.COMPLETED.value == "completed"
        assert TerminationReason.TIME_LIMIT.value == "time_limit"
        assert TerminationReason.COST_LIMIT.value == "cost_limit"
        assert TerminationReason.DIMINISHING_RETURNS.value == "diminishing_returns"


class TestReasoningMetrics:
    """Tests for ReasoningMetrics dataclass."""

    def test_metrics_creation(self):
        metrics = ReasoningMetrics(
            problem_id="test",
            style=ReasoningStyle.DEDUCTIVE,
            start_time=time.time(),
            elapsed_time=5.0,
            steps_taken=10,
            cost_accumulated=15.0,
            progress=0.8,
            confidence=0.9,
            quality_estimate=0.85,
        )

        assert metrics.problem_id == "test"
        assert metrics.progress == 0.8
        assert metrics.steps_taken == 10
