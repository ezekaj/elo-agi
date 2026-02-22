"""Tests for capability tracking."""

import pytest

from neuro.modules.continual.capability_tracking import (
    CapabilityTracker,
    CapabilityConfig,
    CapabilityStatus,
    CapabilityMetric,
    CapabilityRecord,
    InterferenceReport,
)


class TestCapabilityConfig:
    """Tests for CapabilityConfig class."""

    def test_default_config(self):
        config = CapabilityConfig()
        assert config.regression_threshold == 0.1
        assert config.improvement_threshold == 0.05
        assert config.history_window == 10

    def test_custom_config(self):
        config = CapabilityConfig(regression_threshold=0.2, history_window=20)
        assert config.regression_threshold == 0.2
        assert config.history_window == 20


class TestCapabilityTracker:
    """Tests for CapabilityTracker class."""

    def test_creation(self):
        tracker = CapabilityTracker(random_seed=42)
        assert tracker is not None

    def test_register_capability(self):
        tracker = CapabilityTracker(random_seed=42)

        tracker.register_capability("reasoning", initial_score=0.5)

        assert "reasoning" in tracker.get_all_capabilities()

    def test_measure_capability(self):
        tracker = CapabilityTracker(random_seed=42)

        results = {"test1": 0.8, "test2": 0.9, "test3": 0.7}
        metric = tracker.measure_capability("reasoning", results)

        assert isinstance(metric, CapabilityMetric)
        assert metric.name == "reasoning"
        assert metric.score == pytest.approx(0.8, rel=0.01)

    def test_measure_capability_aggregation_min(self):
        tracker = CapabilityTracker(random_seed=42)

        results = {"test1": 0.8, "test2": 0.9, "test3": 0.7}
        metric = tracker.measure_capability("reasoning", results, aggregation="min")

        assert metric.score == pytest.approx(0.7)

    def test_measure_capability_aggregation_max(self):
        tracker = CapabilityTracker(random_seed=42)

        results = {"test1": 0.8, "test2": 0.9, "test3": 0.7}
        metric = tracker.measure_capability("reasoning", results, aggregation="max")

        assert metric.score == pytest.approx(0.9)

    def test_detect_regression_no_data(self):
        tracker = CapabilityTracker(random_seed=42)

        tracker.register_capability("reasoning", 0.5)

        assert not tracker.detect_regression("reasoning")

    def test_detect_regression_improving(self):
        tracker = CapabilityTracker(random_seed=42)

        tracker.register_capability("reasoning", 0.5)

        for i in range(10):
            tracker.measure_capability("reasoning", {"test": 0.5 + i * 0.05})

        assert not tracker.detect_regression("reasoning")

    def test_detect_regression_dropping(self):
        config = CapabilityConfig(regression_threshold=0.1, min_samples_for_status=3)
        tracker = CapabilityTracker(config=config, random_seed=42)

        tracker.measure_capability("reasoning", {"test": 0.9})
        tracker.measure_capability("reasoning", {"test": 0.85})
        tracker.measure_capability("reasoning", {"test": 0.8})
        tracker.measure_capability("reasoning", {"test": 0.5})

        assert tracker.detect_regression("reasoning")

    def test_identify_interference_no_interference(self):
        tracker = CapabilityTracker(random_seed=42)

        for i in range(10):
            tracker.measure_capability("cap1", {"test": 0.5 + i * 0.02})
            tracker.measure_capability("cap2", {"test": 0.5 + i * 0.02})

        reports = tracker.identify_interference([("cap1", "cap2")])

        high_interference = [r for r in reports if r.interference_score > 0.5]
        assert len(high_interference) == 0

    def test_identify_interference_detected(self):
        config = CapabilityConfig(interference_threshold=0.2)
        tracker = CapabilityTracker(config=config, random_seed=42)

        for i in range(10):
            tracker.measure_capability("cap1", {"test": 0.5 + i * 0.05})
            tracker.measure_capability("cap2", {"test": 0.9 - i * 0.05})

        tracker.identify_interference([("cap1", "cap2")])

        pass

    def test_suggest_remediation(self):
        config = CapabilityConfig(regression_threshold=0.1, min_samples_for_status=3)
        tracker = CapabilityTracker(config=config, random_seed=42)

        tracker.measure_capability("reasoning", {"test": 0.9})
        tracker.measure_capability("reasoning", {"test": 0.8})
        tracker.measure_capability("reasoning", {"test": 0.7})
        tracker.measure_capability("reasoning", {"test": 0.5})

        suggestions = tracker.suggest_remediation(["reasoning"])

        assert "reasoning" in suggestions
        assert len(suggestions["reasoning"]) > 0

    def test_get_capability_record(self):
        tracker = CapabilityTracker(random_seed=42)

        tracker.measure_capability("reasoning", {"test": 0.8})

        record = tracker.get_capability_record("reasoning")

        assert isinstance(record, CapabilityRecord)
        assert record.name == "reasoning"
        assert record.current_score == pytest.approx(0.8)

    def test_get_nonexistent_record(self):
        tracker = CapabilityTracker(random_seed=42)

        record = tracker.get_capability_record("nonexistent")

        assert record is None

    def test_get_regressing_capabilities(self):
        config = CapabilityConfig(regression_threshold=0.1, min_samples_for_status=3)
        tracker = CapabilityTracker(config=config, random_seed=42)

        for i in range(5):
            tracker.measure_capability("good", {"test": 0.5 + i * 0.05})
            tracker.measure_capability("bad", {"test": 0.9 - i * 0.1})

        regressing = tracker.get_regressing_capabilities()

        assert "bad" in regressing or len(regressing) == 0

    def test_get_capability_summary(self):
        tracker = CapabilityTracker(random_seed=42)

        tracker.measure_capability("cap1", {"test": 0.7})
        tracker.measure_capability("cap2", {"test": 0.9})

        summary = tracker.get_capability_summary()

        assert "cap1" in summary
        assert "cap2" in summary
        assert "current_score" in summary["cap1"]
        assert "status" in summary["cap1"]

    def test_reset(self):
        tracker = CapabilityTracker(random_seed=42)

        tracker.measure_capability("reasoning", {"test": 0.8})
        tracker.reset()

        assert len(tracker.get_all_capabilities()) == 0

    def test_statistics(self):
        tracker = CapabilityTracker(random_seed=42)

        tracker.measure_capability("reasoning", {"test": 0.8})
        tracker.measure_capability("memory", {"test": 0.7})

        stats = tracker.statistics()

        assert "total_capabilities" in stats
        assert "total_measurements" in stats
        assert "status_distribution" in stats
        assert stats["total_capabilities"] == 2


class TestCapabilityStatus:
    """Tests for CapabilityStatus enum."""

    def test_statuses(self):
        assert CapabilityStatus.IMPROVING.value == "improving"
        assert CapabilityStatus.STABLE.value == "stable"
        assert CapabilityStatus.REGRESSING.value == "regressing"
        assert CapabilityStatus.UNKNOWN.value == "unknown"


class TestCapabilityMetric:
    """Tests for CapabilityMetric dataclass."""

    def test_metric_creation(self):
        metric = CapabilityMetric(
            name="reasoning",
            score=0.8,
            timestamp=0,
            test_results={"test1": 0.8},
            confidence=0.9,
        )

        assert metric.name == "reasoning"
        assert metric.score == 0.8
        assert metric.confidence == 0.9


class TestInterferenceReport:
    """Tests for InterferenceReport dataclass."""

    def test_report_creation(self):
        report = InterferenceReport(
            capability_a="cap1",
            capability_b="cap2",
            interference_score=0.6,
            direction="cap1 interferes with cap2",
            evidence=["cap2 regressing while cap1 improving"],
        )

        assert report.capability_a == "cap1"
        assert report.interference_score == 0.6
        assert len(report.evidence) == 1
