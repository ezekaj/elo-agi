"""
Capability Tracking for Continual Learning

Monitors capability growth and regression:
- Measure capabilities over time
- Detect regression in learned skills
- Identify interference between capabilities
- Suggest remediation strategies
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import numpy as np


class CapabilityStatus(Enum):
    """Status of a capability."""
    IMPROVING = "improving"
    STABLE = "stable"
    REGRESSING = "regressing"
    UNKNOWN = "unknown"


@dataclass
class CapabilityConfig:
    """Configuration for capability tracking."""
    regression_threshold: float = 0.1
    improvement_threshold: float = 0.05
    history_window: int = 10
    min_samples_for_status: int = 3
    interference_threshold: float = 0.3


@dataclass
class CapabilityMetric:
    """Measurement of a capability."""
    name: str
    score: float
    timestamp: int
    test_results: Optional[Dict[str, float]] = None
    confidence: float = 1.0


@dataclass
class CapabilityRecord:
    """Historical record of a capability."""
    name: str
    history: List[CapabilityMetric]
    peak_score: float
    current_score: float
    status: CapabilityStatus
    regression_count: int


@dataclass
class InterferenceReport:
    """Report of interference between capabilities."""
    capability_a: str
    capability_b: str
    interference_score: float
    direction: str
    evidence: List[str]


class CapabilityTracker:
    """
    Tracks capabilities and detects regression.

    Monitors:
    - Performance trends across capabilities
    - Interference patterns between skills
    - Recovery recommendations
    """

    def __init__(
        self,
        config: Optional[CapabilityConfig] = None,
        random_seed: Optional[int] = None,
    ):
        self.config = config or CapabilityConfig()
        self._rng = np.random.default_rng(random_seed)

        self._capabilities: Dict[str, CapabilityRecord] = {}
        self._interference_matrix: Dict[Tuple[str, str], List[float]] = {}

        self._timestep = 0
        self._total_measurements = 0

    def register_capability(self, name: str, initial_score: float = 0.0) -> None:
        """Register a new capability to track."""
        if name not in self._capabilities:
            metric = CapabilityMetric(
                name=name,
                score=initial_score,
                timestamp=self._timestep,
            )
            self._capabilities[name] = CapabilityRecord(
                name=name,
                history=[metric],
                peak_score=initial_score,
                current_score=initial_score,
                status=CapabilityStatus.UNKNOWN,
                regression_count=0,
            )

    def measure_capability(
        self,
        name: str,
        test_results: Dict[str, float],
        aggregation: str = "mean",
    ) -> CapabilityMetric:
        """
        Measure a capability from test results.

        Args:
            name: Capability name
            test_results: Dict mapping test names to scores
            aggregation: How to aggregate results (mean, min, max)

        Returns:
            CapabilityMetric with the measurement
        """
        if not test_results:
            score = 0.0
        elif aggregation == "mean":
            score = float(np.mean(list(test_results.values())))
        elif aggregation == "min":
            score = float(np.min(list(test_results.values())))
        elif aggregation == "max":
            score = float(np.max(list(test_results.values())))
        else:
            score = float(np.mean(list(test_results.values())))

        confidence = min(1.0, len(test_results) / 5.0)

        metric = CapabilityMetric(
            name=name,
            score=score,
            timestamp=self._timestep,
            test_results=test_results,
            confidence=confidence,
        )

        self._record_measurement(name, metric)
        self._timestep += 1
        self._total_measurements += 1

        return metric

    def _record_measurement(self, name: str, metric: CapabilityMetric) -> None:
        """Record a capability measurement."""
        if name not in self._capabilities:
            self.register_capability(name, metric.score)
            return

        record = self._capabilities[name]
        record.history.append(metric)

        if len(record.history) > self.config.history_window * 2:
            record.history = record.history[-self.config.history_window * 2:]

        record.current_score = metric.score
        record.peak_score = max(record.peak_score, metric.score)

        record.status = self._compute_status(record)

        if record.status == CapabilityStatus.REGRESSING:
            record.regression_count += 1

    def _compute_status(self, record: CapabilityRecord) -> CapabilityStatus:
        """Compute capability status from history."""
        if len(record.history) < self.config.min_samples_for_status:
            return CapabilityStatus.UNKNOWN

        recent = record.history[-self.config.history_window:]
        scores = [m.score for m in recent]

        if len(scores) < 2:
            return CapabilityStatus.UNKNOWN

        trend = self._compute_trend(scores)
        gap = record.peak_score - record.current_score

        if gap > self.config.regression_threshold * record.peak_score:
            return CapabilityStatus.REGRESSING
        elif trend > self.config.improvement_threshold:
            return CapabilityStatus.IMPROVING
        elif trend < -self.config.regression_threshold:
            return CapabilityStatus.REGRESSING
        else:
            return CapabilityStatus.STABLE

    def _compute_trend(self, scores: List[float]) -> float:
        """Compute linear trend in scores."""
        if len(scores) < 2:
            return 0.0

        x = np.arange(len(scores))
        y = np.array(scores)

        x_mean = np.mean(x)
        y_mean = np.mean(y)

        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2) + 1e-8

        slope = numerator / denominator

        if y_mean > 1e-8:
            normalized_slope = slope / y_mean
        else:
            normalized_slope = slope

        return float(normalized_slope)

    def detect_regression(self, name: str) -> bool:
        """
        Detect if a capability is regressing.

        Args:
            name: Capability name

        Returns:
            True if capability is regressing
        """
        if name not in self._capabilities:
            return False

        record = self._capabilities[name]
        return record.status == CapabilityStatus.REGRESSING

    def identify_interference(
        self,
        pairs: Optional[List[Tuple[str, str]]] = None,
    ) -> List[InterferenceReport]:
        """
        Identify interference between capabilities.

        Args:
            pairs: Optional list of capability pairs to check

        Returns:
            List of interference reports
        """
        if pairs is None:
            cap_names = list(self._capabilities.keys())
            pairs = [(a, b) for i, a in enumerate(cap_names) for b in cap_names[i + 1:]]

        reports = []

        for cap_a, cap_b in pairs:
            if cap_a not in self._capabilities or cap_b not in self._capabilities:
                continue

            score = self._compute_interference(cap_a, cap_b)

            if abs(score) > self.config.interference_threshold:
                direction = "bidirectional"
                if score > 0:
                    direction = f"{cap_a} interferes with {cap_b}"
                else:
                    direction = f"{cap_b} interferes with {cap_a}"

                evidence = self._gather_interference_evidence(cap_a, cap_b)

                reports.append(InterferenceReport(
                    capability_a=cap_a,
                    capability_b=cap_b,
                    interference_score=abs(score),
                    direction=direction,
                    evidence=evidence,
                ))

        return reports

    def _compute_interference(self, cap_a: str, cap_b: str) -> float:
        """Compute interference score between two capabilities."""
        rec_a = self._capabilities[cap_a]
        rec_b = self._capabilities[cap_b]

        if len(rec_a.history) < 3 or len(rec_b.history) < 3:
            return 0.0

        scores_a = [m.score for m in rec_a.history[-10:]]
        scores_b = [m.score for m in rec_b.history[-10:]]

        min_len = min(len(scores_a), len(scores_b))
        if min_len < 2:
            return 0.0

        scores_a = scores_a[-min_len:]
        scores_b = scores_b[-min_len:]

        changes_a = np.diff(scores_a)
        changes_b = np.diff(scores_b)

        if len(changes_a) == 0 or len(changes_b) == 0:
            return 0.0

        if np.std(changes_a) < 1e-8 or np.std(changes_b) < 1e-8:
            return 0.0

        correlation = np.corrcoef(changes_a, changes_b)[0, 1]
        if np.isnan(correlation):
            return 0.0

        interference = -correlation

        return float(interference)

    def _gather_interference_evidence(self, cap_a: str, cap_b: str) -> List[str]:
        """Gather evidence for interference."""
        evidence = []

        rec_a = self._capabilities[cap_a]
        rec_b = self._capabilities[cap_b]

        if rec_a.status == CapabilityStatus.REGRESSING and rec_b.status == CapabilityStatus.IMPROVING:
            evidence.append(f"{cap_a} regressing while {cap_b} improving")
        elif rec_b.status == CapabilityStatus.REGRESSING and rec_a.status == CapabilityStatus.IMPROVING:
            evidence.append(f"{cap_b} regressing while {cap_a} improving")

        if rec_a.regression_count > 2:
            evidence.append(f"{cap_a} has regressed {rec_a.regression_count} times")
        if rec_b.regression_count > 2:
            evidence.append(f"{cap_b} has regressed {rec_b.regression_count} times")

        return evidence

    def suggest_remediation(
        self,
        regressed: List[str],
    ) -> Dict[str, List[str]]:
        """
        Suggest remediation for regressed capabilities.

        Args:
            regressed: List of regressed capability names

        Returns:
            Dict mapping capability to suggested actions
        """
        suggestions: Dict[str, List[str]] = {}

        for name in regressed:
            if name not in self._capabilities:
                continue

            record = self._capabilities[name]
            actions = []

            gap = record.peak_score - record.current_score
            if gap > 0.2 * record.peak_score:
                actions.append("Increase replay frequency for this capability")

            if record.regression_count > 3:
                actions.append("Apply stronger regularization (EWC/SI)")

            interferences = self.identify_interference(
                [(name, other) for other in self._capabilities if other != name]
            )

            for report in interferences:
                if report.interference_score > 0.5:
                    other = report.capability_b if report.capability_a == name else report.capability_a
                    actions.append(f"Reduce concurrent training with {other}")

            if not actions:
                actions.append("Continue monitoring")

            suggestions[name] = actions

        return suggestions

    def get_capability_record(self, name: str) -> Optional[CapabilityRecord]:
        """Get record for a capability."""
        return self._capabilities.get(name)

    def get_all_capabilities(self) -> List[str]:
        """Get all capability names."""
        return list(self._capabilities.keys())

    def get_regressing_capabilities(self) -> List[str]:
        """Get all regressing capabilities."""
        return [
            name for name, rec in self._capabilities.items()
            if rec.status == CapabilityStatus.REGRESSING
        ]

    def get_capability_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all capabilities."""
        summary = {}

        for name, record in self._capabilities.items():
            summary[name] = {
                "current_score": record.current_score,
                "peak_score": record.peak_score,
                "status": record.status.value,
                "regression_count": record.regression_count,
                "history_length": len(record.history),
            }

        return summary

    def reset(self) -> None:
        """Reset all tracking."""
        self._capabilities.clear()
        self._interference_matrix.clear()
        self._timestep = 0
        self._total_measurements = 0

    def statistics(self) -> Dict[str, Any]:
        """Get tracking statistics."""
        status_counts = {}
        for record in self._capabilities.values():
            status = record.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "total_capabilities": len(self._capabilities),
            "total_measurements": self._total_measurements,
            "timestep": self._timestep,
            "status_distribution": status_counts,
            "regressing_count": len(self.get_regressing_capabilities()),
            "capabilities": list(self._capabilities.keys()),
        }
