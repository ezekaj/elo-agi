"""
Efficiency Monitor for Meta-Reasoning

Tracks reasoning efficiency and decides when to terminate:
- Cost/benefit tracking
- Early termination decisions
- Resource allocation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np
import time

try:
    from .style_selector import ReasoningStyle
except ImportError:
    from style_selector import ReasoningStyle


class TerminationReason(Enum):
    """Reasons for early termination."""
    COMPLETED = "completed"
    TIME_LIMIT = "time_limit"
    COST_LIMIT = "cost_limit"
    DIMINISHING_RETURNS = "diminishing_returns"
    CONFIDENCE_SUFFICIENT = "confidence_sufficient"
    STUCK = "stuck"


@dataclass
class EfficiencyConfig:
    """Configuration for efficiency monitoring."""
    time_limit_seconds: float = 60.0
    cost_limit: float = 100.0
    min_progress_rate: float = 0.01
    confidence_threshold: float = 0.9
    stuck_threshold_steps: int = 10
    enable_early_termination: bool = True


@dataclass
class ReasoningMetrics:
    """Metrics for a reasoning session."""
    problem_id: str
    style: ReasoningStyle
    start_time: float
    elapsed_time: float
    steps_taken: int
    cost_accumulated: float
    progress: float
    confidence: float
    quality_estimate: float


@dataclass
class EfficiencyReport:
    """Report on reasoning efficiency."""
    efficiency_score: float
    cost_per_progress: float
    time_per_step: float
    recommendations: List[str]


class EfficiencyMonitor:
    """
    Monitors reasoning efficiency and resource usage.

    Capabilities:
    - Track time and cost
    - Detect diminishing returns
    - Decide when to terminate
    - Compute efficiency metrics
    """

    def __init__(
        self,
        config: Optional[EfficiencyConfig] = None,
        random_seed: Optional[int] = None,
    ):
        self.config = config or EfficiencyConfig()
        self._rng = np.random.default_rng(random_seed)

        self._active_sessions: Dict[str, ReasoningMetrics] = {}
        self._completed_sessions: List[ReasoningMetrics] = []

        self._progress_history: Dict[str, List[float]] = {}

    def start_monitoring(
        self,
        problem_id: str,
        style: ReasoningStyle,
    ) -> None:
        """
        Start monitoring a reasoning session.

        Args:
            problem_id: Problem identifier
            style: Reasoning style being used
        """
        metrics = ReasoningMetrics(
            problem_id=problem_id,
            style=style,
            start_time=time.time(),
            elapsed_time=0.0,
            steps_taken=0,
            cost_accumulated=0.0,
            progress=0.0,
            confidence=0.0,
            quality_estimate=0.0,
        )

        self._active_sessions[problem_id] = metrics
        self._progress_history[problem_id] = [0.0]

    def update_progress(
        self,
        problem_id: str,
        progress: float,
        confidence: float,
        cost: float = 1.0,
    ) -> None:
        """
        Update progress for a session.

        Args:
            problem_id: Problem identifier
            progress: Current progress [0, 1]
            confidence: Current confidence [0, 1]
            cost: Cost of this step
        """
        if problem_id not in self._active_sessions:
            return

        metrics = self._active_sessions[problem_id]
        metrics.elapsed_time = time.time() - metrics.start_time
        metrics.steps_taken += 1
        metrics.cost_accumulated += cost
        metrics.progress = progress
        metrics.confidence = confidence

        self._progress_history[problem_id].append(progress)

    def should_terminate_early(
        self,
        problem_id: str,
    ) -> Tuple[bool, Optional[TerminationReason]]:
        """
        Check if reasoning should terminate early.

        Args:
            problem_id: Problem identifier

        Returns:
            Tuple of (should_terminate, reason)
        """
        if not self.config.enable_early_termination:
            return False, None

        if problem_id not in self._active_sessions:
            return False, None

        metrics = self._active_sessions[problem_id]

        if metrics.elapsed_time > self.config.time_limit_seconds:
            return True, TerminationReason.TIME_LIMIT

        if metrics.cost_accumulated > self.config.cost_limit:
            return True, TerminationReason.COST_LIMIT

        if metrics.confidence >= self.config.confidence_threshold:
            return True, TerminationReason.CONFIDENCE_SUFFICIENT

        if self._check_diminishing_returns(problem_id):
            return True, TerminationReason.DIMINISHING_RETURNS

        if self._check_stuck(problem_id):
            return True, TerminationReason.STUCK

        return False, None

    def _check_diminishing_returns(self, problem_id: str) -> bool:
        """Check for diminishing returns."""
        history = self._progress_history.get(problem_id, [])

        if len(history) < 5:
            return False

        recent = history[-5:]
        progress_rate = (recent[-1] - recent[0]) / 5

        return progress_rate < self.config.min_progress_rate

    def _check_stuck(self, problem_id: str) -> bool:
        """Check if reasoning is stuck."""
        history = self._progress_history.get(problem_id, [])

        if len(history) < self.config.stuck_threshold_steps:
            return False

        recent = history[-self.config.stuck_threshold_steps:]
        variance = np.var(recent)

        return variance < 1e-6

    def complete_session(
        self,
        problem_id: str,
        final_quality: float,
    ) -> Optional[ReasoningMetrics]:
        """
        Complete a monitoring session.

        Args:
            problem_id: Problem identifier
            final_quality: Final quality of solution

        Returns:
            Final metrics
        """
        if problem_id not in self._active_sessions:
            return None

        metrics = self._active_sessions[problem_id]
        metrics.elapsed_time = time.time() - metrics.start_time
        metrics.quality_estimate = final_quality

        self._completed_sessions.append(metrics)
        del self._active_sessions[problem_id]

        if problem_id in self._progress_history:
            del self._progress_history[problem_id]

        return metrics

    def compute_efficiency_score(
        self,
        metrics: ReasoningMetrics,
    ) -> float:
        """
        Compute efficiency score for a session.

        Args:
            metrics: Session metrics

        Returns:
            Efficiency score [0, 1]
        """
        quality = metrics.quality_estimate
        time_factor = 1.0 / (1.0 + metrics.elapsed_time / self.config.time_limit_seconds)
        cost_factor = 1.0 / (1.0 + metrics.cost_accumulated / self.config.cost_limit)

        efficiency = quality * 0.5 + time_factor * 0.25 + cost_factor * 0.25

        return float(np.clip(efficiency, 0, 1))

    def get_efficiency_report(
        self,
        problem_id: str,
    ) -> Optional[EfficiencyReport]:
        """
        Get efficiency report for a session.

        Args:
            problem_id: Problem identifier

        Returns:
            EfficiencyReport or None
        """
        metrics = None

        if problem_id in self._active_sessions:
            metrics = self._active_sessions[problem_id]
        else:
            for m in self._completed_sessions:
                if m.problem_id == problem_id:
                    metrics = m
                    break

        if metrics is None:
            return None

        efficiency_score = self.compute_efficiency_score(metrics)

        cost_per_progress = (
            metrics.cost_accumulated / (metrics.progress + 1e-8)
            if metrics.progress > 0 else 0.0
        )

        time_per_step = (
            metrics.elapsed_time / metrics.steps_taken
            if metrics.steps_taken > 0 else 0.0
        )

        recommendations = self._generate_recommendations(metrics)

        return EfficiencyReport(
            efficiency_score=efficiency_score,
            cost_per_progress=cost_per_progress,
            time_per_step=time_per_step,
            recommendations=recommendations,
        )

    def _generate_recommendations(self, metrics: ReasoningMetrics) -> List[str]:
        """Generate efficiency recommendations."""
        recommendations = []

        if metrics.elapsed_time > self.config.time_limit_seconds * 0.5:
            recommendations.append("Consider faster heuristic approaches")

        if metrics.steps_taken > 0 and metrics.progress / metrics.steps_taken < 0.05:
            recommendations.append("Progress rate is low; consider switching strategy")

        if metrics.cost_accumulated > self.config.cost_limit * 0.5:
            recommendations.append("Cost is high; prioritize cheaper operations")

        history = self._progress_history.get(metrics.problem_id, [])
        if len(history) > 5:
            recent = history[-5:]
            if np.var(recent) < 1e-6:
                recommendations.append("Progress stalled; try different approach")

        if not recommendations:
            recommendations.append("Efficiency is acceptable")

        return recommendations

    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs."""
        return list(self._active_sessions.keys())

    def get_session_metrics(self, problem_id: str) -> Optional[ReasoningMetrics]:
        """Get metrics for a session."""
        return self._active_sessions.get(problem_id)

    def abort_session(self, problem_id: str) -> None:
        """Abort a monitoring session."""
        if problem_id in self._active_sessions:
            del self._active_sessions[problem_id]
        if problem_id in self._progress_history:
            del self._progress_history[problem_id]

    def statistics(self) -> Dict[str, Any]:
        """Get monitor statistics."""
        if not self._completed_sessions:
            return {
                "active_sessions": len(self._active_sessions),
                "completed_sessions": 0,
                "avg_efficiency": 0.0,
                "avg_time": 0.0,
                "avg_cost": 0.0,
            }

        efficiencies = [self.compute_efficiency_score(m) for m in self._completed_sessions]
        times = [m.elapsed_time for m in self._completed_sessions]
        costs = [m.cost_accumulated for m in self._completed_sessions]

        return {
            "active_sessions": len(self._active_sessions),
            "completed_sessions": len(self._completed_sessions),
            "avg_efficiency": float(np.mean(efficiencies)),
            "avg_time": float(np.mean(times)),
            "avg_cost": float(np.mean(costs)),
            "total_time": float(np.sum(times)),
            "total_cost": float(np.sum(costs)),
        }
