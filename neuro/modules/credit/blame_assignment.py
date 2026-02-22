"""
Blame Assignment for Failure Attribution

Implements multi-step failure attribution using:
- Causal analysis
- Counterfactual reasoning
- Module contribution tracking
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import numpy as np


class FailureType(Enum):
    """Types of failures."""

    TIMEOUT = "timeout"
    CONSTRAINT_VIOLATION = "constraint_violation"
    GOAL_UNREACHED = "goal_unreached"
    PERFORMANCE_DROP = "performance_drop"
    UNEXPECTED_STATE = "unexpected_state"
    MODULE_ERROR = "module_error"


@dataclass
class Failure:
    """A detected failure event."""

    failure_type: FailureType
    description: str
    timestamp: int
    severity: float
    context: Dict[str, Any] = field(default_factory=dict)
    trajectory_index: Optional[int] = None


@dataclass
class BlameResult:
    """Result of blame assignment."""

    module_id: str
    blame_score: float
    confidence: float
    evidence: List[str]
    counterfactual_impact: float
    contribution_to_failure: float


@dataclass
class ModuleAction:
    """Record of a module action."""

    module_id: str
    action: Any
    state_before: Any
    state_after: Any
    timestamp: int
    predicted_outcome: Optional[Any] = None


class CounterfactualBlame:
    """
    Counterfactual blame analysis.

    Estimates what would have happened if a module had acted differently.
    """

    def __init__(
        self,
        world_model: Optional[Callable[[Any, Any], Any]] = None,
        random_seed: Optional[int] = None,
        action_space_size: int = 10,
    ):
        self._rng = np.random.default_rng(random_seed)
        self._world_model = world_model
        self._action_space_size = action_space_size

    def set_world_model(self, model: Callable[[Any, Any], Any]) -> None:
        """Set the world model for counterfactual simulation."""
        self._world_model = model

    def compute_counterfactual(
        self,
        original_trajectory: List[ModuleAction],
        intervention_module: str,
        alternative_actions: List[Any],
    ) -> Tuple[float, List[Any]]:
        """
        Compute counterfactual outcome with alternative actions.

        Args:
            original_trajectory: Original sequence of actions
            intervention_module: Module to intervene on
            alternative_actions: Alternative actions for the module

        Returns:
            Tuple of (counterfactual_outcome, simulated_trajectory)
        """
        if self._world_model is None:
            return 0.0, []

        simulated = []
        current_state = original_trajectory[0].state_before if original_trajectory else None
        alt_idx = 0

        for action_record in original_trajectory:
            if action_record.module_id == intervention_module and alt_idx < len(
                alternative_actions
            ):
                action = alternative_actions[alt_idx]
                alt_idx += 1
            else:
                action = action_record.action

            next_state = self._world_model(current_state, action)
            simulated.append(next_state)
            current_state = next_state

        if simulated:
            if isinstance(simulated[-1], (int, float)):
                outcome = float(simulated[-1])
            elif isinstance(simulated[-1], np.ndarray):
                outcome = float(np.mean(simulated[-1]))
            else:
                outcome = 0.0
        else:
            outcome = 0.0

        return outcome, simulated

    def compute_counterfactual_blame(
        self,
        failure: Failure,
        trajectory: List[ModuleAction],
        intervention_module: str,
    ) -> float:
        """
        Compute counterfactual blame for a module.

        Blame = P(failure | actual_action) - P(failure | optimal_action)

        Args:
            failure: The failure event
            trajectory: Action trajectory
            intervention_module: Module to analyze

        Returns:
            Counterfactual blame score
        """
        module_actions = [a for a in trajectory if a.module_id == intervention_module]

        if not module_actions:
            return 0.0

        if self._world_model is None:
            return failure.severity * 0.5

        actual_outcome, _ = self.compute_counterfactual(
            trajectory, intervention_module, [a.action for a in module_actions]
        )

        random_actions = [self._rng.integers(0, self._action_space_size) for _ in module_actions]
        counterfactual_outcome, _ = self.compute_counterfactual(
            trajectory, intervention_module, random_actions
        )

        blame = actual_outcome - counterfactual_outcome

        return float(np.clip(blame, -1.0, 1.0))


class BlameAssignment:
    """
    Main blame assignment system.

    Combines multiple methods to attribute failures to modules:
    - Temporal proximity
    - Counterfactual analysis
    - Historical patterns
    """

    def __init__(
        self,
        random_seed: Optional[int] = None,
    ):
        self._rng = np.random.default_rng(random_seed)

        self._counterfactual = CounterfactualBlame(random_seed=random_seed)
        self._failure_history: List[Tuple[Failure, Dict[str, float]]] = []
        self._module_failure_counts: Dict[str, int] = {}

        self._total_failures = 0
        self._total_blame_assignments = 0

    def set_world_model(self, model: Callable[[Any, Any], Any]) -> None:
        """Set the world model for counterfactual analysis."""
        self._counterfactual.set_world_model(model)

    def identify_root_cause(
        self,
        failure: Failure,
        trajectory: List[ModuleAction],
        module_ids: Optional[List[str]] = None,
    ) -> Tuple[str, float]:
        """
        Identify the root cause module for a failure.

        Args:
            failure: The failure event
            trajectory: Action trajectory
            module_ids: Optional list of modules to consider

        Returns:
            Tuple of (most_blamed_module, confidence)
        """
        if module_ids is None:
            module_ids = list(set(a.module_id for a in trajectory))

        if not module_ids:
            return "", 0.0

        blame_scores = {}
        for module_id in module_ids:
            result = self.compute_blame(failure, trajectory, module_id)
            blame_scores[module_id] = result.blame_score

        if not blame_scores:
            return "", 0.0

        max_module = max(blame_scores.keys(), key=lambda k: blame_scores[k])
        max_score = blame_scores[max_module]

        total_score = sum(blame_scores.values())
        confidence = max_score / total_score if total_score > 0 else 0.0

        return max_module, confidence

    def compute_blame(
        self,
        failure: Failure,
        trajectory: List[ModuleAction],
        module_id: str,
    ) -> BlameResult:
        """
        Compute blame for a specific module.

        Args:
            failure: The failure event
            trajectory: Action trajectory
            module_id: Module to analyze

        Returns:
            BlameResult with detailed attribution
        """
        evidence = []

        module_actions = [a for a in trajectory if a.module_id == module_id]
        temporal_blame = self._compute_temporal_blame(failure, module_actions)
        evidence.append(f"temporal_proximity: {temporal_blame:.2f}")

        counterfactual_impact = self._counterfactual.compute_counterfactual_blame(
            failure, trajectory, module_id
        )
        evidence.append(f"counterfactual_impact: {counterfactual_impact:.2f}")

        historical_blame = self._compute_historical_blame(module_id, failure.failure_type)
        evidence.append(f"historical_pattern: {historical_blame:.2f}")

        contribution = self._compute_contribution(module_actions, trajectory)
        evidence.append(f"contribution: {contribution:.2f}")

        blame_score = (
            0.3 * temporal_blame
            + 0.3 * counterfactual_impact
            + 0.2 * historical_blame
            + 0.2 * contribution
        )

        confidence = self._compute_confidence(len(module_actions), len(trajectory))

        self._total_blame_assignments += 1

        return BlameResult(
            module_id=module_id,
            blame_score=blame_score,
            confidence=confidence,
            evidence=evidence,
            counterfactual_impact=counterfactual_impact,
            contribution_to_failure=contribution,
        )

    def propagate_blame(
        self,
        failure: Failure,
        blame_distribution: Dict[str, float],
    ) -> None:
        """
        Record blame distribution for a failure.

        Args:
            failure: The failure event
            blame_distribution: Dict mapping module_id to blame score
        """
        self._failure_history.append((failure, blame_distribution))
        self._total_failures += 1

        for module_id, score in blame_distribution.items():
            if score > 0.5:
                self._module_failure_counts[module_id] = (
                    self._module_failure_counts.get(module_id, 0) + 1
                )

    def get_module_blame_history(self, module_id: str) -> List[Tuple[Failure, float]]:
        """Get blame history for a module."""
        history = []
        for failure, distribution in self._failure_history:
            if module_id in distribution:
                history.append((failure, distribution[module_id]))
        return history

    def get_chronic_offenders(self, threshold: int = 3) -> List[Tuple[str, int]]:
        """Get modules with repeated failures."""
        offenders = [(m, c) for m, c in self._module_failure_counts.items() if c >= threshold]
        return sorted(offenders, key=lambda x: x[1], reverse=True)

    def _compute_temporal_blame(
        self,
        failure: Failure,
        module_actions: List[ModuleAction],
    ) -> float:
        """Compute blame based on temporal proximity to failure."""
        if not module_actions:
            return 0.0

        failure_time = failure.timestamp
        blame = 0.0

        for action in module_actions:
            time_diff = abs(failure_time - action.timestamp)
            proximity = np.exp(-time_diff / 10.0)
            blame = max(blame, proximity)

        return float(blame)

    def _compute_historical_blame(
        self,
        module_id: str,
        failure_type: FailureType,
    ) -> float:
        """Compute blame based on historical patterns."""
        relevant_failures = [
            (f, d)
            for f, d in self._failure_history
            if f.failure_type == failure_type and module_id in d
        ]

        if not relevant_failures:
            return 0.0

        avg_blame = np.mean([d[module_id] for _, d in relevant_failures])
        return float(avg_blame)

    def _compute_contribution(
        self,
        module_actions: List[ModuleAction],
        full_trajectory: List[ModuleAction],
    ) -> float:
        """Compute module's contribution to the trajectory."""
        if not full_trajectory:
            return 0.0

        return len(module_actions) / len(full_trajectory)

    def _compute_confidence(
        self,
        module_action_count: int,
        total_action_count: int,
    ) -> float:
        """Compute confidence in blame assignment."""
        if total_action_count == 0:
            return 0.0

        coverage = module_action_count / total_action_count
        confidence = min(1.0, coverage * 2)

        return float(confidence)

    def statistics(self) -> Dict[str, Any]:
        """Get blame assignment statistics."""
        return {
            "total_failures": self._total_failures,
            "total_blame_assignments": self._total_blame_assignments,
            "failure_history_size": len(self._failure_history),
            "module_failure_counts": dict(self._module_failure_counts),
            "chronic_offenders": self.get_chronic_offenders(),
        }
