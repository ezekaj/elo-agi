"""
Eligibility Traces for Temporal Credit Assignment

Implements TD(lambda) style eligibility traces for assigning credit
to past state-action pairs when rewards are received.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
import numpy as np


class TraceType(Enum):
    """Type of eligibility trace."""

    ACCUMULATING = "accumulating"
    REPLACING = "replacing"
    DUTCH = "dutch"


@dataclass
class TraceConfig:
    """Configuration for eligibility traces."""

    trace_type: TraceType = TraceType.ACCUMULATING
    lambda_param: float = 0.9
    gamma: float = 0.99
    initial_trace: float = 1.0
    min_trace: float = 1e-6
    max_trace: float = 10.0
    use_importance_sampling: bool = False


@dataclass
class EligibilityTrace:
    """An individual eligibility trace."""

    state_key: str
    action_key: str
    module_id: str
    trace_value: float
    timestamp: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def decay(self, gamma_lambda: float) -> None:
        """Decay the trace value."""
        self.trace_value *= gamma_lambda

    def accumulate(self, value: float, max_trace: float) -> None:
        """Accumulate to the trace value."""
        self.trace_value = min(self.trace_value + value, max_trace)

    def replace(self, value: float) -> None:
        """Replace the trace value."""
        self.trace_value = value

    def dutch_update(self, value: float, gamma_lambda: float, max_trace: float) -> None:
        """Dutch trace update: e = gamma*lambda*e + (1 - gamma*lambda*e)*value."""
        self.trace_value = (
            gamma_lambda * self.trace_value + (1 - gamma_lambda * self.trace_value) * value
        )
        self.trace_value = min(self.trace_value, max_trace)


class EligibilityTraceManager:
    """
    Manages eligibility traces for temporal credit assignment.

    Implements TD(lambda) with support for:
    - Multiple trace types (accumulating, replacing, Dutch)
    - Cross-module credit distribution
    - Importance sampling corrections
    """

    def __init__(
        self,
        config: Optional[TraceConfig] = None,
        random_seed: Optional[int] = None,
    ):
        self.config = config or TraceConfig()
        self._rng = np.random.default_rng(random_seed)

        self._traces: Dict[Tuple[str, str, str], EligibilityTrace] = {}
        self._timestep: int = 0
        self._module_credits: Dict[str, float] = {}

        self._total_credits_distributed = 0.0
        self._total_decay_events = 0

    def mark_eligible(
        self,
        state: Any,
        action: Any,
        module_id: str,
        importance_weight: float = 1.0,
    ) -> None:
        """
        Mark a state-action pair as eligible for credit.

        Args:
            state: Current state
            action: Action taken
            module_id: ID of the module responsible
            importance_weight: Importance sampling weight
        """
        state_key = self._to_key(state)
        action_key = self._to_key(action)
        key = (state_key, action_key, module_id)

        initial_value = self.config.initial_trace
        if self.config.use_importance_sampling:
            initial_value *= importance_weight

        if key in self._traces:
            trace = self._traces[key]
            if self.config.trace_type == TraceType.ACCUMULATING:
                trace.accumulate(initial_value, self.config.max_trace)
            elif self.config.trace_type == TraceType.REPLACING:
                trace.replace(initial_value)
            elif self.config.trace_type == TraceType.DUTCH:
                gamma_lambda = self.config.gamma * self.config.lambda_param
                trace.dutch_update(initial_value, gamma_lambda, self.config.max_trace)
            trace.timestamp = self._timestep
        else:
            self._traces[key] = EligibilityTrace(
                state_key=state_key,
                action_key=action_key,
                module_id=module_id,
                trace_value=initial_value,
                timestamp=self._timestep,
            )

    def decay_traces(self) -> int:
        """
        Decay all eligibility traces.

        Returns number of traces decayed.
        """
        gamma_lambda = self.config.gamma * self.config.lambda_param
        to_remove = []
        decayed = 0

        for key, trace in self._traces.items():
            trace.decay(gamma_lambda)
            decayed += 1

            if trace.trace_value < self.config.min_trace:
                to_remove.append(key)

        for key in to_remove:
            del self._traces[key]

        self._total_decay_events += 1
        self._timestep += 1

        return decayed

    def distribute_credit(
        self,
        reward: float,
        td_error: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Distribute credit to all eligible state-action pairs.

        Args:
            reward: Reward signal to distribute
            td_error: Optional TD error (uses reward if not provided)

        Returns:
            Dict mapping module_id to credit received
        """
        credit_signal = td_error if td_error is not None else reward

        module_credits: Dict[str, float] = {}
        total_trace = sum(t.trace_value for t in self._traces.values())

        if total_trace < 1e-8:
            return module_credits

        for trace in self._traces.values():
            normalized_trace = trace.trace_value / total_trace
            credit = credit_signal * normalized_trace

            if trace.module_id not in module_credits:
                module_credits[trace.module_id] = 0.0
            module_credits[trace.module_id] += credit

        for module_id, credit in module_credits.items():
            if module_id not in self._module_credits:
                self._module_credits[module_id] = 0.0
            self._module_credits[module_id] += credit

        self._total_credits_distributed += abs(credit_signal)

        return module_credits

    def get_trace(
        self,
        state: Any,
        action: Any,
        module_id: str,
    ) -> Optional[EligibilityTrace]:
        """Get a specific trace."""
        key = (self._to_key(state), self._to_key(action), module_id)
        return self._traces.get(key)

    def get_module_traces(self, module_id: str) -> List[EligibilityTrace]:
        """Get all traces for a module."""
        return [t for t in self._traces.values() if t.module_id == module_id]

    def get_total_trace(self, module_id: Optional[str] = None) -> float:
        """Get total trace value, optionally filtered by module."""
        if module_id is None:
            return sum(t.trace_value for t in self._traces.values())
        return sum(t.trace_value for t in self._traces.values() if t.module_id == module_id)

    def get_cumulative_credits(self) -> Dict[str, float]:
        """Get cumulative credits per module."""
        return dict(self._module_credits)

    def reset_traces(self) -> None:
        """Reset all traces."""
        self._traces.clear()

    def reset_credits(self) -> None:
        """Reset cumulative credit tracking."""
        self._module_credits.clear()

    def prune_old_traces(self, max_age: int) -> int:
        """
        Remove traces older than max_age timesteps.

        Returns number of traces removed.
        """
        current = self._timestep
        to_remove = [
            key for key, trace in self._traces.items() if current - trace.timestamp > max_age
        ]
        for key in to_remove:
            del self._traces[key]
        return len(to_remove)

    def set_lambda(self, lambda_param: float) -> None:
        """Dynamically adjust lambda parameter."""
        self.config.lambda_param = max(0.0, min(1.0, lambda_param))

    def _to_key(self, obj: Any) -> str:
        """Convert object to hashable key."""
        if isinstance(obj, np.ndarray):
            return str(hash(obj.tobytes()))
        return str(hash(str(obj)))

    def statistics(self) -> Dict[str, Any]:
        """Get manager statistics."""
        module_counts = {}
        for trace in self._traces.values():
            module_counts[trace.module_id] = module_counts.get(trace.module_id, 0) + 1

        return {
            "active_traces": len(self._traces),
            "total_trace_value": self.get_total_trace(),
            "modules_with_traces": len(module_counts),
            "traces_per_module": module_counts,
            "total_credits_distributed": self._total_credits_distributed,
            "total_decay_events": self._total_decay_events,
            "timestep": self._timestep,
            "cumulative_credits": self._module_credits,
        }
