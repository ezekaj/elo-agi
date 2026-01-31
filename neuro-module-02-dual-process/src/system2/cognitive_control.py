"""
Cognitive Control - Anterior Cingulate Cortex Simulation

Implements conflict detection, error monitoring, and response inhibition.
This is the "executive" that decides when System 2 needs to override System 1.

Key properties:
- Detects conflicting response tendencies
- Generates error signals for learning
- Can inhibit automatic responses
- Allocates attention resources
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum


class ConflictLevel(Enum):
    """Degree of conflict detected"""
    NONE = 0
    LOW = 1
    MODERATE = 2
    HIGH = 3
    SEVERE = 4


@dataclass
class Response:
    """A potential response/action"""
    id: str
    activation: float
    source: str  # "system1", "system2", "habit", etc.
    content: Any = None


@dataclass
class ConflictSignal:
    """Signal indicating response conflict"""
    level: ConflictLevel
    conflicting_responses: List[Response]
    recommended_action: str  # "proceed", "engage_s2", "inhibit", "abort"
    conflict_energy: float  # Raw conflict magnitude


@dataclass
class ErrorSignal:
    """Signal indicating prediction error or mistake"""
    expected: Any
    actual: Any
    magnitude: float
    error_type: str  # "prediction", "outcome", "action"


class CognitiveControl:
    """
    Conflict detection and response monitoring.

    Simulates anterior cingulate cortex:
    - Monitors for response conflict
    - Detects errors and mismatches
    - Triggers System 2 engagement
    - Inhibits inappropriate responses
    """

    def __init__(self,
                 conflict_threshold: float = 0.4,
                 error_threshold: float = 0.3,
                 inhibition_strength: float = 0.6):
        self.conflict_threshold = conflict_threshold
        self.error_threshold = error_threshold
        self.inhibition_strength = inhibition_strength

        # Track recent errors for learning
        self.error_history: List[ErrorSignal] = []
        self.conflict_history: List[ConflictSignal] = []

        # Attention allocation
        self.attention_budget = 1.0
        self.attention_allocation: Dict[str, float] = {}

    def detect_conflict(self, responses: List[Response]) -> ConflictSignal:
        """
        Detect if multiple responses are competing.

        This is the core ACC function - monitoring for situations
        where multiple response tendencies are simultaneously active.
        """
        if len(responses) <= 1:
            return ConflictSignal(
                level=ConflictLevel.NONE,
                conflicting_responses=[],
                recommended_action="proceed",
                conflict_energy=0.0
            )

        # Sort by activation
        sorted_responses = sorted(responses, key=lambda r: r.activation, reverse=True)

        # Compute conflict energy using Hopfield-style energy
        # High energy when multiple responses have similar high activation
        conflict_energy = 0.0
        for i, r1 in enumerate(sorted_responses):
            for r2 in sorted_responses[i + 1:]:
                # Conflict is highest when both are similarly activated
                energy = r1.activation * r2.activation
                conflict_energy += energy

        # Normalize by number of pairs
        num_pairs = len(responses) * (len(responses) - 1) / 2
        if num_pairs > 0:
            conflict_energy /= num_pairs

        # Determine conflict level
        if conflict_energy < 0.1:
            level = ConflictLevel.NONE
        elif conflict_energy < 0.3:
            level = ConflictLevel.LOW
        elif conflict_energy < 0.5:
            level = ConflictLevel.MODERATE
        elif conflict_energy < 0.7:
            level = ConflictLevel.HIGH
        else:
            level = ConflictLevel.SEVERE

        # Determine recommended action
        if level == ConflictLevel.NONE:
            action = "proceed"
        elif level == ConflictLevel.LOW:
            action = "proceed"  # Minor conflict, let System 1 handle
        elif level == ConflictLevel.MODERATE:
            action = "engage_s2"  # Get System 2 involved
        elif level == ConflictLevel.HIGH:
            action = "engage_s2"
        else:
            action = "inhibit"  # Too much conflict, need to stop and think

        signal = ConflictSignal(
            level=level,
            conflicting_responses=sorted_responses[:3],  # Top conflicting
            recommended_action=action,
            conflict_energy=conflict_energy
        )

        self.conflict_history.append(signal)
        return signal

    def error_signal(self,
                     expected: Any,
                     actual: Any,
                     error_type: str = "prediction") -> ErrorSignal:
        """
        Generate error signal when outcome doesn't match expectation.

        This drives learning - errors indicate model needs updating.
        """
        # Compute error magnitude
        if isinstance(expected, np.ndarray) and isinstance(actual, np.ndarray):
            magnitude = float(np.linalg.norm(expected - actual))
        elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            magnitude = abs(expected - actual)
        else:
            # Binary match/mismatch
            magnitude = 0.0 if expected == actual else 1.0

        signal = ErrorSignal(
            expected=expected,
            actual=actual,
            magnitude=magnitude,
            error_type=error_type
        )

        if magnitude > self.error_threshold:
            self.error_history.append(signal)

        return signal

    def inhibit(self, response: Response) -> Tuple[bool, float]:
        """
        Attempt to inhibit a response.

        Returns (success, remaining_activation).
        Strong responses are harder to inhibit.
        """
        inhibition_force = self.inhibition_strength

        # Stronger responses need more inhibition
        remaining = response.activation - inhibition_force

        if remaining <= 0:
            return (True, 0.0)
        else:
            return (False, remaining)

    def inhibit_all_except(self,
                           responses: List[Response],
                           keep_id: str) -> List[Response]:
        """
        Inhibit all responses except the specified one.

        This is selective attention - boosting one response while
        suppressing competitors.
        """
        result = []
        for r in responses:
            if r.id == keep_id:
                # Boost the selected response
                result.append(Response(
                    id=r.id,
                    activation=min(1.0, r.activation * 1.3),
                    source=r.source,
                    content=r.content
                ))
            else:
                # Inhibit competitors
                success, remaining = self.inhibit(r)
                if not success:
                    result.append(Response(
                        id=r.id,
                        activation=remaining,
                        source=r.source,
                        content=r.content
                    ))

        return result

    def allocate_attention(self,
                           targets: Dict[str, float]) -> Dict[str, float]:
        """
        Allocate limited attention resources.

        Attention is a limited resource - allocating more to one thing
        means less for others.
        """
        total_requested = sum(targets.values())

        if total_requested <= self.attention_budget:
            self.attention_allocation = targets.copy()
            return self.attention_allocation

        # Need to normalize - can't exceed budget
        scale = self.attention_budget / total_requested
        self.attention_allocation = {k: v * scale for k, v in targets.items()}

        return self.attention_allocation

    def get_attention(self, target: str) -> float:
        """Get current attention allocated to target"""
        return self.attention_allocation.get(target, 0.0)

    def should_engage_system2(self,
                              responses: List[Response],
                              uncertainty: float = 0.0) -> bool:
        """
        Decide if System 2 should be engaged.

        This is the key switching mechanism between Systems 1 and 2.
        """
        conflict = self.detect_conflict(responses)

        # High conflict -> engage S2
        if conflict.level in [ConflictLevel.HIGH, ConflictLevel.SEVERE]:
            return True

        # Moderate conflict with high uncertainty -> engage S2
        if conflict.level == ConflictLevel.MODERATE and uncertainty > 0.5:
            return True

        # Recent errors -> engage S2 (be more careful)
        recent_errors = [e for e in self.error_history[-10:]
                         if e.magnitude > self.error_threshold]
        if len(recent_errors) >= 3:
            return True

        return False

    def get_error_rate(self, window: int = 10) -> float:
        """Get recent error rate"""
        if len(self.error_history) == 0:
            return 0.0

        recent = self.error_history[-window:]
        return sum(1 for e in recent if e.magnitude > self.error_threshold) / len(recent)

    def get_conflict_rate(self, window: int = 10) -> float:
        """Get recent conflict rate"""
        if len(self.conflict_history) == 0:
            return 0.0

        recent = self.conflict_history[-window:]
        return sum(1 for c in recent
                   if c.level in [ConflictLevel.MODERATE, ConflictLevel.HIGH, ConflictLevel.SEVERE]) / len(recent)

    def reset_monitoring(self):
        """Reset error and conflict history"""
        self.error_history.clear()
        self.conflict_history.clear()

    def adaptive_threshold(self):
        """
        Adaptively adjust conflict threshold based on recent history.

        If we're making lots of errors, become more conservative.
        """
        error_rate = self.get_error_rate()

        if error_rate > 0.5:
            # Many errors - lower threshold, engage S2 more
            self.conflict_threshold = max(0.2, self.conflict_threshold - 0.05)
        elif error_rate < 0.1:
            # Few errors - raise threshold, trust S1 more
            self.conflict_threshold = min(0.6, self.conflict_threshold + 0.02)
