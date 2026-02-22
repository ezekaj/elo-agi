"""
Procedural Memory: Skills and habits

Based on research showing implicit motor and cognitive skills
stored in cerebellum and basal ganglia.

Brain regions: Cerebellum, Basal ganglia
"""

import time
from typing import Optional, Any, List, Dict, Callable, Tuple
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Pattern:
    """Trigger pattern for procedure activation"""

    features: Dict[str, Any] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)

    def match(self, stimulus: Dict[str, Any]) -> float:
        """Compute match score against stimulus"""
        if not self.features:
            return 0.0

        total_weight = sum(self.weights.values()) or len(self.features)
        score = 0.0

        for key, value in self.features.items():
            if key in stimulus:
                weight = self.weights.get(key, 1.0)
                if stimulus[key] == value:
                    score += weight
                elif isinstance(value, (int, float)) and isinstance(stimulus[key], (int, float)):
                    # Partial match for numeric values
                    diff = abs(stimulus[key] - value) / (abs(value) + 1e-6)
                    score += weight * max(0, 1 - diff)

        return score / total_weight


@dataclass
class Action:
    """Single action in a procedure"""

    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    duration: float = 0.1  # seconds

    def execute(self) -> Any:
        """Execute the action (returns action description)"""
        return {"action": self.name, "params": self.parameters}


@dataclass
class Procedure:
    """Skill or habit representation"""

    name: str
    trigger: Pattern
    action_sequence: List[Action] = field(default_factory=list)
    strength: float = 0.1  # Automaticity level 0-1
    error_history: List[float] = field(default_factory=list)
    execution_count: int = 0
    last_execution: Optional[float] = None

    def is_automatic(self, threshold: float = 0.7) -> bool:
        """Check if procedure is fully proceduralized"""
        return self.strength >= threshold


class ProceduralMemory:
    """
    Cerebellum & basal ganglia - skills & habits

    Implements procedural memory with:
    - Trigger-based activation
    - Automaticity through practice
    - Competition between procedures
    """

    def __init__(self):
        self._procedures: Dict[str, Procedure] = {}
        self._active_procedure: Optional[Procedure] = None
        self._time_fn = time.time

    def encode(
        self, name: str, trigger: Pattern, actions: List[Action], initial_strength: float = 0.1
    ) -> Procedure:
        """
        Create new procedure (initially weak).

        Args:
            name: Procedure name
            trigger: Activation pattern
            actions: Sequence of actions
            initial_strength: Starting automaticity

        Returns:
            The created Procedure
        """
        procedure = Procedure(
            name=name, trigger=trigger, action_sequence=actions, strength=initial_strength
        )
        self._procedures[name] = procedure
        return procedure

    def encode_simple(
        self, name: str, trigger_features: Dict[str, Any], action_names: List[str]
    ) -> Procedure:
        """
        Simplified procedure creation.

        Args:
            name: Procedure name
            trigger_features: Features that trigger the procedure
            action_names: List of action names

        Returns:
            The created Procedure
        """
        trigger = Pattern(features=trigger_features)
        actions = [Action(name=a) for a in action_names]
        return self.encode(name, trigger, actions)

    def execute(self, stimulus: Dict[str, Any]) -> Optional[List[Any]]:
        """
        Find and run matching procedure.

        Args:
            stimulus: Current stimulus features

        Returns:
            List of action results, or None if no procedure matches
        """
        # Find matching procedures
        matches = self.get_matching(stimulus)

        if not matches:
            return None

        # Compete and select winner
        winner = self.compete(matches)

        if not winner:
            return None

        # Execute the procedure
        self._active_procedure = winner
        winner.execution_count += 1
        winner.last_execution = self._time_fn()

        results = []
        for action in winner.action_sequence:
            results.append(action.execute())

        self._active_procedure = None
        return results

    def get_matching(
        self, stimulus: Dict[str, Any], threshold: float = 0.5
    ) -> List[Tuple[Procedure, float]]:
        """
        Find all procedures that could fire.

        Args:
            stimulus: Current stimulus features
            threshold: Minimum match score

        Returns:
            List of (procedure, match_score) tuples
        """
        matches = []

        for procedure in self._procedures.values():
            score = procedure.trigger.match(stimulus)
            if score >= threshold:
                matches.append((procedure, score))

        return matches

    def compete(self, candidates: List[Tuple[Procedure, float]]) -> Optional[Procedure]:
        """
        Resolve conflicts between procedures.

        Selection based on:
        - Match score
        - Automaticity (strength)
        - Recent success

        Args:
            candidates: List of (procedure, match_score)

        Returns:
            Winning procedure
        """
        if not candidates:
            return None

        if len(candidates) == 1:
            return candidates[0][0]

        # Score each candidate
        scored = []
        for procedure, match_score in candidates:
            # Combined score: match * strength * (1 - avg_error)
            avg_error = np.mean(procedure.error_history[-10:]) if procedure.error_history else 0.5
            combined = match_score * procedure.strength * (1 - avg_error)
            scored.append((procedure, combined))

        # Select highest scorer
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]

    def strengthen(self, procedure: Procedure, success: bool = True) -> None:
        """
        Increase automaticity through practice.

        Args:
            procedure: Procedure to strengthen
            success: Whether execution was successful
        """
        if success:
            # Successful execution increases strength
            # Diminishing returns as strength approaches 1
            increment = 0.1 * (1 - procedure.strength)
            procedure.strength = min(1.0, procedure.strength + increment)
            procedure.error_history.append(0.0)
        else:
            procedure.error_history.append(1.0)

    def weaken(self, procedure: Procedure, amount: float = 0.1) -> None:
        """
        Extinction from non-use or errors.

        Args:
            procedure: Procedure to weaken
            amount: Amount to reduce strength
        """
        procedure.strength = max(0.0, procedure.strength - amount)

    def is_automatic(self, name: str, threshold: float = 0.7) -> bool:
        """
        Check if procedure is fully proceduralized.

        Args:
            name: Procedure name
            threshold: Automaticity threshold

        Returns:
            True if procedure strength exceeds threshold
        """
        procedure = self._procedures.get(name)
        return procedure.is_automatic(threshold) if procedure else False

    def get_procedure(self, name: str) -> Optional[Procedure]:
        """Get procedure by name"""
        return self._procedures.get(name)

    def list_procedures(self) -> List[str]:
        """List all procedure names"""
        return list(self._procedures.keys())

    def get_automatic_procedures(self, threshold: float = 0.7) -> List[Procedure]:
        """Get all fully automatic procedures"""
        return [p for p in self._procedures.values() if p.is_automatic(threshold)]

    def decay_all(self, amount: float = 0.01) -> None:
        """Apply decay to all procedures (simulates forgetting from non-use)"""
        for procedure in self._procedures.values():
            procedure.strength = max(0.0, procedure.strength - amount)

    def remove(self, name: str) -> bool:
        """Remove a procedure"""
        if name in self._procedures:
            del self._procedures[name]
            return True
        return False

    def __len__(self) -> int:
        return len(self._procedures)

    def __contains__(self, name: str) -> bool:
        return name in self._procedures

    def set_time_function(self, time_fn) -> None:
        """Set custom time function for simulation"""
        self._time_fn = time_fn
