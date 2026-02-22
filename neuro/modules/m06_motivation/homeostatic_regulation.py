"""
Homeostatic Regulation

Implements state-dependent valuation where rewards = physiological resources
and value depends on internal state.

Key insight: The value of a reward is not fixed but depends on:
- Current need state (hungry person values food more)
- Satiation level (diminishing returns)
- Resource predictions (anticipatory regulation)

This explains:
- Why the same reward has different values at different times
- Allostasis (predictive regulation, not just reactive)
- Drive reduction vs drive induction theories
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import deque
from enum import Enum


class NeedType(Enum):
    """Types of homeostatic needs"""

    ENERGY = "energy"  # Food, glucose
    HYDRATION = "hydration"  # Water
    TEMPERATURE = "temperature"  # Thermal regulation
    SOCIAL = "social"  # Social connection
    NOVELTY = "novelty"  # Stimulation/information
    REST = "rest"  # Sleep, recovery
    SAFETY = "safety"  # Security, predictability


@dataclass
class HomeostaticSetpoint:
    """A homeostatic setpoint for a need"""

    optimal_value: float = 0.7  # Target level
    tolerance_low: float = 0.3  # Below this triggers strong drive
    tolerance_high: float = 0.9  # Above this is satiated
    depletion_rate: float = 0.01  # Natural decay toward 0
    importance: float = 1.0  # Priority weight


@dataclass
class NeedState:
    """Current state of a single need"""

    current_level: float
    setpoint: HomeostaticSetpoint
    drive_strength: float
    time_since_satiation: float
    trend: float  # Positive = improving, negative = depleting


class HomeostaticState:
    """Represents internal state across multiple needs.

    Manages multiple homeostatic variables and computes
    overall internal state.
    """

    def __init__(self):
        # Define setpoints for each need
        self.setpoints: Dict[NeedType, HomeostaticSetpoint] = {
            NeedType.ENERGY: HomeostaticSetpoint(0.7, 0.3, 0.95, 0.005, 1.0),
            NeedType.HYDRATION: HomeostaticSetpoint(0.8, 0.4, 0.95, 0.008, 1.2),
            NeedType.TEMPERATURE: HomeostaticSetpoint(0.5, 0.3, 0.7, 0.0, 0.8),
            NeedType.SOCIAL: HomeostaticSetpoint(0.6, 0.2, 0.9, 0.003, 0.7),
            NeedType.NOVELTY: HomeostaticSetpoint(0.5, 0.2, 0.8, 0.01, 0.6),
            NeedType.REST: HomeostaticSetpoint(0.7, 0.3, 1.0, 0.004, 0.9),
            NeedType.SAFETY: HomeostaticSetpoint(0.8, 0.4, 0.95, 0.002, 1.1),
        }

        # Current levels
        self.levels: Dict[NeedType, float] = {
            need: sp.optimal_value for need, sp in self.setpoints.items()
        }

        # Track time since last satiation
        self.satiation_times: Dict[NeedType, float] = {need: 0.0 for need in NeedType}

        # History for trend analysis
        self.level_history: Dict[NeedType, deque] = {need: deque(maxlen=100) for need in NeedType}

        # Current time
        self.current_time = 0.0

    def update(self, dt: float = 1.0) -> None:
        """Update all need states over time."""
        self.current_time += dt

        for need in NeedType:
            setpoint = self.setpoints[need]
            current = self.levels[need]

            # Natural depletion
            depleted = current - setpoint.depletion_rate * dt
            self.levels[need] = np.clip(depleted, 0.0, 1.0)

            # Track history
            self.level_history[need].append(self.levels[need])

            # Update satiation time
            if self.levels[need] >= setpoint.tolerance_high:
                self.satiation_times[need] = self.current_time

    def consume_resource(self, need: NeedType, amount: float) -> float:
        """Consume a resource to satisfy a need.

        Args:
            need: The need being addressed
            amount: Resource amount (0-1 scale)

        Returns:
            Actual satisfaction gained (accounting for satiation)
        """
        setpoint = self.setpoints[need]
        current = self.levels[need]

        # Diminishing returns near satiation
        room = setpoint.tolerance_high - current
        effective_amount = min(amount, room)

        # Satisfaction is proportional to need level
        drive_before = self.get_drive(need)
        self.levels[need] = np.clip(current + effective_amount, 0, 1)
        drive_after = self.get_drive(need)

        # Update satiation time if satisfied
        if self.levels[need] >= setpoint.tolerance_high:
            self.satiation_times[need] = self.current_time

        # Satisfaction is drive reduction
        return max(0, drive_before - drive_after)

    def get_drive(self, need: NeedType) -> float:
        """Get current drive strength for a need.

        Drive = how urgently this need wants to be satisfied.
        """
        setpoint = self.setpoints[need]
        level = self.levels[need]

        if level >= setpoint.optimal_value:
            # Above optimal - no drive
            return 0.0
        elif level <= setpoint.tolerance_low:
            # Below tolerance - maximum drive
            return setpoint.importance * 1.0
        else:
            # Linear interpolation in middle zone
            range_size = setpoint.optimal_value - setpoint.tolerance_low
            deficit = setpoint.optimal_value - level
            drive = deficit / range_size
            return setpoint.importance * drive

    def get_all_drives(self) -> Dict[NeedType, float]:
        """Get drive strengths for all needs."""
        return {need: self.get_drive(need) for need in NeedType}

    def get_dominant_drive(self) -> Tuple[NeedType, float]:
        """Get the most urgent need."""
        drives = self.get_all_drives()
        dominant = max(drives.items(), key=lambda x: x[1])
        return dominant

    def get_need_state(self, need: NeedType) -> NeedState:
        """Get complete state for a single need."""
        setpoint = self.setpoints[need]
        level = self.levels[need]
        history = list(self.level_history[need])

        # Compute trend
        if len(history) >= 10:
            recent = np.mean(history[-5:])
            earlier = np.mean(history[-10:-5])
            trend = recent - earlier
        else:
            trend = 0.0

        return NeedState(
            current_level=level,
            setpoint=setpoint,
            drive_strength=self.get_drive(need),
            time_since_satiation=self.current_time - self.satiation_times[need],
            trend=trend,
        )

    def get_overall_wellbeing(self) -> float:
        """Get overall wellbeing (inverse of total drive)."""
        total_drive = sum(self.get_all_drives().values())
        max_possible = sum(sp.importance for sp in self.setpoints.values())
        return 1.0 - (total_drive / max_possible)

    def set_level(self, need: NeedType, level: float) -> None:
        """Directly set a need level."""
        self.levels[need] = np.clip(level, 0, 1)


class NeedBasedValuation:
    """Computes reward value based on internal need state.

    The value of a resource depends on:
    - Current need level
    - Resource relevance to need
    - Satiation effects
    """

    def __init__(self, homeostatic_state: HomeostaticState, satiation_rate: float = 0.3):
        self.state = homeostatic_state
        self.satiation_rate = satiation_rate

        # Resource-need mappings (which resources satisfy which needs)
        self.resource_mappings: Dict[str, Dict[NeedType, float]] = {
            "food": {NeedType.ENERGY: 0.8, NeedType.SOCIAL: 0.1},
            "water": {NeedType.HYDRATION: 0.9},
            "shelter": {NeedType.SAFETY: 0.7, NeedType.TEMPERATURE: 0.5},
            "social_contact": {NeedType.SOCIAL: 0.8, NeedType.SAFETY: 0.2},
            "information": {NeedType.NOVELTY: 0.7},
            "rest": {NeedType.REST: 0.9, NeedType.ENERGY: 0.2},
        }

        # Consumption history for satiation
        self.consumption_history: Dict[str, deque] = {}

    def compute_value(self, resource_type: str, resource_amount: float = 1.0) -> float:
        """Compute subjective value of a resource.

        Args:
            resource_type: Type of resource (food, water, etc.)
            resource_amount: Amount of resource

        Returns:
            Subjective value based on current needs
        """
        if resource_type not in self.resource_mappings:
            return resource_amount * 0.1  # Unknown resource has low value

        # Get need mappings
        mappings = self.resource_mappings[resource_type]

        total_value = 0.0
        for need, relevance in mappings.items():
            # Value proportional to drive and relevance
            drive = self.state.get_drive(need)
            need_value = drive * relevance * resource_amount

            # Apply satiation effect
            satiation = self._get_satiation(resource_type)
            need_value *= 1 - satiation * self.satiation_rate

            total_value += need_value

        return total_value

    def _get_satiation(self, resource_type: str) -> float:
        """Get satiation level for a resource type."""
        if resource_type not in self.consumption_history:
            return 0.0

        history = self.consumption_history[resource_type]
        if len(history) < 3:
            return 0.0

        # Recent consumption leads to satiation
        recent = list(history)[-10:]
        return min(1.0, np.sum(recent) / 5.0)

    def record_consumption(self, resource_type: str, amount: float) -> None:
        """Record consumption of a resource."""
        if resource_type not in self.consumption_history:
            self.consumption_history[resource_type] = deque(maxlen=50)
        self.consumption_history[resource_type].append(amount)

    def get_most_valuable_resource(self, available: List[str]) -> Tuple[str, float]:
        """Find which available resource has highest value."""
        if not available:
            return "", 0.0

        values = [(r, self.compute_value(r)) for r in available]
        best = max(values, key=lambda x: x[1])
        return best

    def register_resource(self, resource_type: str, need_mappings: Dict[NeedType, float]) -> None:
        """Register a new resource type."""
        self.resource_mappings[resource_type] = need_mappings


class InternalStateTracker:
    """Tracks internal state over time and predicts future needs.

    Implements allostasis - predictive regulation that anticipates
    future needs rather than just reacting to current deficits.
    """

    def __init__(self, homeostatic_state: HomeostaticState, prediction_horizon: float = 100.0):
        self.state = homeostatic_state
        self.prediction_horizon = prediction_horizon

        # History for prediction
        self.state_history: deque = deque(maxlen=1000)

        # Learned patterns (e.g., energy drops after activity)
        self.depletion_patterns: Dict[str, Dict[NeedType, float]] = {
            "exercise": {NeedType.ENERGY: 0.1, NeedType.HYDRATION: 0.08},
            "social_interaction": {NeedType.SOCIAL: -0.05, NeedType.ENERGY: 0.02},
            "work": {NeedType.ENERGY: 0.05, NeedType.REST: 0.03, NeedType.NOVELTY: -0.01},
        }

        # Predicted future state
        self.predictions: Dict[NeedType, float] = {}

    def record_state(self) -> None:
        """Record current state for pattern learning."""
        snapshot = {
            "time": self.state.current_time,
            "levels": dict(self.state.levels),
            "drives": self.state.get_all_drives(),
        }
        self.state_history.append(snapshot)

    def predict_future_state(
        self, planned_activities: List[str], time_horizon: float
    ) -> Dict[NeedType, float]:
        """Predict future need levels based on planned activities.

        Args:
            planned_activities: List of planned activities
            time_horizon: How far ahead to predict

        Returns:
            Predicted need levels
        """
        # Start from current state
        predicted = dict(self.state.levels)

        # Apply natural depletion
        for need in NeedType:
            rate = self.state.setpoints[need].depletion_rate
            predicted[need] -= rate * time_horizon

        # Apply activity effects
        for activity in planned_activities:
            if activity in self.depletion_patterns:
                for need, effect in self.depletion_patterns[activity].items():
                    predicted[need] -= effect

        # Clip to valid range
        predicted = {n: np.clip(v, 0, 1) for n, v in predicted.items()}

        self.predictions = predicted
        return predicted

    def get_anticipatory_drive(self, need: NeedType) -> float:
        """Get drive based on predicted future state (allostasis).

        This implements anticipatory regulation - acting before
        deficit actually occurs.
        """
        if need not in self.predictions:
            return self.state.get_drive(need)

        current_drive = self.state.get_drive(need)

        # Compute predicted drive
        setpoint = self.state.setpoints[need]
        predicted_level = self.predictions[need]

        if predicted_level <= setpoint.tolerance_low:
            predicted_drive = setpoint.importance * 1.0
        elif predicted_level >= setpoint.optimal_value:
            predicted_drive = 0.0
        else:
            range_size = setpoint.optimal_value - setpoint.tolerance_low
            deficit = setpoint.optimal_value - predicted_level
            predicted_drive = setpoint.importance * (deficit / range_size)

        # Anticipatory drive is weighted average
        anticipatory_weight = 0.3
        return (1 - anticipatory_weight) * current_drive + anticipatory_weight * predicted_drive

    def identify_upcoming_needs(self, threshold: float = 0.5) -> List[NeedType]:
        """Identify needs that will become urgent."""
        if not self.predictions:
            return []

        upcoming = []
        for need in NeedType:
            current_drive = self.state.get_drive(need)
            predicted_drive = self.get_anticipatory_drive(need)

            # Flag if drive will increase significantly
            if predicted_drive > current_drive + threshold:
                upcoming.append(need)

        return upcoming

    def learn_pattern(
        self,
        activity: str,
        before_levels: Dict[NeedType, float],
        after_levels: Dict[NeedType, float],
    ) -> None:
        """Learn how an activity affects need levels."""
        effects = {}
        for need in NeedType:
            if need in before_levels and need in after_levels:
                effect = before_levels[need] - after_levels[need]
                effects[need] = effect

        # Update pattern with momentum
        if activity in self.depletion_patterns:
            for need, effect in effects.items():
                old = self.depletion_patterns[activity].get(need, 0)
                self.depletion_patterns[activity][need] = 0.8 * old + 0.2 * effect
        else:
            self.depletion_patterns[activity] = effects

    def get_resource_priority(self) -> List[Tuple[NeedType, float]]:
        """Get prioritized list of needs based on anticipatory drives."""
        priorities = []
        for need in NeedType:
            anticipatory = self.get_anticipatory_drive(need)
            priorities.append((need, anticipatory))

        return sorted(priorities, key=lambda x: x[1], reverse=True)
