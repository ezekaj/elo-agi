"""
Motivational States: Goal-directed action drives.

Based on research distinguishing motivational from emotional states:
- Motivational: Goal-directed actions to obtain rewards/avoid punishments (BEFORE outcome)
- Emotional: Reactions when rewards ARE or ARE NOT received (AFTER outcome)

Also implements Berridge's wanting/liking distinction.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import numpy as np


class DriveDirection(Enum):
    """Direction of motivational drive."""

    APPROACH = "approach"
    AVOID = "avoid"


@dataclass
class Drive:
    """
    Single motivational drive.

    Represents a goal-directed tendency to obtain something (approach)
    or avoid something (avoid).
    """

    target: str
    intensity: float  # 0-1
    direction: DriveDirection
    decay_rate: float = 0.01  # How fast drive fades without reinforcement
    created_at: float = 0.0
    last_reinforced: float = 0.0

    def decay(self, time_steps: int = 1):
        """Drive fades over time without reinforcement."""
        self.intensity *= (1 - self.decay_rate) ** time_steps
        self.intensity = max(0.0, self.intensity)

    def reinforce(self, amount: float = 0.1):
        """Strengthen drive through reinforcement."""
        self.intensity = min(1.0, self.intensity + amount)
        self.last_reinforced = self.created_at  # Would be current time


@dataclass
class IncentiveSignal:
    """
    Incentive salience signal.

    Distinguishes 'wanting' (motivational pull) from 'liking' (hedonic pleasure).
    These can be dissociated (e.g., addiction: high wanting, low liking).
    """

    wanting: float  # Motivational pull (dopaminergic)
    liking: float  # Hedonic pleasure (opioid)
    stimulus_id: str


class MotivationalSystem:
    """
    Manages goal-directed action tendencies.

    Key functions:
    - Track active drives (what we want to pursue/avoid)
    - Compute drive strength for goals
    - Resolve conflicts between approach and avoidance
    """

    def __init__(self, base_arousal: float = 0.5):
        self.active_drives: List[Drive] = []
        self.reward_history: Dict[str, List[float]] = {}
        self.punishment_history: Dict[str, List[float]] = {}
        self.base_arousal = base_arousal
        self.time_step = 0

    def add_drive(self, target: str, intensity: float, direction: DriveDirection) -> Drive:
        """Add a new motivational drive."""
        drive = Drive(
            target=target, intensity=intensity, direction=direction, created_at=self.time_step
        )
        self.active_drives.append(drive)
        return drive

    def get_drive(self, target: str) -> Optional[Drive]:
        """Get drive for a specific target."""
        for drive in self.active_drives:
            if drive.target == target:
                return drive
        return None

    def compute_drive(self, goal: str, current_state: Dict[str, Any]) -> float:
        """
        Compute overall motivation to pursue a goal.

        Combines:
        - Existing drives for this goal
        - Past reward history
        - Current state (deprivation, satiation)
        """
        # Check for existing drive
        existing_drive = self.get_drive(goal)
        base_motivation = existing_drive.intensity if existing_drive else 0.0

        # Reward history influences motivation
        history = self.reward_history.get(goal, [])
        if history:
            avg_reward = np.mean(history)
            recency_weight = np.exp(-len(history) * 0.1)  # Recent matters more
            history_motivation = avg_reward * recency_weight
        else:
            history_motivation = 0.0

        # Current state modulation
        deprivation = current_state.get("deprivation", 0.5)
        satiation = current_state.get("satiation", 0.0)

        state_modulation = 1.0 + deprivation - satiation

        total = (base_motivation + history_motivation) * state_modulation * self.base_arousal
        return np.clip(total, 0.0, 1.0)

    def approach_tendency(self, reward_magnitude: float, probability: float = 1.0) -> float:
        """
        Compute strength of approach motivation.

        Higher reward and probability = stronger approach.
        """
        expected_value = reward_magnitude * probability

        # Nonlinear response (diminishing returns for very high rewards)
        approach = 1 - np.exp(-expected_value * 2)

        return approach * self.base_arousal

    def avoidance_tendency(self, punishment_magnitude: float, probability: float = 1.0) -> float:
        """
        Compute strength of avoidance motivation.

        Higher punishment and probability = stronger avoidance.
        """
        expected_cost = punishment_magnitude * probability

        # Avoidance often stronger than approach for same magnitude (loss aversion)
        loss_aversion_factor = 1.5
        avoidance = 1 - np.exp(-expected_cost * 2 * loss_aversion_factor)

        return avoidance * self.base_arousal

    def resolve_conflict(
        self, approach_strength: float, avoidance_strength: float
    ) -> Dict[str, Any]:
        """
        Resolve approach-avoidance conflict.

        Returns decision and confidence.
        Classic approach-avoidance gradients: avoidance steeper near goal.
        """
        # Net motivation
        net = approach_strength - avoidance_strength

        # Conflict level (both strong = high conflict)
        conflict = min(approach_strength, avoidance_strength)

        # Decision
        if net > 0.1:
            decision = "approach"
            confidence = net / (approach_strength + 0.01)
        elif net < -0.1:
            decision = "avoid"
            confidence = abs(net) / (avoidance_strength + 0.01)
        else:
            decision = "freeze"  # Ambivalence/conflict
            confidence = 0.5

        # High conflict reduces confidence
        confidence *= 1 - conflict * 0.5

        return {
            "decision": decision,
            "confidence": confidence,
            "approach_strength": approach_strength,
            "avoidance_strength": avoidance_strength,
            "conflict_level": conflict,
            "net_motivation": net,
        }

    def record_reward(self, target: str, magnitude: float):
        """Record a reward outcome for a target."""
        if target not in self.reward_history:
            self.reward_history[target] = []
        self.reward_history[target].append(magnitude)

        # Reinforce drive if exists
        drive = self.get_drive(target)
        if drive:
            drive.reinforce(magnitude * 0.2)

    def record_punishment(self, target: str, magnitude: float):
        """Record a punishment outcome for a target."""
        if target not in self.punishment_history:
            self.punishment_history[target] = []
        self.punishment_history[target].append(magnitude)

    def step(self):
        """Advance time and decay drives."""
        self.time_step += 1
        for drive in self.active_drives:
            drive.decay()

        # Remove extinguished drives
        self.active_drives = [d for d in self.active_drives if d.intensity > 0.01]


class IncentiveSalience:
    """
    Implements Berridge's 'wanting' vs 'liking' distinction.

    Key insight: These are neurologically separable:
    - Wanting (dopaminergic): motivational pull, drives pursuit
    - Liking (opioid): hedonic pleasure, actual enjoyment

    Addiction example: High wanting (compulsion to pursue) + Low liking (no pleasure)
    """

    def __init__(self):
        self.wanting_associations: Dict[str, float] = {}
        self.liking_associations: Dict[str, float] = {}

        # Sensitization (increases wanting without liking)
        self.sensitization_level: Dict[str, float] = {}

    def wanting(self, stimulus_id: str) -> float:
        """
        Compute motivational pull ('wanting') for a stimulus.

        Dopaminergic system - can be sensitized (addiction).
        """
        base_wanting = self.wanting_associations.get(stimulus_id, 0.0)
        sensitization = self.sensitization_level.get(stimulus_id, 1.0)

        return min(1.0, base_wanting * sensitization)

    def liking(self, stimulus_id: str) -> float:
        """
        Compute hedonic pleasure ('liking') for a stimulus.

        Opioid system - actual enjoyment/pleasure.
        """
        return self.liking_associations.get(stimulus_id, 0.0)

    def dissociate(self, stimulus_id: str) -> IncentiveSignal:
        """
        Get both wanting and liking for a stimulus.

        Highlights their potential dissociation.
        """
        return IncentiveSignal(
            wanting=self.wanting(stimulus_id),
            liking=self.liking(stimulus_id),
            stimulus_id=stimulus_id,
        )

    def learn_association(
        self, stimulus_id: str, wanting_update: float = 0.0, liking_update: float = 0.0
    ):
        """Learn wanting and/or liking association."""
        if wanting_update != 0:
            current = self.wanting_associations.get(stimulus_id, 0.0)
            self.wanting_associations[stimulus_id] = np.clip(current + wanting_update, 0.0, 1.0)

        if liking_update != 0:
            current = self.liking_associations.get(stimulus_id, 0.0)
            self.liking_associations[stimulus_id] = np.clip(current + liking_update, 0.0, 1.0)

    def sensitize(self, stimulus_id: str, amount: float = 0.1):
        """
        Increase sensitization (wanting increases without liking).

        Models addiction: repeated exposure increases wanting
        even as liking remains same or decreases.
        """
        current = self.sensitization_level.get(stimulus_id, 1.0)
        self.sensitization_level[stimulus_id] = min(3.0, current + amount)

    def desensitize(self, stimulus_id: str, amount: float = 0.05):
        """Reduce sensitization (recovery)."""
        current = self.sensitization_level.get(stimulus_id, 1.0)
        self.sensitization_level[stimulus_id] = max(1.0, current - amount)

    def tolerance(self, stimulus_id: str, amount: float = 0.05):
        """
        Develop tolerance (liking decreases with repeated exposure).

        Common in addiction: need more to get same pleasure.
        """
        current = self.liking_associations.get(stimulus_id, 0.0)
        self.liking_associations[stimulus_id] = max(0.0, current - amount)

    def is_addictive_pattern(self, stimulus_id: str) -> bool:
        """
        Check if stimulus shows addictive pattern:
        High wanting + Low/declining liking
        """
        wanting = self.wanting(stimulus_id)
        liking_val = self.liking(stimulus_id)
        sensitization = self.sensitization_level.get(stimulus_id, 1.0)

        return wanting > 0.6 and liking_val < 0.4 and sensitization > 1.5

    def create_healthy_desire(self, stimulus_id: str, level: float = 0.5):
        """Create balanced wanting and liking (healthy desire)."""
        self.wanting_associations[stimulus_id] = level
        self.liking_associations[stimulus_id] = level
        self.sensitization_level[stimulus_id] = 1.0

    def create_addiction_pattern(self, stimulus_id: str):
        """
        Create addictive pattern for demonstration:
        High wanting, low liking, high sensitization.
        """
        self.wanting_associations[stimulus_id] = 0.3  # Base wanting
        self.liking_associations[stimulus_id] = 0.2  # Low liking
        self.sensitization_level[stimulus_id] = 2.5  # High sensitization
