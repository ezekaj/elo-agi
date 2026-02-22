"""
Effort Valuation

Implements the complex relationship between effort and value:
- Effort is generally aversive (cost)
- But effort can paradoxically ADD value under certain conditions
- Motivation transforms how effort is experienced

Key insights from research:
- Mental effort is aversive but can paradoxically add value
- Under deadline pressure, effort becomes valued (not costly)
- "Effort justification" - we value things we worked hard for
- Dopamine modulates effort tolerance
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
from enum import Enum


class EffortType(Enum):
    """Types of effort"""

    PHYSICAL = "physical"  # Bodily exertion
    COGNITIVE = "cognitive"  # Mental processing
    EMOTIONAL = "emotional"  # Emotional regulation
    SOCIAL = "social"  # Social interaction effort
    ATTENTIONAL = "attentional"  # Sustained focus


@dataclass
class EffortProfile:
    """Profile of effort requirements for a task"""

    physical: float = 0.0
    cognitive: float = 0.0
    emotional: float = 0.0
    social: float = 0.0
    attentional: float = 0.0

    def total(self) -> float:
        return self.physical + self.cognitive + self.emotional + self.social + self.attentional


@dataclass
class EffortOutcome:
    """Outcome of an effortful action"""

    effort_expended: EffortProfile
    reward_gained: float
    subjective_cost: float
    value_added: float  # Paradoxical effort value


class EffortCostModel:
    """Models the cost of effort.

    Effort cost is not fixed - it depends on:
    - Current fatigue/resources
    - Dopamine level
    - Context (deadline, goals)
    - History (habituation)
    """

    def __init__(
        self,
        base_cost_per_unit: float = 0.1,
        fatigue_sensitivity: float = 0.5,
        dopamine_sensitivity: float = 0.3,
    ):
        self.base_cost = base_cost_per_unit
        self.fatigue_sensitivity = fatigue_sensitivity
        self.dopamine_sensitivity = dopamine_sensitivity

        # Resource pools (deplete with effort)
        self.resources: Dict[EffortType, float] = {et: 1.0 for et in EffortType}

        # Recovery rates
        self.recovery_rates: Dict[EffortType, float] = {
            EffortType.PHYSICAL: 0.02,
            EffortType.COGNITIVE: 0.01,
            EffortType.EMOTIONAL: 0.008,
            EffortType.SOCIAL: 0.015,
            EffortType.ATTENTIONAL: 0.012,
        }

        # Current dopamine level (modulates cost)
        self.dopamine_level = 0.5

        # Effort history
        self.effort_history: deque = deque(maxlen=200)

    def compute_cost(self, effort_profile: EffortProfile, context: Optional[Dict] = None) -> float:
        """Compute subjective cost of effort.

        Args:
            effort_profile: Required effort amounts
            context: Contextual factors (deadline, importance, etc.)

        Returns:
            Subjective effort cost
        """
        total_cost = 0.0

        effort_dict = {
            EffortType.PHYSICAL: effort_profile.physical,
            EffortType.COGNITIVE: effort_profile.cognitive,
            EffortType.EMOTIONAL: effort_profile.emotional,
            EffortType.SOCIAL: effort_profile.social,
            EffortType.ATTENTIONAL: effort_profile.attentional,
        }

        for effort_type, amount in effort_dict.items():
            if amount <= 0:
                continue

            # Base cost
            cost = self.base_cost * amount

            # Fatigue increases cost
            resource = self.resources[effort_type]
            fatigue_factor = 1 + self.fatigue_sensitivity * (1 - resource)
            cost *= fatigue_factor

            # Dopamine reduces cost
            dopamine_factor = 1 - self.dopamine_sensitivity * (self.dopamine_level - 0.5)
            cost *= max(0.5, dopamine_factor)

            total_cost += cost

        # Context modulation
        if context:
            total_cost = self._apply_context(total_cost, context)

        return total_cost

    def _apply_context(self, cost: float, context: Dict) -> float:
        """Apply contextual modulations to effort cost."""
        modified_cost = cost

        # Deadline pressure reduces perceived cost
        if context.get("deadline_pressure", 0) > 0.5:
            modified_cost *= 0.7

        # High importance reduces cost
        if context.get("importance", 0) > 0.7:
            modified_cost *= 0.8

        # Autonomy reduces cost
        if context.get("autonomy", 0) > 0.5:
            modified_cost *= 0.85

        # Boredom increases cost
        if context.get("boredom", 0) > 0.5:
            modified_cost *= 1.3

        return modified_cost

    def expend_effort(self, effort_profile: EffortProfile) -> float:
        """Expend effort and deplete resources.

        Returns actual cost incurred.
        """
        cost = self.compute_cost(effort_profile)

        # Deplete resources
        effort_dict = {
            EffortType.PHYSICAL: effort_profile.physical,
            EffortType.COGNITIVE: effort_profile.cognitive,
            EffortType.EMOTIONAL: effort_profile.emotional,
            EffortType.SOCIAL: effort_profile.social,
            EffortType.ATTENTIONAL: effort_profile.attentional,
        }

        for effort_type, amount in effort_dict.items():
            depletion = amount * 0.1
            self.resources[effort_type] = max(0, self.resources[effort_type] - depletion)

        self.effort_history.append(effort_profile)
        return cost

    def recover(self, dt: float = 1.0) -> None:
        """Recover resources over time."""
        for effort_type in EffortType:
            rate = self.recovery_rates[effort_type]
            current = self.resources[effort_type]
            self.resources[effort_type] = min(1.0, current + rate * dt)

    def set_dopamine(self, level: float) -> None:
        """Set dopamine level."""
        self.dopamine_level = np.clip(level, 0, 1)

    def get_fatigue_level(self) -> float:
        """Get overall fatigue level."""
        return 1 - np.mean(list(self.resources.values()))

    def get_willingness(self, effort_profile: EffortProfile, reward: float) -> float:
        """Get willingness to exert effort for reward."""
        cost = self.compute_cost(effort_profile)

        # Willingness based on reward/cost ratio
        if cost < 0.01:
            return 1.0

        ratio = reward / cost

        # Sigmoid transformation
        willingness = 1.0 / (1.0 + np.exp(-2 * (ratio - 1)))

        return willingness


class ParadoxicalEffort:
    """Models paradoxical effects where effort ADDS value.

    Phenomena explained:
    - Effort justification (IKEA effect)
    - Challenge preference (optimal difficulty)
    - Earned reward feels better
    - Deadline-induced effort valuation
    """

    def __init__(
        self,
        effort_justification_rate: float = 0.3,
        challenge_preference: float = 0.5,
        mastery_sensitivity: float = 0.4,
    ):
        self.justification_rate = effort_justification_rate
        self.challenge_pref = challenge_preference
        self.mastery_sensitivity = mastery_sensitivity

        # Track effort-outcome history
        self.effort_outcome_history: deque = deque(maxlen=100)

        # Skill level (affects optimal challenge)
        self.skill_levels: Dict[str, float] = {}

    def compute_effort_value(
        self, effort_expended: float, task_difficulty: float, skill_level: float, success: bool
    ) -> float:
        """Compute value ADDED by effort (can be negative or positive).

        Args:
            effort_expended: Amount of effort used
            task_difficulty: Task difficulty level
            skill_level: Agent's skill at this task
            success: Whether task was completed successfully

        Returns:
            Value added by the effort (can be positive!)
        """
        value = 0.0

        # 1. Effort justification - we value things we worked for
        if success:
            justification_value = self.justification_rate * effort_expended
            value += justification_value

        # 2. Challenge-skill match (flow)
        # Optimal challenge is slightly above skill level
        optimal_difficulty = skill_level + 0.1
        challenge_match = 1.0 - abs(task_difficulty - optimal_difficulty)
        challenge_match = max(0, challenge_match)

        if challenge_match > 0.5:
            # Good challenge match adds value
            flow_value = self.challenge_pref * challenge_match * effort_expended
            value += flow_value

        # 3. Mastery/learning signal
        # Difficulty above skill = learning opportunity
        learning_potential = max(0, task_difficulty - skill_level)
        if success and learning_potential > 0:
            mastery_value = self.mastery_sensitivity * learning_potential
            value += mastery_value

        # 4. But excessive effort without success is very negative
        if not success and effort_expended > 0.5:
            frustration = -0.5 * effort_expended
            value += frustration

        return value

    def compute_optimal_difficulty(self, skill_level: float) -> float:
        """Compute optimal task difficulty for maximum engagement."""
        # Optimal = slightly above skill (zone of proximal development)
        return skill_level + 0.1 + 0.05 * self.challenge_pref

    def update_skill(self, task_type: str, difficulty: float, success: bool) -> None:
        """Update skill level based on task outcome."""
        current = self.skill_levels.get(task_type, 0.5)

        if success:
            # Skill increases with successful harder tasks
            if difficulty > current:
                learning = 0.1 * (difficulty - current)
                self.skill_levels[task_type] = min(1.0, current + learning)
            else:
                # Small increase for easy tasks
                self.skill_levels[task_type] = min(1.0, current + 0.01)
        else:
            # Failure at appropriate difficulty is okay
            # Failure at easy tasks might indicate skill regression
            if difficulty < current - 0.2:
                self.skill_levels[task_type] = max(0, current - 0.05)

    def should_choose_harder(
        self,
        easy_reward: float,
        hard_reward: float,
        easy_effort: float,
        hard_effort: float,
        skill: float,
    ) -> bool:
        """Decide whether to choose harder option (children often do!).

        Implements the observation that children choose harder games.
        """
        # Easy option value
        easy_value = easy_reward - 0.5 * easy_effort

        # Hard option value (includes paradoxical effort value)
        hard_difficulty = hard_effort  # Proxy difficulty by effort
        paradoxical_value = self.compute_effort_value(
            hard_effort, hard_difficulty, skill, success=True
        )
        hard_value = hard_reward - 0.5 * hard_effort + paradoxical_value

        # Also consider optimal challenge
        optimal_diff = self.compute_optimal_difficulty(skill)
        if abs(hard_difficulty - optimal_diff) < abs(easy_effort - optimal_diff):
            hard_value += 0.2  # Bonus for being closer to optimal

        return hard_value > easy_value

    def record_outcome(
        self, effort: float, difficulty: float, success: bool, subjective_value: float
    ) -> None:
        """Record effort outcome for learning."""
        self.effort_outcome_history.append(
            {
                "effort": effort,
                "difficulty": difficulty,
                "success": success,
                "value": subjective_value,
            }
        )


class MotivationalTransform:
    """Transforms how effort is experienced based on motivational state.

    Key insight: Under high motivation (deadline, importance),
    effort becomes valued rather than costly.
    """

    def __init__(
        self,
        deadline_effect: float = 0.5,
        importance_effect: float = 0.4,
        autonomy_effect: float = 0.3,
    ):
        self.deadline_effect = deadline_effect
        self.importance_effect = importance_effect
        self.autonomy_effect = autonomy_effect

        # Current motivational state
        self.deadline_pressure = 0.0
        self.task_importance = 0.5
        self.autonomy_level = 0.5
        self.intrinsic_interest = 0.5

    def set_context(
        self,
        deadline: float = 0.0,
        importance: float = 0.5,
        autonomy: float = 0.5,
        interest: float = 0.5,
    ) -> None:
        """Set motivational context."""
        self.deadline_pressure = np.clip(deadline, 0, 1)
        self.task_importance = np.clip(importance, 0, 1)
        self.autonomy_level = np.clip(autonomy, 0, 1)
        self.intrinsic_interest = np.clip(interest, 0, 1)

    def transform_effort_cost(self, raw_cost: float) -> float:
        """Transform raw effort cost based on motivational state.

        High motivation can make effort feel less costly or even valuable.
        """
        transformed = raw_cost

        # Deadline pressure reduces perceived cost
        if self.deadline_pressure > 0.5:
            deadline_reduction = self.deadline_effect * (self.deadline_pressure - 0.5) * 2
            transformed *= 1 - deadline_reduction

        # High importance reduces cost
        if self.task_importance > 0.5:
            importance_reduction = self.importance_effect * (self.task_importance - 0.5) * 2
            transformed *= 1 - importance_reduction

        # Autonomy reduces cost
        if self.autonomy_level > 0.5:
            autonomy_reduction = self.autonomy_effect * (self.autonomy_level - 0.5) * 2
            transformed *= 1 - autonomy_reduction

        # Intrinsic interest can make effort positive!
        if self.intrinsic_interest > 0.7:
            interest_bonus = (self.intrinsic_interest - 0.7) * 3
            transformed -= interest_bonus * raw_cost

        return max(0, transformed)

    def transform_effort_value(self, raw_value: float, effort: float) -> float:
        """Transform the value of effort expenditure.

        Under right conditions, effort ADDS value to outcomes.
        """
        transformed = raw_value

        # Effort-reward combo under deadline feels more valuable
        if self.deadline_pressure > 0.7 and effort > 0:
            deadline_bonus = 0.3 * effort * self.deadline_pressure
            transformed += deadline_bonus

        # High autonomy + effort = ownership = more value
        if self.autonomy_level > 0.5 and effort > 0:
            ownership_bonus = 0.2 * effort * self.autonomy_level
            transformed += ownership_bonus

        # Intrinsic interest transforms effort into enjoyment
        if self.intrinsic_interest > 0.6:
            enjoyment = 0.4 * effort * (self.intrinsic_interest - 0.6)
            transformed += enjoyment

        return transformed

    def get_motivation_level(self) -> float:
        """Get overall motivation level."""
        return np.mean(
            [
                self.deadline_pressure,
                self.task_importance,
                self.autonomy_level,
                self.intrinsic_interest,
            ]
        )

    def predict_engagement(self, task_effort: float, task_reward: float) -> float:
        """Predict engagement level for a task."""
        # Raw value
        raw_value = task_reward - task_effort * 0.5

        # Transform
        transformed_cost = self.transform_effort_cost(task_effort * 0.5)
        transformed_value = self.transform_effort_value(task_reward, task_effort)

        net_value = transformed_value - transformed_cost

        # Engagement is sigmoid of net value
        engagement = 1.0 / (1.0 + np.exp(-2 * net_value))

        return engagement
