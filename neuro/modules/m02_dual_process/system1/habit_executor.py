"""
Habit Executor - Basal Ganglia Simulation

Implements automatic stimulus-response mappings that execute without deliberation.
The basal ganglia stores learned action sequences that can be triggered by
specific stimuli - this is how habits work.

Key properties:
- No deliberation required - direct mapping
- Context-dependent activation
- Strengthened through repetition
- Can be weakened through extinction
"""

import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time


class HabitStrength(Enum):
    """How ingrained a habit is"""
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    AUTOMATIC = 4


@dataclass
class Action:
    """An action or action sequence"""
    id: str
    execute: Optional[Callable] = None
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        return f"Action({self.id})"


@dataclass
class Habit:
    """A stimulus-response association"""
    trigger_pattern: np.ndarray
    action: Action
    strength: float = 0.5
    repetition_count: int = 0
    last_triggered: float = 0.0
    contexts: List[Any] = field(default_factory=list)
    success_rate: float = 1.0


@dataclass
class HabitResponse:
    """Result of habit execution attempt"""
    triggered: bool
    action: Optional[Action]
    habit_strength: float
    confidence: float


class HabitExecutor:
    """
    Stimulus-response mappings with automatic execution.

    Simulates basal ganglia:
    - Direct stimulus->action mapping (no deliberation)
    - Habits strengthen with repetition
    - Context-dependent triggering
    - Extinction through non-reinforcement
    """

    def __init__(self,
                 trigger_threshold: float = 0.6,
                 learning_rate: float = 0.1,
                 decay_rate: float = 0.01):
        self.habits: List[Habit] = []
        self.trigger_threshold = trigger_threshold
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate

    def add_habit(self,
                  trigger_pattern: np.ndarray,
                  action: Action,
                  initial_strength: float = 0.3,
                  contexts: Optional[List[Any]] = None) -> Habit:
        """
        Create a new habit association.

        Initially weak - must be strengthened through repetition.
        """
        habit = Habit(
            trigger_pattern=trigger_pattern,
            action=action,
            strength=initial_strength,
            contexts=contexts or []
        )
        self.habits.append(habit)
        return habit

    def execute(self,
                stimulus: np.ndarray,
                context: Optional[Any] = None,
                allow_multiple: bool = False) -> HabitResponse:
        """
        Check if stimulus triggers any habits.

        This is FAST and AUTOMATIC - no deliberation.
        Returns the strongest matching habit response.
        """
        matches = []

        for habit in self.habits:
            # Context check - habit may only fire in certain contexts
            if habit.contexts and context not in habit.contexts:
                continue

            # Compute stimulus similarity to trigger pattern
            similarity = self._compute_similarity(stimulus, habit.trigger_pattern)

            # Effective threshold is lower for stronger habits
            effective_threshold = self.trigger_threshold * (1.0 - habit.strength * 0.3)

            if similarity >= effective_threshold:
                confidence = similarity * habit.strength
                matches.append((habit, confidence))

        if not matches:
            return HabitResponse(
                triggered=False,
                action=None,
                habit_strength=0.0,
                confidence=0.0
            )

        # Sort by confidence, get strongest
        matches.sort(key=lambda x: x[1], reverse=True)

        if allow_multiple:
            # Return all triggered habits (rare case)
            best_habit, best_confidence = matches[0]
        else:
            # Winner-take-all (typical basal ganglia behavior)
            best_habit, best_confidence = matches[0]

        # Update last triggered time
        best_habit.last_triggered = time.time()

        return HabitResponse(
            triggered=True,
            action=best_habit.action,
            habit_strength=best_habit.strength,
            confidence=best_confidence
        )

    def _compute_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute similarity between stimulus and trigger pattern"""
        a_norm = a / (np.linalg.norm(a) + 1e-8)
        b_norm = b / (np.linalg.norm(b) + 1e-8)
        return float(np.clip(np.dot(a_norm, b_norm), 0, 1))

    def strengthen(self,
                   stimulus: np.ndarray,
                   action: Action,
                   reward: float = 1.0) -> Optional[Habit]:
        """
        Strengthen habit through successful execution.

        This is how habits form - repetition with positive outcome
        strengthens the stimulus-action association.
        """
        # Find matching habit
        best_habit = None
        best_similarity = 0.0

        for habit in self.habits:
            if habit.action.id == action.id:
                similarity = self._compute_similarity(stimulus, habit.trigger_pattern)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_habit = habit

        if best_habit is None:
            # Create new habit if none exists
            return self.add_habit(stimulus, action, initial_strength=0.2)

        # Strengthen existing habit
        # Diminishing returns as habit gets stronger
        delta = self.learning_rate * reward * (1.0 - best_habit.strength)
        best_habit.strength = min(1.0, best_habit.strength + delta)
        best_habit.repetition_count += 1

        # Update trigger pattern (weighted average)
        weight = 1.0 / (best_habit.repetition_count + 1)
        best_habit.trigger_pattern = (
            best_habit.trigger_pattern * (1 - weight) +
            stimulus * weight
        )

        return best_habit

    def weaken(self,
               stimulus: np.ndarray,
               extinction_rate: float = 0.2) -> Optional[Habit]:
        """
        Weaken habit through non-reinforcement or negative outcome.

        This is extinction - habits fade when not reinforced.
        """
        best_habit = None
        best_similarity = 0.0

        for habit in self.habits:
            similarity = self._compute_similarity(stimulus, habit.trigger_pattern)
            if similarity > best_similarity:
                best_similarity = similarity
                best_habit = habit

        if best_habit:
            best_habit.strength = max(0.0, best_habit.strength - extinction_rate)
            best_habit.success_rate *= 0.9  # Track declining success

            # Remove very weak habits
            if best_habit.strength < 0.05:
                self.habits.remove(best_habit)
                return None

        return best_habit

    def decay_all(self):
        """
        Apply time-based decay to all habits.

        Habits that aren't used gradually weaken.
        """
        current_time = time.time()

        for habit in self.habits[:]:  # Copy list for safe removal
            time_since_use = current_time - habit.last_triggered

            # Decay based on time and strength
            # Stronger habits decay slower
            decay = self.decay_rate * (1.0 - habit.strength * 0.5) * min(time_since_use / 3600, 1.0)
            habit.strength = max(0.0, habit.strength - decay)

            # Remove very weak habits
            if habit.strength < 0.05:
                self.habits.remove(habit)

    def get_strength(self, stimulus: np.ndarray) -> HabitStrength:
        """Get the strength category for habits matching this stimulus"""
        response = self.execute(stimulus)

        if not response.triggered:
            return HabitStrength.WEAK

        if response.habit_strength >= 0.9:
            return HabitStrength.AUTOMATIC
        elif response.habit_strength >= 0.7:
            return HabitStrength.STRONG
        elif response.habit_strength >= 0.4:
            return HabitStrength.MODERATE
        else:
            return HabitStrength.WEAK

    def inhibit(self, stimulus: np.ndarray) -> bool:
        """
        Attempt to inhibit habit execution.

        This is called by cognitive control (System 2) to override habits.
        Returns True if inhibition was successful (habit was weak enough).
        """
        response = self.execute(stimulus)

        if not response.triggered:
            return True  # Nothing to inhibit

        # Strong habits are hard to inhibit
        # This creates the "automatic" feeling of habits
        inhibition_difficulty = response.habit_strength ** 2

        # Random element - sometimes you can override, sometimes not
        return np.random.random() > inhibition_difficulty
