"""Executive Network - Integrated PFC control system

Combines inhibition, working memory, and cognitive flexibility
Includes conflict monitoring (ACC) and goal maintenance
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

from .inhibition import InhibitionSystem, InhibitionParams
from .working_memory import WorkingMemory, WMParams
from .cognitive_flexibility import CognitiveFlexibility, FlexibilityParams


@dataclass
class ExecutiveParams:
    """Parameters for executive network"""

    n_units: int = 100
    conflict_threshold: float = 0.5
    control_strength: float = 0.7
    goal_decay_rate: float = 0.01
    learning_rate: float = 0.1


class ConflictMonitor:
    """Anterior cingulate cortex (ACC) conflict monitoring

    Detects response conflict and signals need for control
    """

    def __init__(self, n_responses: int = 4, threshold: float = 0.5):
        self.n_responses = n_responses
        self.threshold = threshold

        # Response activations
        self.response_activation = np.zeros(n_responses)
        # Conflict signal (Hopfield energy-like measure)
        self.conflict = 0.0
        # History of conflict
        self.conflict_history = []

    def compute_conflict(self, response_activation: np.ndarray) -> float:
        """Compute conflict from response activations

        Uses energy-based conflict measure: high when multiple responses active
        """
        self.response_activation = response_activation.copy()

        # Conflict = product of activations (high when multiple responses compete)
        sorted_act = np.sort(response_activation)[::-1]
        if len(sorted_act) >= 2:
            # Conflict is high when top two responses are both active
            self.conflict = sorted_act[0] * sorted_act[1]
        else:
            self.conflict = 0.0

        self.conflict_history.append(self.conflict)

        return self.conflict

    def needs_control(self) -> bool:
        """Check if control adjustment is needed"""
        return self.conflict > self.threshold

    def get_control_signal(self) -> float:
        """Get control adjustment signal

        Higher conflict = stronger control signal
        """
        if self.conflict > self.threshold:
            return (self.conflict - self.threshold) / (1 - self.threshold)
        return 0.0

    def get_mean_conflict(self, window: int = 10) -> float:
        """Get mean conflict over recent trials"""
        if len(self.conflict_history) == 0:
            return 0.0
        recent = self.conflict_history[-window:]
        return np.mean(recent)


class PFCController:
    """Prefrontal cortex controller for top-down regulation

    Implements goal maintenance and biasing of processing
    """

    def __init__(self, n_goals: int = 5, params: Optional[ExecutiveParams] = None):
        self.n_goals = n_goals
        self.params = params or ExecutiveParams()

        # Goal representations
        self.goals = np.zeros((n_goals, self.params.n_units))
        # Goal activation levels
        self.goal_activation = np.zeros(n_goals)
        # Current primary goal
        self.current_goal = 0

        # Control signal strength
        self.control_level = 0.5

    def set_goal(self, goal_idx: int, goal_repr: np.ndarray):
        """Set a goal representation"""
        if 0 <= goal_idx < self.n_goals:
            if len(goal_repr) != self.params.n_units:
                goal_repr = np.resize(goal_repr, self.params.n_units)
            self.goals[goal_idx] = goal_repr
            self.goal_activation[goal_idx] = 1.0
            self.current_goal = goal_idx

    def maintain_goals(self, dt: float = 1.0):
        """Active maintenance of goal representations"""
        # Goals decay over time
        self.goal_activation *= 1 - self.params.goal_decay_rate * dt

        # Current goal gets maintenance boost
        self.goal_activation[self.current_goal] = min(
            1.0, self.goal_activation[self.current_goal] + 0.1 * dt
        )

    def get_biasing_signal(self, input_pattern: np.ndarray) -> np.ndarray:
        """Generate top-down biasing signal for input processing"""
        if len(input_pattern) != self.params.n_units:
            input_pattern = np.resize(input_pattern, self.params.n_units)

        # Bias is weighted sum of goals
        bias = np.zeros(self.params.n_units)
        for i in range(self.n_goals):
            bias += self.goals[i] * self.goal_activation[i]

        # Scale by control level
        bias *= self.control_level * self.params.control_strength

        return bias

    def adjust_control(self, conflict_signal: float):
        """Adjust control level based on conflict"""
        # High conflict -> increase control
        self.control_level = np.clip(
            self.control_level + conflict_signal * self.params.learning_rate, 0.2, 1.0
        )

    def get_state(self) -> dict:
        """Get controller state"""
        return {
            "current_goal": self.current_goal,
            "goal_activation": self.goal_activation.copy(),
            "control_level": self.control_level,
        }


class ExecutiveNetwork:
    """Integrated executive function network

    Combines all executive components with PFC coordination
    """

    def __init__(self, params: Optional[ExecutiveParams] = None):
        self.params = params or ExecutiveParams()

        # Component systems
        self.inhibition = InhibitionSystem(n_actions=10)
        self.working_memory = WorkingMemory()
        self.flexibility = CognitiveFlexibility()

        # Control systems
        self.conflict_monitor = ConflictMonitor(n_responses=10)
        self.controller = PFCController(n_goals=5, params=self.params)

        # Global state
        self.arousal = 0.5  # Overall activation level
        self.fatigue = 0.0  # Accumulates with use

    def process_stimulus(self, stimulus: np.ndarray, task_context: int = 0) -> dict:
        """Process a stimulus with full executive control

        Args:
            stimulus: Input stimulus pattern
            task_context: Current task index

        Returns:
            Processing results
        """
        # Ensure stimulus is right size
        if len(stimulus) != self.params.n_units:
            stimulus = np.resize(stimulus, self.params.n_units)

        # 1. Get top-down biasing from goals
        bias = self.controller.get_biasing_signal(stimulus)

        # 2. Biased stimulus processing
        biased_stimulus = stimulus + bias
        biased_stimulus = np.clip(biased_stimulus, 0, 1)

        # 3. Generate response activations
        n_responses = 10
        response_weights = np.random.randn(n_responses, self.params.n_units) * 0.1
        response_activation = np.dot(response_weights, biased_stimulus)
        response_activation = 1 / (1 + np.exp(-response_activation))  # Sigmoid

        # 4. Monitor conflict
        conflict = self.conflict_monitor.compute_conflict(response_activation)

        # 5. Apply inhibition if needed
        if self.conflict_monitor.needs_control():
            inhibition_strength = self.inhibition.get_inhibition_strength()
            response_activation *= 1 - inhibition_strength * 0.5
            self.controller.adjust_control(self.conflict_monitor.get_control_signal())

        # 6. Select response (winner-take-all with noise)
        response_activation += np.random.randn(n_responses) * 0.05
        selected_response = np.argmax(response_activation)

        # 7. Store in working memory if relevant
        if np.max(response_activation) > 0.7:
            self.working_memory.encode(stimulus)

        # Update fatigue
        self.fatigue = min(1.0, self.fatigue + 0.01)

        return {
            "selected_response": selected_response,
            "response_activation": response_activation,
            "conflict": conflict,
            "control_level": self.controller.control_level,
            "wm_load": self.working_memory.get_load(),
            "fatigue": self.fatigue,
        }

    def update(self, dt: float = 1.0):
        """Update all executive systems"""
        self.controller.maintain_goals(dt)
        self.working_memory.update_state(dt)
        self.flexibility.update(dt)
        self.inhibition.update(dt)

        # Arousal affects overall processing
        self.arousal = np.clip(0.5 - self.fatigue * 0.3 + np.random.randn() * 0.05, 0.2, 1.0)

        # Recovery from fatigue
        self.fatigue = max(0, self.fatigue - 0.005 * dt)

    def set_goal(self, goal_idx: int, goal_pattern: np.ndarray):
        """Set a new goal"""
        self.controller.set_goal(goal_idx, goal_pattern)

    def switch_task(self, new_task: int):
        """Switch to a new task"""
        self.flexibility.task_switcher.switch_task(new_task)

    def get_executive_state(self) -> dict:
        """Get comprehensive executive state"""
        return {
            "controller": self.controller.get_state(),
            "conflict": self.conflict_monitor.conflict,
            "mean_conflict": self.conflict_monitor.get_mean_conflict(),
            "flexibility": self.flexibility.get_flexibility_state(),
            "wm_load": self.working_memory.get_load(),
            "inhibition_strength": self.inhibition.get_inhibition_strength(),
            "arousal": self.arousal,
            "fatigue": self.fatigue,
        }

    def run_stroop_trial(self, word: str, color: str, task: str = "color") -> dict:
        """Run a Stroop-like trial

        Args:
            word: The written word
            color: The ink color
            task: 'color' (name color) or 'word' (read word)

        Returns:
            Trial results
        """
        # Encode word and color as patterns
        word_pattern = np.random.rand(self.params.n_units)
        word_pattern *= hash(word) % 100 / 100  # Deterministic from word

        color_pattern = np.random.rand(self.params.n_units)
        color_pattern *= hash(color) % 100 / 100

        # Congruent if word matches color
        congruent = word.lower() == color.lower()

        # Combined stimulus
        stimulus = word_pattern * 0.5 + color_pattern * 0.5

        # Set goal based on task
        if task == "color":
            self.controller.set_goal(0, color_pattern)
        else:
            self.controller.set_goal(0, word_pattern)

        # Process
        result = self.process_stimulus(stimulus)

        # Determine correctness (simplified)
        result["congruent"] = congruent
        result["task"] = task

        # Incongruent trials typically have higher conflict
        if not congruent:
            result["conflict"] *= 1.5

        return result
