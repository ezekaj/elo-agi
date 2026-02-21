"""Cognitive Flexibility - Task switching and set shifting

Neural basis: Lateral prefrontal cortex (shares substrate with working memory)
Key functions: Task switching, rule switching, set shifting
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Callable


@dataclass
class FlexibilityParams:
    """Parameters for cognitive flexibility"""
    n_rules: int = 4
    n_dimensions: int = 3
    switch_cost: float = 0.2  # RT increase on switch trials
    mixing_cost: float = 0.1  # RT increase in mixed blocks
    preparation_benefit: float = 0.1
    perseveration_strength: float = 0.3
    learning_rate: float = 0.1


class TaskSwitcher:
    """Models task switching between different rule sets

    Implements switch costs and task-set reconfiguration
    """

    def __init__(self, n_tasks: int = 2, params: Optional[FlexibilityParams] = None):
        self.n_tasks = n_tasks
        self.params = params or FlexibilityParams()

        # Current active task
        self.current_task = 0
        # Task activation levels
        self.task_activation = np.zeros(n_tasks)
        self.task_activation[0] = 1.0
        # History of task switches
        self.switch_history = []
        # Preparation state
        self.prepared = False
        self.preparation_time = 0.0

    def switch_task(self, new_task: int):
        """Switch to a new task"""
        if new_task == self.current_task:
            return

        is_switch = True
        self.switch_history.append({
            "from": self.current_task,
            "to": new_task,
            "prepared": self.prepared
        })

        # Task-set reconfiguration
        self.task_activation[self.current_task] *= 0.5  # Decay old task
        self.current_task = new_task
        self.task_activation[new_task] = 0.5  # Partial activation of new task
        self.prepared = False
        self.preparation_time = 0.0

    def prepare_task(self, task: int, prep_time: float):
        """Prepare for upcoming task (advance cuing)"""
        self.preparation_time = prep_time
        self.task_activation[task] = min(1.0, 0.5 + prep_time / 1000.0)
        if task == self.current_task:
            self.prepared = True

    def execute_trial(self, stimulus: np.ndarray, task: int) -> dict:
        """Execute a trial with given stimulus and task

        Args:
            stimulus: Input stimulus
            task: Which task to perform

        Returns:
            Trial results including RT and accuracy
        """
        is_switch = task != self.current_task
        was_prepared = self.prepared

        if is_switch:
            self.switch_task(task)

        # Base reaction time
        base_rt = 500.0  # ms

        # Add switch cost
        if is_switch:
            switch_cost = self.params.switch_cost * 1000  # Convert to ms
            if was_prepared:
                switch_cost *= (1 - self.params.preparation_benefit)
            base_rt += switch_cost

        # Add noise
        rt = base_rt + np.random.randn() * 50

        # Boost task activation
        self.task_activation[task] = min(1.0, self.task_activation[task] + 0.2)

        # Accuracy depends on task activation
        error_prob = 0.05 + 0.1 * (1 - self.task_activation[task])
        correct = np.random.rand() > error_prob

        return {
            "task": task,
            "is_switch": is_switch,
            "prepared": was_prepared,
            "rt": rt,
            "correct": correct,
            "task_activation": self.task_activation[task]
        }

    def get_switch_cost(self) -> float:
        """Calculate average switch cost from history"""
        if len(self.switch_history) < 2:
            return self.params.switch_cost
        return self.params.switch_cost


class SetShifter:
    """Models set shifting (e.g., Wisconsin Card Sort Test)

    Implements rule learning and perseveration
    """

    def __init__(self, params: Optional[FlexibilityParams] = None):
        self.params = params or FlexibilityParams()

        # Dimensions: color, shape, number (for WCST-like task)
        self.n_dimensions = self.params.n_dimensions

        # Current sorting rule (which dimension)
        self.current_rule = 0
        # Rule strength for each dimension
        self.rule_strength = np.zeros(self.n_dimensions)
        self.rule_strength[0] = 1.0

        # Consecutive correct responses
        self.consecutive_correct = 0
        # Errors
        self.perseverative_errors = 0
        self.total_errors = 0
        # Categories completed
        self.categories_completed = 0

    def set_rule(self, rule: int):
        """Set the correct sorting rule"""
        if 0 <= rule < self.n_dimensions:
            self.current_rule = rule

    def shift_rule(self):
        """Shift to a new rule (after criterion reached)"""
        old_rule = self.current_rule
        # Pick a different rule
        available = [r for r in range(self.n_dimensions) if r != old_rule]
        self.current_rule = np.random.choice(available)
        self.consecutive_correct = 0
        self.categories_completed += 1

    def sort_card(self, card: Dict, choice_dimension: int) -> dict:
        """Sort a card by chosen dimension

        Args:
            card: Card features {dimension: value}
            choice_dimension: Which dimension the subject sorted by

        Returns:
            Feedback dictionary
        """
        correct = choice_dimension == self.current_rule

        # Update rule strengths based on feedback
        if correct:
            self.consecutive_correct += 1
            self.rule_strength[choice_dimension] = min(
                1.0, self.rule_strength[choice_dimension] + self.params.learning_rate
            )
        else:
            self.total_errors += 1
            self.consecutive_correct = 0

            # Check for perseverative error (sticking to old rule)
            if self.rule_strength[choice_dimension] > 0.5:
                self.perseverative_errors += 1

            # Decrease strength of wrong rule
            self.rule_strength[choice_dimension] = max(
                0.0, self.rule_strength[choice_dimension] - self.params.learning_rate
            )

        # Shift rule after criterion (e.g., 10 correct)
        if self.consecutive_correct >= 10:
            self.shift_rule()

        return {
            "correct": correct,
            "correct_rule": self.current_rule,
            "chosen_rule": choice_dimension,
            "consecutive_correct": self.consecutive_correct,
            "is_perseverative": not correct and self.rule_strength[choice_dimension] > 0.5
        }

    def model_response(self, card: Dict) -> int:
        """Model which dimension the system would choose

        Uses softmax over rule strengths with perseveration bias
        """
        # Add perseveration to current strongest rule
        biased_strengths = self.rule_strength.copy()
        strongest = np.argmax(biased_strengths)
        biased_strengths[strongest] += self.params.perseveration_strength

        # Softmax selection
        exp_strengths = np.exp(biased_strengths * 2)  # Temperature parameter
        probs = exp_strengths / np.sum(exp_strengths)

        return np.random.choice(self.n_dimensions, p=probs)

    def get_performance_summary(self) -> dict:
        """Get summary of performance"""
        return {
            "categories_completed": self.categories_completed,
            "total_errors": self.total_errors,
            "perseverative_errors": self.perseverative_errors,
            "perseveration_rate": self.perseverative_errors / max(1, self.total_errors),
            "current_rule_strength": self.rule_strength.copy()
        }

    def reset(self):
        """Reset for new test"""
        self.current_rule = 0
        self.rule_strength = np.zeros(self.n_dimensions)
        self.rule_strength[0] = 1.0
        self.consecutive_correct = 0
        self.perseverative_errors = 0
        self.total_errors = 0
        self.categories_completed = 0


class CognitiveFlexibility:
    """Integrated cognitive flexibility system

    Combines task switching and set shifting with lateral PFC dynamics
    """

    def __init__(self, params: Optional[FlexibilityParams] = None):
        self.params = params or FlexibilityParams()

        self.task_switcher = TaskSwitcher(self.params.n_rules, self.params)
        self.set_shifter = SetShifter(self.params)

        # Lateral PFC activation
        self.lpfc_activation = np.zeros(50)

        # Flexibility index (0 = rigid, 1 = flexible)
        self.flexibility_index = 0.5

    def adapt_to_feedback(self, correct: bool):
        """Adapt flexibility based on feedback"""
        if correct:
            # Successful - can be slightly less flexible (exploit)
            self.flexibility_index = max(0.2, self.flexibility_index - 0.02)
        else:
            # Error - increase flexibility (explore)
            self.flexibility_index = min(1.0, self.flexibility_index + 0.05)

    def get_flexibility_state(self) -> dict:
        """Get current flexibility state"""
        return {
            "flexibility_index": self.flexibility_index,
            "current_task": self.task_switcher.current_task,
            "task_activations": self.task_switcher.task_activation.copy(),
            "rule_strengths": self.set_shifter.rule_strength.copy(),
            "lpfc_mean_activity": np.mean(self.lpfc_activation)
        }

    def update(self, dt: float = 1.0):
        """Update flexibility system state"""
        # lPFC dynamics
        target = np.ones(50) * self.flexibility_index
        self.lpfc_activation += 0.1 * (target - self.lpfc_activation) * dt

        # Decay task activations
        self.task_switcher.task_activation *= (1 - 0.01 * dt)
        # Keep current task active
        self.task_switcher.task_activation[self.task_switcher.current_task] = max(
            0.5, self.task_switcher.task_activation[self.task_switcher.current_task]
        )
