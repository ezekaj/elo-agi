"""
Feedback Adaptation - Continuous Environment Learning

Learn from environmental feedback through action-outcome associations.
Implements exploration-exploitation tradeoff.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum


class ActionOutcome(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


@dataclass
class Action:
    """An action that can be taken"""

    action_id: str
    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    preconditions: List[str] = field(default_factory=list)


@dataclass
class State:
    """A state of the environment"""

    state_id: str
    features: Dict[str, Any] = field(default_factory=dict)

    def to_tuple(self) -> tuple:
        """Convert to hashable tuple for dict keys"""
        return tuple(sorted(self.features.items()))


@dataclass
class Experience:
    """A single experience (state, action, outcome, next_state)"""

    state: State
    action: Action
    outcome: ActionOutcome
    reward: float
    next_state: State
    timestamp: float = 0.0


class AdaptivePolicy:
    """A policy that improves from feedback"""

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        exploration_rate: float = 0.2,
    ):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

        self.q_values: Dict[tuple, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.action_counts: Dict[tuple, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.total_reward = 0.0
        self.n_updates = 0

    def select_action(
        self, state: State, available_actions: List[Action], explore: bool = True
    ) -> Action:
        """Select an action using epsilon-greedy policy"""
        if not available_actions:
            return None

        state_key = state.to_tuple()

        if explore and np.random.random() < self.exploration_rate:
            return np.random.choice(available_actions)

        best_action = None
        best_value = float("-inf")

        for action in available_actions:
            value = self.q_values[state_key][action.action_id]

            count = self.action_counts[state_key][action.action_id]
            if count > 0:
                value += np.sqrt(
                    2 * np.log(sum(self.action_counts[state_key].values()) + 1) / count
                )

            if value > best_value:
                best_value = value
                best_action = action

        return best_action or available_actions[0]

    def update(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: State,
        next_actions: List[Action] = None,
    ):
        """Update Q-values based on experience (Q-learning)"""
        state_key = state.to_tuple()
        next_state_key = next_state.to_tuple()

        self.action_counts[state_key][action.action_id] += 1

        current_q = self.q_values[state_key][action.action_id]

        if next_actions:
            max_next_q = (
                max(self.q_values[next_state_key][a.action_id] for a in next_actions)
                if next_actions
                else 0
            )
        else:
            max_next_q = (
                max(self.q_values[next_state_key].values()) if self.q_values[next_state_key] else 0
            )

        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )

        self.q_values[state_key][action.action_id] = new_q
        self.total_reward += reward
        self.n_updates += 1

    def get_policy_value(self, state: State) -> float:
        """Get the value of a state under current policy"""
        state_key = state.to_tuple()
        if not self.q_values[state_key]:
            return 0.0
        return max(self.q_values[state_key].values())

    def decay_exploration(self, decay_rate: float = 0.99):
        """Reduce exploration rate over time"""
        self.exploration_rate *= decay_rate
        self.exploration_rate = max(0.01, self.exploration_rate)


class FeedbackAdapter:
    """
    Learn from environmental feedback.
    Continuous adaptation through experience.
    """

    def __init__(self, learning_rate: float = 0.1, exploration_rate: float = 0.3):
        self.policy = AdaptivePolicy(learning_rate=learning_rate, exploration_rate=exploration_rate)
        self.experiences: List[Experience] = []
        self.current_state: Optional[State] = None
        self.available_actions: List[Action] = []
        self.action_models: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.outcome_predictions: Dict[str, List[float]] = defaultdict(list)

    def set_state(self, state: State):
        """Set the current state"""
        self.current_state = state

    def set_available_actions(self, actions: List[Action]):
        """Set available actions in current state"""
        self.available_actions = actions

    def act(self, action: Action = None) -> Action:
        """Select and return an action"""
        if action is None:
            action = self.policy.select_action(self.current_state, self.available_actions)
        return action

    def observe_outcome(
        self, action: Action, outcome: ActionOutcome, reward: float, next_state: State
    ):
        """Observe the outcome of an action"""
        experience = Experience(
            state=self.current_state,
            action=action,
            outcome=outcome,
            reward=reward,
            next_state=next_state,
            timestamp=len(self.experiences),
        )
        self.experiences.append(experience)

        return experience

    def update_model(
        self,
        action: Action,
        outcome: ActionOutcome,
        reward: float,
        next_state: State,
        next_actions: List[Action] = None,
    ):
        """Update internal model from experience"""
        # Record experience first
        experience = Experience(
            state=self.current_state,
            action=action,
            outcome=outcome,
            reward=reward,
            next_state=next_state,
            timestamp=len(self.experiences),
        )
        self.experiences.append(experience)

        # Update policy
        self.policy.update(self.current_state, action, reward, next_state, next_actions)

        state_key = str(self.current_state.to_tuple())
        action_key = action.action_id

        self.action_models[action_key][state_key] = reward

        if outcome == ActionOutcome.SUCCESS:
            self.outcome_predictions[action_key].append(1.0)
        elif outcome == ActionOutcome.FAILURE:
            self.outcome_predictions[action_key].append(0.0)
        else:
            self.outcome_predictions[action_key].append(0.5)

        self.current_state = next_state

    def exploration_exploitation(
        self, state: State, available_actions: List[Action]
    ) -> Tuple[Action, str]:
        """Decide between exploration and exploitation"""
        state_key = state.to_tuple()

        for action in available_actions:
            if self.policy.action_counts[state_key][action.action_id] == 0:
                return action, "explore_novel"

        total_uncertainty = sum(
            1.0 / (self.policy.action_counts[state_key][a.action_id] + 1) for a in available_actions
        )

        if total_uncertainty > len(available_actions) * 0.5:
            least_tried = min(
                available_actions, key=lambda a: self.policy.action_counts[state_key][a.action_id]
            )
            return least_tried, "explore_uncertain"

        best_action = self.policy.select_action(state, available_actions, explore=False)
        return best_action, "exploit"

    def predict_outcome(self, action: Action) -> Tuple[float, float]:
        """Predict outcome probability and expected reward for an action"""
        if action.action_id not in self.outcome_predictions:
            return 0.5, 0.0

        outcomes = self.outcome_predictions[action.action_id]
        if not outcomes:
            return 0.5, 0.0

        success_prob = np.mean(outcomes)

        rewards = [
            exp.reward for exp in self.experiences if exp.action.action_id == action.action_id
        ]
        expected_reward = np.mean(rewards) if rewards else 0.0

        return success_prob, expected_reward

    def adapt_strategy(self, recent_window: int = 10) -> Dict[str, Any]:
        """Analyze recent performance and adapt strategy"""
        if len(self.experiences) < recent_window:
            return {"status": "insufficient_data"}

        recent = self.experiences[-recent_window:]

        avg_reward = np.mean([exp.reward for exp in recent])
        success_rate = np.mean(
            [1.0 if exp.outcome == ActionOutcome.SUCCESS else 0.0 for exp in recent]
        )

        if success_rate < 0.3:
            self.policy.exploration_rate = min(0.5, self.policy.exploration_rate * 1.2)
            strategy_change = "increase_exploration"
        elif success_rate > 0.7:
            self.policy.decay_exploration(0.95)
            strategy_change = "decrease_exploration"
        else:
            strategy_change = "maintain"

        return {
            "avg_reward": avg_reward,
            "success_rate": success_rate,
            "exploration_rate": self.policy.exploration_rate,
            "strategy_change": strategy_change,
        }

    def replay_experiences(self, n_replays: int = 10, prioritized: bool = True):
        """Replay past experiences to improve learning"""
        if not self.experiences:
            return

        if prioritized:
            surprises = []
            for exp in self.experiences:
                state_key = exp.state.to_tuple()
                predicted_q = self.policy.q_values[state_key].get(exp.action.action_id, 0)
                actual = exp.reward
                surprise = abs(actual - predicted_q)
                surprises.append((surprise, exp))

            surprises.sort(reverse=True)
            to_replay = [exp for _, exp in surprises[:n_replays]]
        else:
            to_replay = np.random.choice(
                self.experiences, size=min(n_replays, len(self.experiences)), replace=False
            )

        for exp in to_replay:
            self.policy.update(exp.state, exp.action, exp.reward, exp.next_state)
