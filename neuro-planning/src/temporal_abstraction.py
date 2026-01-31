"""
Temporal Abstraction with Options Framework

Implements the options framework for temporal abstraction in hierarchical RL,
enabling multi-step action sequences with learned initiation and termination.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple, Set
from enum import Enum
import numpy as np


class OptionState(Enum):
    """State of an option during execution."""
    INACTIVE = "inactive"
    EXECUTING = "executing"
    TERMINATED = "terminated"


@dataclass
class TerminationCondition:
    """Termination condition for an option."""
    name: str
    predicate: Callable[[Any], bool]
    probability: Callable[[Any], float] = field(
        default_factory=lambda: lambda s: 0.0
    )

    def should_terminate(self, state: Any) -> bool:
        """Check if option should terminate deterministically."""
        return self.predicate(state)

    def termination_probability(self, state: Any) -> float:
        """Get probability of termination in state."""
        if self.predicate(state):
            return 1.0
        return self.probability(state)


@dataclass
class OptionPolicy:
    """Policy for selecting actions within an option."""
    name: str
    policy_fn: Callable[[Any], Any]
    stochastic: bool = False
    probability_fn: Optional[Callable[[Any, Any], float]] = None

    def select_action(self, state: Any, rng: Optional[np.random.Generator] = None) -> Any:
        """Select an action given the state."""
        return self.policy_fn(state)

    def action_probability(self, state: Any, action: Any) -> float:
        """Get probability of selecting action in state."""
        if not self.stochastic:
            selected = self.select_action(state)
            return 1.0 if action == selected else 0.0
        if self.probability_fn is not None:
            return self.probability_fn(state, action)
        # Default uniform probability if no probability function provided
        return 0.5


@dataclass
class Option:
    """
    An option in the options framework.

    An option o = <I, π, β> consists of:
    - I: Initiation set (states where option can start)
    - π: Option policy (action selection within option)
    - β: Termination condition (when option ends)
    """
    name: str
    initiation_set: Callable[[Any], bool]
    policy: OptionPolicy
    termination: TerminationCondition
    reward_model: Optional[Callable[[Any, Any, Any], float]] = None

    state: OptionState = OptionState.INACTIVE
    start_state: Optional[Any] = None
    steps_taken: int = 0
    cumulative_reward: float = 0.0

    def can_initiate(self, state: Any) -> bool:
        """Check if option can be initiated in state."""
        return self.initiation_set(state)

    def initiate(self, state: Any) -> bool:
        """Initiate the option in the given state."""
        if not self.can_initiate(state):
            return False

        self.state = OptionState.EXECUTING
        self.start_state = state
        self.steps_taken = 0
        self.cumulative_reward = 0.0
        return True

    def step(self, state: Any, rng: Optional[np.random.Generator] = None) -> Tuple[Any, bool]:
        """
        Execute one step of the option.

        Returns (action, terminated).
        """
        if self.state != OptionState.EXECUTING:
            return None, True

        action = self.policy.select_action(state, rng)
        self.steps_taken += 1

        should_terminate = self.termination.should_terminate(state)
        if should_terminate:
            self.state = OptionState.TERMINATED

        return action, should_terminate

    def add_reward(self, reward: float, discount: float = 0.99) -> None:
        """Add reward to cumulative total."""
        self.cumulative_reward += (discount ** (self.steps_taken - 1)) * reward

    def reset(self) -> None:
        """Reset option to inactive state."""
        self.state = OptionState.INACTIVE
        self.start_state = None
        self.steps_taken = 0
        self.cumulative_reward = 0.0


@dataclass
class IntraOptionLearning:
    """
    Intra-option learning for updating option values.

    Implements learning that occurs during option execution,
    not just at option boundaries.
    """
    discount: float = 0.99
    learning_rate: float = 0.1

    option_values: Dict[str, Dict[str, float]] = field(default_factory=dict)
    option_q_values: Dict[str, Dict[Tuple[str, str], float]] = field(default_factory=dict)
    update_count: int = 0

    def get_option_value(
        self,
        option_name: str,
        state: Any,
        state_key: Optional[str] = None,
    ) -> float:
        """Get V_o(s) - value of option in state."""
        if state_key is None:
            state_key = str(hash(str(state)))

        return self.option_values.get(option_name, {}).get(state_key, 0.0)

    def get_option_q_value(
        self,
        option_name: str,
        state: Any,
        action: Any,
        state_key: Optional[str] = None,
    ) -> float:
        """Get Q_o(s, a) - action value within option."""
        if state_key is None:
            state_key = str(hash(str(state)))

        action_key = str(hash(str(action)))
        return self.option_q_values.get(option_name, {}).get(
            (state_key, action_key), 0.0
        )

    def update(
        self,
        option: Option,
        state: Any,
        action: Any,
        reward: float,
        next_state: Any,
        terminated: bool,
        state_key: Optional[str] = None,
        next_state_key: Optional[str] = None,
    ) -> float:
        """
        Intra-option TD update.

        Updates Q_o(s, a) using:
        Q_o(s, a) <- Q_o(s, a) + α[r + γ * U(s') - Q_o(s, a)]

        Where U(s') = (1 - β(s')) * Q_o(s', π(s')) + β(s') * V(s')
        """
        if state_key is None:
            state_key = str(hash(str(state)))
        if next_state_key is None:
            next_state_key = str(hash(str(next_state)))

        action_key = str(hash(str(action)))
        option_name = option.name

        if option_name not in self.option_q_values:
            self.option_q_values[option_name] = {}
        if option_name not in self.option_values:
            self.option_values[option_name] = {}

        current_q = self.option_q_values[option_name].get((state_key, action_key), 0.0)

        term_prob = option.termination.termination_probability(next_state)

        if terminated:
            next_value = self.option_values.get(option_name, {}).get(next_state_key, 0.0)
        else:
            continuation_value = self.get_option_q_value(
                option_name, next_state, option.policy.select_action(next_state)
            )
            terminal_value = self.option_values.get(option_name, {}).get(
                next_state_key, 0.0
            )
            next_value = (1 - term_prob) * continuation_value + term_prob * terminal_value

        td_target = reward + self.discount * next_value
        td_error = td_target - current_q

        self.option_q_values[option_name][(state_key, action_key)] = (
            current_q + self.learning_rate * td_error
        )

        self.update_count += 1

        return td_error

    def update_option_value(
        self,
        option_name: str,
        state: Any,
        value: float,
        state_key: Optional[str] = None,
    ) -> None:
        """Update option value function directly."""
        if state_key is None:
            state_key = str(hash(str(state)))

        if option_name not in self.option_values:
            self.option_values[option_name] = {}

        current = self.option_values[option_name].get(state_key, 0.0)
        self.option_values[option_name][state_key] = (
            current + self.learning_rate * (value - current)
        )

    def statistics(self) -> Dict[str, Any]:
        """Get learning statistics."""
        return {
            "options_tracked": len(self.option_values),
            "q_values_stored": sum(len(v) for v in self.option_q_values.values()),
            "update_count": self.update_count,
        }


class OptionsFramework:
    """
    Main options framework manager.

    Handles option creation, selection, and execution with support
    for hierarchical option composition.
    """

    def __init__(
        self,
        discount: float = 0.99,
        learning_rate: float = 0.1,
        random_seed: Optional[int] = None,
    ):
        self.discount = discount
        self.learning_rate = learning_rate
        self._rng = np.random.default_rng(random_seed)

        self._options: Dict[str, Option] = {}
        self._option_hierarchy: Dict[str, List[str]] = {}
        self._learner = IntraOptionLearning(discount, learning_rate)

        self._active_option: Optional[str] = None
        self._option_stack: List[str] = []

        self._total_options_executed = 0
        self._total_steps = 0

    def create_option(
        self,
        name: str,
        initiation_set: Callable[[Any], bool],
        policy: Callable[[Any], Any],
        termination: Callable[[Any], bool],
        termination_prob: Optional[Callable[[Any], float]] = None,
    ) -> Option:
        """
        Create and register a new option.

        Args:
            name: Unique option name
            initiation_set: Function returning True if option can start in state
            policy: Function mapping states to actions
            termination: Function returning True if option should terminate
            termination_prob: Optional soft termination probability

        Returns:
            Created Option instance
        """
        option_policy = OptionPolicy(name=f"{name}_policy", policy_fn=policy)
        term_condition = TerminationCondition(
            name=f"{name}_termination",
            predicate=termination,
            probability=termination_prob or (lambda s: 0.0),
        )

        option = Option(
            name=name,
            initiation_set=initiation_set,
            policy=option_policy,
            termination=term_condition,
        )

        self._options[name] = option
        return option

    def get_option(self, name: str) -> Optional[Option]:
        """Get option by name."""
        return self._options.get(name)

    def get_available_options(self, state: Any) -> List[Option]:
        """Get all options that can be initiated in state."""
        return [o for o in self._options.values() if o.can_initiate(state)]

    def select_option(
        self,
        state: Any,
        epsilon: float = 0.1,
        available: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Select an option using epsilon-greedy over option values.

        Args:
            state: Current state
            epsilon: Exploration probability
            available: Optional list of option names to consider

        Returns:
            Name of selected option or None
        """
        if available is None:
            options = self.get_available_options(state)
        else:
            options = [self._options[n] for n in available if n in self._options]

        if not options:
            return None

        if self._rng.random() < epsilon:
            return self._rng.choice(options).name

        best_value = float("-inf")
        best_option = None

        for option in options:
            value = self._learner.get_option_value(option.name, state)
            if value > best_value:
                best_value = value
                best_option = option.name

        return best_option or options[0].name

    def initiate_option(self, option_name: str, state: Any) -> bool:
        """Initiate an option."""
        option = self._options.get(option_name)
        if option is None:
            return False

        if option.initiate(state):
            self._active_option = option_name
            self._option_stack.append(option_name)
            self._total_options_executed += 1
            return True

        return False

    def step_option(self, state: Any) -> Tuple[Optional[Any], bool]:
        """
        Execute one step of the currently active option.

        Returns (action, terminated).
        """
        if self._active_option is None:
            return None, True

        option = self._options[self._active_option]
        action, terminated = option.step(state, self._rng)

        self._total_steps += 1

        if terminated:
            self._active_option = None
            if self._option_stack:
                self._option_stack.pop()
            if self._option_stack:
                self._active_option = self._option_stack[-1]

        return action, terminated

    def update_from_transition(
        self,
        state: Any,
        action: Any,
        reward: float,
        next_state: Any,
        terminated: bool,
    ) -> float:
        """
        Update option learning from a transition.

        Returns TD error.
        """
        if self._active_option is None:
            return 0.0

        option = self._options[self._active_option]
        option.add_reward(reward, self.discount)

        return self._learner.update(
            option, state, action, reward, next_state, terminated
        )

    def option_value(self, option_name: str, state: Any) -> float:
        """Get the value of executing an option in a state."""
        return self._learner.get_option_value(option_name, state)

    def set_option_value(self, option_name: str, state: Any, value: float) -> None:
        """Set the value of an option in a state."""
        self._learner.update_option_value(option_name, state, value)

    def add_option_hierarchy(self, parent: str, children: List[str]) -> None:
        """Define hierarchical relationship between options."""
        self._option_hierarchy[parent] = children

    def get_child_options(self, option_name: str) -> List[str]:
        """Get child options in hierarchy."""
        return self._option_hierarchy.get(option_name, [])

    def get_active_option(self) -> Optional[str]:
        """Get currently active option name."""
        return self._active_option

    def get_option_stack(self) -> List[str]:
        """Get current option execution stack."""
        return list(self._option_stack)

    def reset(self) -> None:
        """Reset all options to inactive state."""
        for option in self._options.values():
            option.reset()
        self._active_option = None
        self._option_stack = []

    def list_options(self) -> List[str]:
        """List all registered option names."""
        return list(self._options.keys())

    def statistics(self) -> Dict[str, Any]:
        """Get framework statistics."""
        return {
            "total_options": len(self._options),
            "total_options_executed": self._total_options_executed,
            "total_steps": self._total_steps,
            "active_option": self._active_option,
            "stack_depth": len(self._option_stack),
            "learner_stats": self._learner.statistics(),
        }
