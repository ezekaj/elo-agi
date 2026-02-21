"""
Goal Hierarchy and MAXQ Decomposition

Implements hierarchical goal decomposition based on the MAXQ framework,
enabling multi-level task decomposition and value function decomposition.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Set, Tuple
from enum import Enum
import numpy as np


class GoalStatus(Enum):
    """Status of a goal in the hierarchy."""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class Goal:
    """A goal with associated metadata."""
    name: str
    description: str = ""
    preconditions: List[Callable[[Any], bool]] = field(default_factory=list)
    effects: List[Callable[[Any], Any]] = field(default_factory=list)
    priority: float = 1.0
    deadline: Optional[float] = None

    def check_preconditions(self, state: Any) -> bool:
        """Check if all preconditions are satisfied."""
        return all(p(state) for p in self.preconditions)

    def apply_effects(self, state: Any) -> Any:
        """Apply goal effects to state."""
        for effect in self.effects:
            state = effect(state)
        return state


@dataclass
class GoalNode:
    """A node in the goal tree hierarchy."""
    goal: Goal
    children: List["GoalNode"] = field(default_factory=list)
    parent: Optional["GoalNode"] = None
    status: GoalStatus = GoalStatus.PENDING
    value: float = 0.0
    completion: float = 0.0

    @property
    def is_primitive(self) -> bool:
        """Check if this is a primitive (leaf) goal."""
        return len(self.children) == 0

    @property
    def is_composite(self) -> bool:
        """Check if this is a composite (internal) goal."""
        return len(self.children) > 0

    def add_child(self, child: "GoalNode") -> None:
        """Add a child goal."""
        child.parent = self
        self.children.append(child)

    def get_depth(self) -> int:
        """Get depth in the tree."""
        depth = 0
        node = self
        while node.parent is not None:
            depth += 1
            node = node.parent
        return depth

    def get_ancestors(self) -> List["GoalNode"]:
        """Get all ancestor nodes."""
        ancestors = []
        node = self.parent
        while node is not None:
            ancestors.append(node)
            node = node.parent
        return ancestors

    def get_descendants(self) -> List["GoalNode"]:
        """Get all descendant nodes."""
        descendants = []
        for child in self.children:
            descendants.append(child)
            descendants.extend(child.get_descendants())
        return descendants


@dataclass
class GoalTree:
    """A hierarchical goal tree."""
    root: GoalNode
    nodes: Dict[str, GoalNode] = field(default_factory=dict)

    def __post_init__(self):
        self._index_nodes(self.root)

    def _index_nodes(self, node: GoalNode) -> None:
        """Index all nodes by name."""
        self.nodes[node.goal.name] = node
        for child in node.children:
            self._index_nodes(child)

    def get_node(self, name: str) -> Optional[GoalNode]:
        """Get a node by goal name."""
        return self.nodes.get(name)

    def get_active_goals(self) -> List[GoalNode]:
        """Get all currently active goals."""
        return [n for n in self.nodes.values() if n.status == GoalStatus.ACTIVE]

    def get_primitive_goals(self) -> List[GoalNode]:
        """Get all primitive (leaf) goals."""
        return [n for n in self.nodes.values() if n.is_primitive]

    def get_executable_goals(self, state: Any) -> List[GoalNode]:
        """Get primitive goals whose preconditions are satisfied."""
        executable = []
        for node in self.get_primitive_goals():
            if node.status == GoalStatus.PENDING and node.goal.check_preconditions(state):
                executable.append(node)
        return executable

    def update_completion(self) -> None:
        """Update completion status bottom-up."""
        self._update_node_completion(self.root)

    def _update_node_completion(self, node: GoalNode) -> float:
        """Recursively update node completion."""
        if node.is_primitive:
            return node.completion

        child_completions = [self._update_node_completion(c) for c in node.children]
        node.completion = np.mean(child_completions) if child_completions else 0.0
        return node.completion

    def statistics(self) -> Dict[str, Any]:
        """Get tree statistics."""
        return {
            "total_nodes": len(self.nodes),
            "primitive_goals": len(self.get_primitive_goals()),
            "active_goals": len(self.get_active_goals()),
            "root_completion": self.root.completion,
            "max_depth": max(n.get_depth() for n in self.nodes.values()),
        }


@dataclass
class CompletionFunction:
    """Completion function for a goal."""
    goal_name: str
    function: Callable[[Any], float]
    learned_params: Optional[np.ndarray] = None

    def evaluate(self, state: Any) -> float:
        """Evaluate completion in state."""
        return self.function(state)

    def update(self, state: Any, target: float, learning_rate: float = 0.01) -> float:
        """Update learned parameters if applicable."""
        if self.learned_params is not None:
            current = self.evaluate(state)
            error = target - current
            return error
        return 0.0


class MAXQDecomposition:
    """
    MAXQ-style hierarchical task decomposition.

    Implements value function decomposition where:
    V(i, s) = V(a, s) + C(i, s, a)

    V(i, s) = value of executing subtask i in state s
    V(a, s) = value of primitive action a
    C(i, s, a) = completion function (expected reward after a completes)
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

        self._goal_tree: Optional[GoalTree] = None
        self._completion_functions: Dict[str, CompletionFunction] = {}
        self._value_functions: Dict[str, Dict[str, float]] = {}
        self._policy: Dict[str, Dict[str, str]] = {}

        self._total_decompositions = 0
        self._total_value_updates = 0

    def decompose(self, root_goal: Goal, subtasks: Dict[str, List[str]]) -> GoalTree:
        """
        Decompose a root goal into a hierarchy.

        Args:
            root_goal: The top-level goal
            subtasks: Dict mapping goal names to list of subtask names

        Returns:
            GoalTree representing the hierarchy
        """
        goals_by_name = {root_goal.name: root_goal}

        def build_subtask_goals(goal_name: str, depth: int = 0) -> List[Goal]:
            if goal_name not in subtasks:
                return []

            child_goals = []
            for subtask_name in subtasks[goal_name]:
                if subtask_name not in goals_by_name:
                    child_goal = Goal(
                        name=subtask_name,
                        description=f"Subtask of {goal_name}",
                        priority=goals_by_name[goal_name].priority * 0.9,
                    )
                    goals_by_name[subtask_name] = child_goal
                child_goals.append(goals_by_name[subtask_name])
            return child_goals

        def build_tree(goal: Goal) -> GoalNode:
            node = GoalNode(goal=goal)
            child_goals = build_subtask_goals(goal.name)
            for child_goal in child_goals:
                child_node = build_tree(child_goal)
                node.add_child(child_node)
            return node

        root_node = build_tree(root_goal)
        self._goal_tree = GoalTree(root=root_node)
        self._total_decompositions += 1

        for name in self._goal_tree.nodes:
            if name not in self._completion_functions:
                self._completion_functions[name] = CompletionFunction(
                    goal_name=name,
                    function=lambda s, n=name: 0.0,
                )
            if name not in self._value_functions:
                self._value_functions[name] = {}
            if name not in self._policy:
                self._policy[name] = {}

        return self._goal_tree

    def compute_completion_function(
        self,
        goal_name: str,
        state: Any,
        state_key: Optional[str] = None,
    ) -> float:
        """
        Compute completion function C(i, s, a).

        The completion function represents the expected cumulative
        reward after completing subtask a while executing parent task i.
        """
        if goal_name not in self._completion_functions:
            return 0.0

        return self._completion_functions[goal_name].evaluate(state)

    def get_value(
        self,
        goal_name: str,
        state: Any,
        state_key: Optional[str] = None,
    ) -> float:
        """
        Get the value V(i, s) of executing goal i in state s.

        For primitive goals: V(a, s) = expected immediate reward
        For composite goals: V(i, s) = max_a[V(a, s) + C(i, s, a)]
        """
        if self._goal_tree is None:
            return 0.0

        node = self._goal_tree.get_node(goal_name)
        if node is None:
            return 0.0

        if state_key is None:
            state_key = str(hash(str(state)))

        if node.is_primitive:
            return self._value_functions.get(goal_name, {}).get(state_key, 0.0)

        max_value = float("-inf")
        for child in node.children:
            child_value = self.get_value(child.goal.name, state, state_key)
            completion = self.compute_completion_function(goal_name, state, state_key)
            total_value = child_value + self.discount * completion
            max_value = max(max_value, total_value)

        return max_value if max_value > float("-inf") else 0.0

    def select_subtask(
        self,
        goal_name: str,
        state: Any,
        epsilon: float = 0.1,
    ) -> Optional[str]:
        """
        Select a subtask using epsilon-greedy policy.

        Returns the name of the selected child subtask.
        """
        if self._goal_tree is None:
            return None

        node = self._goal_tree.get_node(goal_name)
        if node is None or node.is_primitive:
            return None

        if self._rng.random() < epsilon:
            child = self._rng.choice(node.children)
            return child.goal.name

        best_value = float("-inf")
        best_child = None

        for child in node.children:
            value = self.get_value(child.goal.name, state)
            if value > best_value:
                best_value = value
                best_child = child.goal.name

        return best_child or (node.children[0].goal.name if node.children else None)

    def update_value(
        self,
        goal_name: str,
        state: Any,
        reward: float,
        next_state: Any,
        state_key: Optional[str] = None,
        next_state_key: Optional[str] = None,
    ) -> float:
        """
        Update value function using TD learning.

        Returns the TD error.
        """
        if state_key is None:
            state_key = str(hash(str(state)))
        if next_state_key is None:
            next_state_key = str(hash(str(next_state)))

        if goal_name not in self._value_functions:
            self._value_functions[goal_name] = {}

        current_value = self._value_functions[goal_name].get(state_key, 0.0)
        next_value = self.get_value(goal_name, next_state, next_state_key)

        td_target = reward + self.discount * next_value
        td_error = td_target - current_value

        self._value_functions[goal_name][state_key] = (
            current_value + self.learning_rate * td_error
        )

        self._total_value_updates += 1

        return td_error

    def update_completion_function(
        self,
        goal_name: str,
        child_name: str,
        state: Any,
        cumulative_reward: float,
    ) -> None:
        """
        Update completion function after executing a child subtask.

        C(i, s, a) <- C(i, s, a) + alpha * (reward - C(i, s, a))
        """
        if goal_name not in self._completion_functions:
            return

        cf = self._completion_functions[goal_name]
        error = cf.update(state, cumulative_reward, self.learning_rate)

    def get_hierarchical_policy(
        self,
        state: Any,
        top_goal: Optional[str] = None,
    ) -> List[Tuple[str, str]]:
        """
        Get the full hierarchical policy from top to primitive action.

        Returns list of (goal, selected_subtask) pairs.
        """
        if self._goal_tree is None:
            return []

        policy_trace = []
        current_goal = top_goal or self._goal_tree.root.goal.name

        while True:
            node = self._goal_tree.get_node(current_goal)
            if node is None or node.is_primitive:
                break

            selected = self.select_subtask(current_goal, state, epsilon=0.0)
            if selected is None:
                break

            policy_trace.append((current_goal, selected))
            current_goal = selected

        return policy_trace

    def set_completion_function(
        self,
        goal_name: str,
        function: Callable[[Any], float],
    ) -> None:
        """Set a custom completion function for a goal."""
        self._completion_functions[goal_name] = CompletionFunction(
            goal_name=goal_name,
            function=function,
        )

    def get_goal_tree(self) -> Optional[GoalTree]:
        """Get the current goal tree."""
        return self._goal_tree

    def statistics(self) -> Dict[str, Any]:
        """Get decomposition statistics."""
        return {
            "total_decompositions": self._total_decompositions,
            "total_value_updates": self._total_value_updates,
            "goals_tracked": len(self._value_functions),
            "completion_functions": len(self._completion_functions),
            "tree_nodes": len(self._goal_tree.nodes) if self._goal_tree else 0,
        }
