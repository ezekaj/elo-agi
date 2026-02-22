"""
Planning Search with Hierarchical MCTS

Implements Monte Carlo Tree Search with:
- Hierarchical action abstraction
- World model imagination for rollouts
- Progressive widening for continuous actions
- UCB exploration with uncertainty bonuses
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple
from enum import Enum
import numpy as np


class NodeType(Enum):
    """Type of search node."""

    STATE = "state"
    ACTION = "action"
    OPTION = "option"


@dataclass
class SearchNode:
    """A node in the search tree."""

    node_type: NodeType
    state: Optional[np.ndarray] = None
    action: Optional[Any] = None
    option: Optional[str] = None

    parent: Optional["SearchNode"] = None
    children: List["SearchNode"] = field(default_factory=list)

    visit_count: int = 0
    value_sum: float = 0.0
    prior: float = 1.0
    depth: int = 0

    @property
    def value(self) -> float:
        """Average value of the node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    @property
    def is_leaf(self) -> bool:
        """Check if node is a leaf."""
        return len(self.children) == 0

    def ucb_score(
        self,
        exploration_weight: float = 1.41,
        parent_visits: int = 1,
    ) -> float:
        """
        Compute UCB1 score for selection.

        UCB = Q(s,a) + c * sqrt(ln(N) / n)
        """
        if self.visit_count == 0:
            return float("inf")

        exploitation = self.value
        exploration = exploration_weight * np.sqrt(np.log(parent_visits + 1) / self.visit_count)

        return exploitation + exploration * self.prior

    def add_child(self, child: "SearchNode") -> None:
        """Add a child node."""
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)

    def backpropagate(self, value: float, discount: float = 0.99) -> None:
        """Backpropagate value up the tree."""
        node = self
        depth = 0

        while node is not None:
            node.visit_count += 1
            discounted_value = value * (discount**depth)
            node.value_sum += discounted_value
            node = node.parent
            depth += 1


@dataclass
class MCTSConfig:
    """Configuration for MCTS."""

    num_simulations: int = 100
    exploration_weight: float = 1.41
    discount: float = 0.99
    max_depth: int = 50
    progressive_widening_alpha: float = 0.5
    use_uncertainty_bonus: bool = True
    uncertainty_weight: float = 0.5
    imagination_horizon: int = 10
    random_seed: Optional[int] = None


@dataclass
class PlanResult:
    """Result of planning."""

    actions: List[Any]
    options: List[str]
    expected_value: float
    confidence: float
    tree_depth: int
    simulations_run: int
    nodes_expanded: int
    planning_time: float = 0.0


class WorldModelAdapter:
    """
    Adapter for world model imagination.

    Provides interface for MCTS to use world model predictions.
    """

    def __init__(
        self,
        transition_fn: Optional[Callable[[np.ndarray, Any], np.ndarray]] = None,
        reward_fn: Optional[Callable[[np.ndarray, Any, np.ndarray], float]] = None,
        terminal_fn: Optional[Callable[[np.ndarray], bool]] = None,
        uncertainty_fn: Optional[Callable[[np.ndarray, Any], float]] = None,
        random_seed: Optional[int] = None,
    ):
        self._rng = np.random.default_rng(random_seed)

        if transition_fn is None:
            self.transition_fn = lambda s, a: s + self._rng.normal(0, 0.1, s.shape)
        else:
            self.transition_fn = transition_fn

        if reward_fn is None:
            self.reward_fn = lambda s, a, ns: 0.0
        else:
            self.reward_fn = reward_fn

        if terminal_fn is None:
            self.terminal_fn = lambda s: False
        else:
            self.terminal_fn = terminal_fn

        if uncertainty_fn is None:
            self.uncertainty_fn = lambda s, a: 0.1
        else:
            self.uncertainty_fn = uncertainty_fn

    def imagine_step(
        self,
        state: np.ndarray,
        action: Any,
    ) -> Tuple[np.ndarray, float, bool, float]:
        """
        Imagine one step forward.

        Returns (next_state, reward, terminal, uncertainty).
        """
        next_state = self.transition_fn(state, action)
        reward = self.reward_fn(state, action, next_state)
        terminal = self.terminal_fn(next_state)
        uncertainty = self.uncertainty_fn(state, action)

        return next_state, reward, terminal, uncertainty

    def imagine_trajectory(
        self,
        state: np.ndarray,
        actions: List[Any],
    ) -> Tuple[List[np.ndarray], List[float], bool]:
        """
        Imagine a full trajectory.

        Returns (states, rewards, reached_terminal).
        """
        states = [state]
        rewards = []
        terminal = False

        current = state
        for action in actions:
            next_state, reward, terminal, _ = self.imagine_step(current, action)
            states.append(next_state)
            rewards.append(reward)
            current = next_state

            if terminal:
                break

        return states, rewards, terminal


class HierarchicalMCTS:
    """
    Monte Carlo Tree Search with hierarchical actions.

    Supports both primitive actions and options (temporal abstractions).
    Uses world model imagination for rollouts.
    """

    def __init__(
        self,
        config: Optional[MCTSConfig] = None,
        world_model: Optional[WorldModelAdapter] = None,
        action_space: Optional[List[Any]] = None,
        option_names: Optional[List[str]] = None,
        option_policies: Optional[Dict[str, Callable[[np.ndarray], Any]]] = None,
        option_terminations: Optional[Dict[str, Callable[[np.ndarray], bool]]] = None,
    ):
        self.config = config or MCTSConfig()
        self._rng = np.random.default_rng(self.config.random_seed)

        self.world_model = world_model or WorldModelAdapter(random_seed=self.config.random_seed)

        self.action_space = action_space or [0, 1, 2, 3]
        self.option_names = option_names or []
        self.option_policies = option_policies or {}
        self.option_terminations = option_terminations or {}

        self._root: Optional[SearchNode] = None
        self._nodes_expanded = 0
        self._simulations_run = 0

    def search(
        self,
        root_state: np.ndarray,
        goal: Optional[np.ndarray] = None,
        num_simulations: Optional[int] = None,
    ) -> PlanResult:
        """
        Run MCTS from root state.

        Args:
            root_state: Starting state
            goal: Optional goal state for reward shaping
            num_simulations: Override config num_simulations

        Returns:
            PlanResult with best action sequence
        """
        self._root = SearchNode(
            node_type=NodeType.STATE,
            state=root_state.copy(),
        )
        self._nodes_expanded = 1
        self._simulations_run = 0

        n_sims = num_simulations or self.config.num_simulations

        for _ in range(n_sims):
            self._run_simulation(goal)
            self._simulations_run += 1

        actions, options, expected_value = self._extract_best_plan()

        confidence = self._compute_confidence()

        return PlanResult(
            actions=actions,
            options=options,
            expected_value=expected_value,
            confidence=confidence,
            tree_depth=self._get_tree_depth(),
            simulations_run=self._simulations_run,
            nodes_expanded=self._nodes_expanded,
        )

    def _run_simulation(self, goal: Optional[np.ndarray] = None) -> None:
        """Run one MCTS simulation."""
        node = self._root
        state = self._root.state.copy()
        path = [node]
        total_reward = 0.0
        depth = 0

        while not node.is_leaf and depth < self.config.max_depth:
            node = self._select_child(node)
            path.append(node)

            if node.node_type == NodeType.ACTION:
                next_state, reward, terminal, uncertainty = self.world_model.imagine_step(
                    state, node.action
                )
                total_reward += (self.config.discount**depth) * reward
                state = next_state
                depth += 1

                if terminal:
                    break

        if depth < self.config.max_depth and not self.world_model.terminal_fn(state):
            self._expand(node, state)

        rollout_value = self._simulate_rollout(state, goal, depth)
        total_value = total_reward + (self.config.discount**depth) * rollout_value

        node.backpropagate(total_value, self.config.discount)

    def _select_child(self, node: SearchNode) -> SearchNode:
        """Select child using UCB."""
        if not node.children:
            return node

        best_score = float("-inf")
        best_child = None

        for child in node.children:
            score = child.ucb_score(
                self.config.exploration_weight,
                node.visit_count,
            )

            if self.config.use_uncertainty_bonus and child.state is not None:
                uncertainty = self.world_model.uncertainty_fn(
                    node.state, child.action if child.action else 0
                )
                score += self.config.uncertainty_weight * uncertainty

            if score > best_score:
                best_score = score
                best_child = child

        return best_child or node.children[0]

    def _expand(self, node: SearchNode, state: np.ndarray) -> None:
        """Expand node with children."""
        n_children = int((node.visit_count + 1) ** self.config.progressive_widening_alpha)
        n_children = min(n_children, len(self.action_space) + len(self.option_names))
        n_children = max(1, n_children)

        for action in self.action_space[:n_children]:
            child = SearchNode(
                node_type=NodeType.ACTION,
                action=action,
                state=state.copy(),
            )
            node.add_child(child)
            self._nodes_expanded += 1

        for option_name in self.option_names[: max(0, n_children - len(self.action_space))]:
            child = SearchNode(
                node_type=NodeType.OPTION,
                option=option_name,
                state=state.copy(),
            )
            node.add_child(child)
            self._nodes_expanded += 1

    def _simulate_rollout(
        self,
        state: np.ndarray,
        goal: Optional[np.ndarray],
        current_depth: int,
    ) -> float:
        """Simulate random rollout from state."""
        total_reward = 0.0
        current = state.copy()
        depth = 0

        horizon = min(
            self.config.imagination_horizon,
            self.config.max_depth - current_depth,
        )

        for step in range(horizon):
            if self.world_model.terminal_fn(current):
                break

            action = self._rng.choice(self.action_space)
            next_state, reward, terminal, _ = self.world_model.imagine_step(current, action)

            if goal is not None:
                goal_distance = np.linalg.norm(next_state - goal)
                reward += -0.1 * goal_distance

            total_reward += (self.config.discount**step) * reward
            current = next_state
            depth += 1

            if terminal:
                break

        return total_reward

    def _extract_best_plan(self) -> Tuple[List[Any], List[str], float]:
        """Extract the best action sequence from tree."""
        actions = []
        options = []
        node = self._root

        while node and node.children:
            best_child = max(node.children, key=lambda c: c.visit_count)

            if best_child.node_type == NodeType.ACTION:
                actions.append(best_child.action)
            elif best_child.node_type == NodeType.OPTION:
                options.append(best_child.option)

            node = best_child

        expected_value = self._root.value if self._root else 0.0

        return actions, options, expected_value

    def _compute_confidence(self) -> float:
        """Compute planning confidence."""
        if not self._root or not self._root.children:
            return 0.0

        visit_counts = [c.visit_count for c in self._root.children]
        total = sum(visit_counts)

        if total == 0:
            return 0.0

        max_visits = max(visit_counts)
        return max_visits / total

    def _get_tree_depth(self) -> int:
        """Get maximum tree depth."""
        if not self._root:
            return 0

        max_depth = 0
        stack = [(self._root, 0)]

        while stack:
            node, depth = stack.pop()
            max_depth = max(max_depth, depth)
            for child in node.children:
                stack.append((child, depth + 1))

        return max_depth

    def replan(
        self,
        current_state: np.ndarray,
        previous_plan: PlanResult,
        actions_taken: int = 1,
    ) -> PlanResult:
        """
        Replan from current state, reusing previous tree if possible.

        Args:
            current_state: Current state after taking actions
            previous_plan: Previous planning result
            actions_taken: Number of actions taken since last plan

        Returns:
            New PlanResult
        """
        return self.search(current_state)

    def get_action_values(self) -> Dict[Any, float]:
        """Get value estimates for each root action."""
        if not self._root:
            return {}

        values = {}
        for child in self._root.children:
            if child.node_type == NodeType.ACTION:
                values[child.action] = child.value

        return values

    def get_option_values(self) -> Dict[str, float]:
        """Get value estimates for each root option."""
        if not self._root:
            return {}

        values = {}
        for child in self._root.children:
            if child.node_type == NodeType.OPTION:
                values[child.option] = child.value

        return values

    def statistics(self) -> Dict[str, Any]:
        """Get MCTS statistics."""
        return {
            "simulations_run": self._simulations_run,
            "nodes_expanded": self._nodes_expanded,
            "tree_depth": self._get_tree_depth(),
            "root_visits": self._root.visit_count if self._root else 0,
            "root_value": self._root.value if self._root else 0.0,
        }
