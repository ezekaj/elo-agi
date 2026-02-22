"""
Imagination: Mental simulation and trajectory rollout.

The imagination module enables mental simulation of potential futures
before taking action. This is essential for planning, problem-solving,
and creative thinking.

Based on:
- Simulation theory of cognition
- Model-based reinforcement learning
- arXiv:2508.12752 - Simulate Before Act framework
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
import numpy as np
import time

from .state_encoder import EncodedState, StateEncoder
from .transition_model import TransitionModel, Transition


class RolloutStrategy(Enum):
    """Strategies for imagination rollout."""

    GREEDY = "greedy"  # Always take best action
    EPSILON_GREEDY = "epsilon"  # Mostly best, sometimes random
    SAMPLING = "sampling"  # Sample from policy
    RANDOM = "random"  # Random actions
    GOAL_DIRECTED = "goal"  # Actions toward goal


@dataclass
class ImaginationParams:
    """Parameters for imagination."""

    max_horizon: int = 50  # Maximum rollout length
    n_rollouts: int = 10  # Number of parallel rollouts
    discount: float = 0.99  # Discount factor for rewards
    uncertainty_penalty: float = 0.1  # Penalty for uncertain predictions
    pruning_threshold: float = 0.8  # Prune branches above this uncertainty
    temperature: float = 1.0  # Temperature for action sampling


@dataclass
class Rollout:
    """A single imagination rollout."""

    initial_state: np.ndarray
    actions: List[np.ndarray]
    states: List[np.ndarray]
    rewards: List[float]
    uncertainties: List[float]
    total_return: float
    final_state: np.ndarray
    was_pruned: bool = False
    prune_reason: str = ""
    timestamp: float = field(default_factory=time.time)

    @property
    def length(self) -> int:
        return len(self.actions)

    @property
    def avg_uncertainty(self) -> float:
        return float(np.mean(self.uncertainties)) if self.uncertainties else 0.0


@dataclass
class Trajectory:
    """Collection of rollouts for analysis."""

    rollouts: List[Rollout]
    best_rollout: Optional[Rollout]
    best_return: float
    avg_return: float
    success_rate: float  # Fraction not pruned
    timestamp: float = field(default_factory=time.time)


class Imagination:
    """
    System for mental simulation and planning.

    The imagination module uses the learned world model to simulate
    potential futures and evaluate different action sequences. Key features:

    1. **Multi-step rollout**: Can imagine far into the future
    2. **Uncertainty tracking**: Knows when predictions become unreliable
    3. **Parallel rollouts**: Explores multiple futures simultaneously
    4. **Goal-directed**: Can simulate toward specific goals
    5. **Pruning**: Stops simulating unpromising branches

    This implements the "Simulate Before Act" paradigm where agents
    mentally rehearse actions before execution.
    """

    def __init__(
        self,
        transition_model: TransitionModel,
        params: Optional[ImaginationParams] = None,
    ):
        self.transition_model = transition_model
        self.params = params or ImaginationParams()

        # Action policy (for generating actions during imagination)
        self._action_dim = transition_model.params.n_action
        self._action_policy: Optional[Callable] = None

        # Goal state for goal-directed imagination
        self._goal_state: Optional[np.ndarray] = None

        # Value function for evaluating states (optional)
        self._value_function: Optional[Callable] = None

        # History
        self._rollout_history: List[Rollout] = []
        self._trajectory_history: List[Trajectory] = []

    def set_policy(self, policy: Callable[[np.ndarray], np.ndarray]) -> None:
        """Set action policy for imagination."""
        self._action_policy = policy

    def set_goal(self, goal_state: np.ndarray) -> None:
        """Set goal state for goal-directed imagination."""
        self._goal_state = goal_state

    def set_value_function(self, value_fn: Callable[[np.ndarray], float]) -> None:
        """Set value function for state evaluation."""
        self._value_function = value_fn

    def imagine(
        self,
        initial_state: np.ndarray,
        action_sequence: Optional[List[np.ndarray]] = None,
        horizon: Optional[int] = None,
        strategy: RolloutStrategy = RolloutStrategy.SAMPLING,
    ) -> Rollout:
        """
        Perform a single imagination rollout.

        Args:
            initial_state: Starting state
            action_sequence: Optional fixed action sequence
            horizon: Rollout length (default: max_horizon)
            strategy: Action selection strategy

        Returns:
            Rollout with simulated trajectory
        """
        horizon = horizon or self.params.max_horizon
        current_state = initial_state.copy()

        actions = []
        states = [current_state.copy()]
        rewards = []
        uncertainties = []
        total_return = 0.0
        was_pruned = False
        prune_reason = ""

        for t in range(horizon):
            # Get action
            if action_sequence is not None and t < len(action_sequence):
                action = action_sequence[t]
            else:
                action = self._select_action(current_state, strategy)

            # Predict next state
            transition = self.transition_model.predict(current_state, action)

            actions.append(action)
            states.append(transition.predicted_state)
            rewards.append(transition.predicted_reward)
            uncertainties.append(transition.uncertainty)

            # Compute discounted return
            discount = self.params.discount**t
            adjusted_reward = transition.predicted_reward
            adjusted_reward -= self.params.uncertainty_penalty * transition.uncertainty
            total_return += discount * adjusted_reward

            # Check for pruning
            if transition.uncertainty > self.params.pruning_threshold:
                was_pruned = True
                prune_reason = f"uncertainty={transition.uncertainty:.3f}"
                break

            # Check for goal achievement
            if self._goal_state is not None:
                distance = np.linalg.norm(transition.predicted_state - self._goal_state)
                if distance < 0.1:
                    break  # Goal achieved

            current_state = transition.predicted_state

        # Add terminal value if value function available
        if self._value_function is not None and not was_pruned:
            terminal_value = self._value_function(current_state)
            discount = self.params.discount ** len(actions)
            total_return += discount * terminal_value

        rollout = Rollout(
            initial_state=initial_state,
            actions=actions,
            states=states,
            rewards=rewards,
            uncertainties=uncertainties,
            total_return=total_return,
            final_state=current_state,
            was_pruned=was_pruned,
            prune_reason=prune_reason,
        )

        self._rollout_history.append(rollout)
        if len(self._rollout_history) > 10000:
            self._rollout_history.pop(0)

        return rollout

    def _select_action(
        self,
        state: np.ndarray,
        strategy: RolloutStrategy,
    ) -> np.ndarray:
        """Select action based on strategy."""
        if strategy == RolloutStrategy.RANDOM:
            return np.random.randn(self._action_dim)

        elif strategy == RolloutStrategy.GREEDY:
            if self._action_policy is not None:
                return self._action_policy(state)
            return np.zeros(self._action_dim)

        elif strategy == RolloutStrategy.EPSILON_GREEDY:
            if np.random.rand() < 0.1 or self._action_policy is None:
                return np.random.randn(self._action_dim)
            return self._action_policy(state)

        elif strategy == RolloutStrategy.SAMPLING:
            if self._action_policy is not None:
                base_action = self._action_policy(state)
                noise = np.random.randn(self._action_dim) * self.params.temperature * 0.1
                return base_action + noise
            return np.random.randn(self._action_dim) * self.params.temperature

        elif strategy == RolloutStrategy.GOAL_DIRECTED:
            if self._goal_state is not None:
                # Simple goal-directed: action toward goal
                direction = self._goal_state[: self._action_dim] - state[: self._action_dim]
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm
                return direction
            return np.random.randn(self._action_dim)

        return np.random.randn(self._action_dim)

    def imagine_multiple(
        self,
        initial_state: np.ndarray,
        n_rollouts: Optional[int] = None,
        horizon: Optional[int] = None,
        strategy: RolloutStrategy = RolloutStrategy.SAMPLING,
    ) -> Trajectory:
        """
        Perform multiple parallel imagination rollouts.

        Returns trajectory analysis with best rollout.
        """
        n_rollouts = n_rollouts or self.params.n_rollouts
        horizon = horizon or self.params.max_horizon

        rollouts = []
        for _ in range(n_rollouts):
            rollout = self.imagine(initial_state, horizon=horizon, strategy=strategy)
            rollouts.append(rollout)

        # Find best rollout
        successful_rollouts = [r for r in rollouts if not r.was_pruned]
        if successful_rollouts:
            best_rollout = max(successful_rollouts, key=lambda r: r.total_return)
            best_return = best_rollout.total_return
        else:
            best_rollout = max(rollouts, key=lambda r: r.total_return)
            best_return = best_rollout.total_return

        # Statistics
        avg_return = float(np.mean([r.total_return for r in rollouts]))
        success_rate = len(successful_rollouts) / len(rollouts)

        trajectory = Trajectory(
            rollouts=rollouts,
            best_rollout=best_rollout,
            best_return=best_return,
            avg_return=avg_return,
            success_rate=success_rate,
        )

        self._trajectory_history.append(trajectory)
        if len(self._trajectory_history) > 1000:
            self._trajectory_history.pop(0)

        return trajectory

    def plan(
        self,
        initial_state: np.ndarray,
        horizon: int = 10,
        n_candidates: int = 100,
    ) -> List[np.ndarray]:
        """
        Plan optimal action sequence via imagination.

        Uses random shooting with trajectory evaluation.
        """
        best_return = float("-inf")
        best_actions = []

        for _ in range(n_candidates):
            # Generate random action sequence
            actions = [np.random.randn(self._action_dim) for _ in range(horizon)]

            # Evaluate via imagination
            rollout = self.imagine(initial_state, action_sequence=actions)

            if rollout.total_return > best_return:
                best_return = rollout.total_return
                best_actions = actions.copy()

        return best_actions

    def mpc_step(
        self,
        current_state: np.ndarray,
        horizon: int = 10,
        n_candidates: int = 50,
    ) -> np.ndarray:
        """
        Model Predictive Control step.

        Plans over horizon but only returns first action.
        """
        best_actions = self.plan(current_state, horizon, n_candidates)
        if best_actions:
            return best_actions[0]
        return np.zeros(self._action_dim)

    def counterfactual_imagine(
        self,
        initial_state: np.ndarray,
        actual_actions: List[np.ndarray],
        alternative_action: np.ndarray,
        action_index: int,
    ) -> Tuple[Rollout, Rollout]:
        """
        Compare actual trajectory with counterfactual.

        Returns:
            Tuple of (actual_rollout, counterfactual_rollout)
        """
        # Actual trajectory
        actual = self.imagine(initial_state, action_sequence=actual_actions)

        # Counterfactual: replace action at index
        counterfactual_actions = actual_actions.copy()
        counterfactual_actions[action_index] = alternative_action
        counterfactual = self.imagine(initial_state, action_sequence=counterfactual_actions)

        return actual, counterfactual

    def get_statistics(self) -> Dict[str, Any]:
        """Get imagination statistics."""
        if not self._rollout_history:
            return {
                "n_rollouts": 0,
                "avg_return": 0.0,
                "avg_length": 0.0,
                "prune_rate": 0.0,
            }

        recent = self._rollout_history[-1000:]
        pruned = [r for r in recent if r.was_pruned]

        return {
            "n_rollouts": len(self._rollout_history),
            "avg_return": float(np.mean([r.total_return for r in recent])),
            "avg_length": float(np.mean([r.length for r in recent])),
            "avg_uncertainty": float(np.mean([r.avg_uncertainty for r in recent])),
            "prune_rate": len(pruned) / len(recent),
            "n_trajectories": len(self._trajectory_history),
        }

    def reset(self) -> None:
        """Reset imagination history."""
        self._rollout_history = []
        self._trajectory_history = []
        self._goal_state = None
