"""
Planning Integration

Integrates the planning module with other neuro modules:
- World model for imagination
- Executive control for goal maintenance
- Motivation for reward signals
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple
import numpy as np

from .goal_hierarchy import MAXQDecomposition, GoalTree, Goal
from .temporal_abstraction import OptionsFramework, Option
from .skill_library import SkillLibrary, Skill, SkillType
from .subgoal_discovery import SubgoalDiscovery, Subgoal, Trajectory
from .planning_search import HierarchicalMCTS, MCTSConfig, PlanResult, WorldModelAdapter


@dataclass
class PlanningConfig:
    """Configuration for the integrated planning system."""

    state_dim: int = 64
    action_dim: int = 4
    mcts_simulations: int = 100
    mcts_depth: int = 50
    discount: float = 0.99
    learning_rate: float = 0.1
    exploration_weight: float = 1.41
    skill_embedding_dim: int = 64
    random_seed: Optional[int] = None


class WorldModelAdapter:
    """
    Adapter connecting to external world model.

    Provides imagination capabilities for planning.
    """

    def __init__(
        self,
        transition_model: Optional[Callable[[np.ndarray, Any], np.ndarray]] = None,
        reward_model: Optional[Callable[[np.ndarray, Any, np.ndarray], float]] = None,
        terminal_model: Optional[Callable[[np.ndarray], bool]] = None,
        uncertainty_model: Optional[Callable[[np.ndarray, Any], float]] = None,
        random_seed: Optional[int] = None,
    ):
        self._rng = np.random.default_rng(random_seed)

        self._transition_model = transition_model
        self._reward_model = reward_model
        self._terminal_model = terminal_model
        self._uncertainty_model = uncertainty_model

        self._imagination_calls = 0

    def set_transition_model(self, model: Callable[[np.ndarray, Any], np.ndarray]) -> None:
        """Set the transition model."""
        self._transition_model = model

    def set_reward_model(self, model: Callable[[np.ndarray, Any, np.ndarray], float]) -> None:
        """Set the reward model."""
        self._reward_model = model

    def set_terminal_model(self, model: Callable[[np.ndarray], bool]) -> None:
        """Set the terminal state model."""
        self._terminal_model = model

    def set_uncertainty_model(self, model: Callable[[np.ndarray, Any], float]) -> None:
        """Set the uncertainty model."""
        self._uncertainty_model = model

    def predict_transition(
        self,
        state: np.ndarray,
        action: Any,
    ) -> np.ndarray:
        """Predict next state."""
        self._imagination_calls += 1

        if self._transition_model is not None:
            return self._transition_model(state, action)

        noise = self._rng.normal(0, 0.1, state.shape)
        return state + noise

    def predict_reward(
        self,
        state: np.ndarray,
        action: Any,
        next_state: np.ndarray,
    ) -> float:
        """Predict reward for transition."""
        if self._reward_model is not None:
            return self._reward_model(state, action, next_state)
        return 0.0

    def predict_terminal(self, state: np.ndarray) -> bool:
        """Predict if state is terminal."""
        if self._terminal_model is not None:
            return self._terminal_model(state)
        return False

    def predict_uncertainty(self, state: np.ndarray, action: Any) -> float:
        """Predict model uncertainty for state-action pair."""
        if self._uncertainty_model is not None:
            return self._uncertainty_model(state, action)
        return 0.1

    def imagine_trajectory(
        self,
        state: np.ndarray,
        actions: List[Any],
        max_steps: Optional[int] = None,
    ) -> Tuple[List[np.ndarray], List[float], bool]:
        """
        Imagine a full trajectory.

        Returns (states, rewards, reached_terminal).
        """
        states = [state.copy()]
        rewards = []
        current = state.copy()

        max_steps = max_steps or len(actions)

        for i, action in enumerate(actions[:max_steps]):
            next_state = self.predict_transition(current, action)
            reward = self.predict_reward(current, action, next_state)
            terminal = self.predict_terminal(next_state)

            states.append(next_state)
            rewards.append(reward)
            current = next_state

            if terminal:
                return states, rewards, True

        return states, rewards, False

    def statistics(self) -> Dict[str, Any]:
        """Get adapter statistics."""
        return {
            "imagination_calls": self._imagination_calls,
            "has_transition_model": self._transition_model is not None,
            "has_reward_model": self._reward_model is not None,
            "has_terminal_model": self._terminal_model is not None,
        }


class PlanningIntegration:
    """
    Integrated planning system combining all planning components.

    Provides a unified interface for:
    - Hierarchical goal decomposition
    - Temporal abstraction with options
    - Skill library management
    - Subgoal discovery
    - MCTS-based planning
    """

    def __init__(self, config: Optional[PlanningConfig] = None):
        self.config = config or PlanningConfig()
        self._rng = np.random.default_rng(self.config.random_seed)

        self._decomposition = MAXQDecomposition(
            discount=self.config.discount,
            learning_rate=self.config.learning_rate,
            random_seed=self.config.random_seed,
        )

        self._options = OptionsFramework(
            discount=self.config.discount,
            learning_rate=self.config.learning_rate,
            random_seed=self.config.random_seed,
        )

        self._skills = SkillLibrary(
            embedding_dim=self.config.skill_embedding_dim,
            random_seed=self.config.random_seed,
        )

        self._subgoal_discovery = SubgoalDiscovery(
            state_dim=self.config.state_dim,
            random_seed=self.config.random_seed,
        )

        self._world_model = WorldModelAdapter(random_seed=self.config.random_seed)

        self._current_goal: Optional[Goal] = None
        self._current_plan: Optional[PlanResult] = None
        self._plan_step: int = 0

        self._total_plans = 0
        self._successful_plans = 0

    def set_world_model(
        self,
        transition_fn: Optional[Callable] = None,
        reward_fn: Optional[Callable] = None,
        terminal_fn: Optional[Callable] = None,
        uncertainty_fn: Optional[Callable] = None,
    ) -> None:
        """Configure the world model for imagination."""
        if transition_fn:
            self._world_model.set_transition_model(transition_fn)
        if reward_fn:
            self._world_model.set_reward_model(reward_fn)
        if terminal_fn:
            self._world_model.set_terminal_model(terminal_fn)
        if uncertainty_fn:
            self._world_model.set_uncertainty_model(uncertainty_fn)

    def set_goal(
        self,
        goal: Goal,
        decomposition: Optional[Dict[str, List[str]]] = None,
    ) -> GoalTree:
        """
        Set the current goal and optionally decompose it.

        Args:
            goal: Top-level goal
            decomposition: Optional task decomposition

        Returns:
            Goal tree
        """
        self._current_goal = goal

        if decomposition:
            goal_tree = self._decomposition.decompose(goal, decomposition)
        else:
            goal_tree = self._decomposition.decompose(goal, {})

        return goal_tree

    def plan(
        self,
        state: np.ndarray,
        goal_state: Optional[np.ndarray] = None,
        use_skills: bool = True,
        use_options: bool = True,
    ) -> PlanResult:
        """
        Generate a plan from the current state.

        Args:
            state: Current state
            goal_state: Optional explicit goal state
            use_skills: Whether to include skills in action space
            use_options: Whether to include options in action space

        Returns:
            PlanResult with action sequence
        """
        action_space = list(range(self.config.action_dim))

        option_names = []
        option_policies = {}
        option_terminations = {}

        if use_options:
            for name in self._options.list_options():
                option = self._options.get_option(name)
                if option and option.can_initiate(state):
                    option_names.append(name)
                    option_policies[name] = option.policy.select_action
                    option_terminations[name] = option.termination.should_terminate

        if use_skills:
            applicable_skills = self._skills.retrieve_applicable(state)
            for skill in applicable_skills[:5]:
                if skill.name not in option_names:
                    option_names.append(skill.name)
                    option_policies[skill.name] = lambda s, sk=skill: sk.execute_step(s)[0]
                    option_terminations[skill.name] = lambda s, sk=skill: sk.termination(s, None)

        mcts_config = MCTSConfig(
            num_simulations=self.config.mcts_simulations,
            exploration_weight=self.config.exploration_weight,
            discount=self.config.discount,
            max_depth=self.config.mcts_depth,
            random_seed=self.config.random_seed,
        )

        from .planning_search import WorldModelAdapter as MCTSWorldModel

        mcts_world_model = MCTSWorldModel(
            transition_fn=self._world_model.predict_transition,
            reward_fn=self._world_model.predict_reward,
            terminal_fn=self._world_model.predict_terminal,
            uncertainty_fn=self._world_model.predict_uncertainty,
            random_seed=self.config.random_seed,
        )

        mcts = HierarchicalMCTS(
            config=mcts_config,
            world_model=mcts_world_model,
            action_space=action_space,
            option_names=option_names,
            option_policies=option_policies,
            option_terminations=option_terminations,
        )

        result = mcts.search(state, goal=goal_state)

        self._current_plan = result
        self._plan_step = 0
        self._total_plans += 1

        return result

    def get_action(self, state: np.ndarray) -> Optional[Any]:
        """
        Get the next action from the current plan.

        Returns None if no plan or plan exhausted.
        """
        if self._current_plan is None:
            return None

        if self._plan_step >= len(self._current_plan.actions):
            return None

        action = self._current_plan.actions[self._plan_step]
        self._plan_step += 1

        return action

    def should_replan(
        self,
        state: np.ndarray,
        expected_state: Optional[np.ndarray] = None,
        threshold: float = 0.5,
    ) -> bool:
        """
        Check if replanning is needed.

        Args:
            state: Current actual state
            expected_state: State we expected to be in
            threshold: Deviation threshold for replanning

        Returns:
            True if replanning is recommended
        """
        if self._current_plan is None:
            return True

        if self._plan_step >= len(self._current_plan.actions):
            return True

        if expected_state is not None:
            deviation = np.linalg.norm(state - expected_state)
            if deviation > threshold:
                return True

        if self._current_plan.confidence < 0.3:
            return True

        return False

    def register_option(
        self,
        name: str,
        initiation: Callable[[Any], bool],
        policy: Callable[[Any], Any],
        termination: Callable[[Any], bool],
    ) -> Option:
        """Register an option for planning."""
        return self._options.create_option(
            name=name,
            initiation_set=initiation,
            policy=policy,
            termination=termination,
        )

    def register_skill(self, skill: Skill) -> str:
        """Register a skill for planning."""
        return self._skills.register_skill(skill)

    def discover_subgoals(
        self,
        trajectories: List[Trajectory],
        methods: Optional[List[str]] = None,
    ) -> List[Subgoal]:
        """Discover subgoals from experience."""
        return self._subgoal_discovery.discover_subgoals(trajectories, methods=methods)

    def learn_from_trajectory(
        self,
        trajectory: Trajectory,
        learn_subgoals: bool = True,
        learn_options: bool = True,
    ) -> Dict[str, Any]:
        """
        Learn from a trajectory.

        Updates option values, discovers subgoals, and updates decomposition.
        """
        results = {
            "trajectory_length": trajectory.length,
            "trajectory_success": trajectory.success,
        }

        if learn_subgoals:
            subgoals = self._subgoal_discovery.discover_subgoals([trajectory])
            results["subgoals_discovered"] = len(subgoals)

        if learn_options and trajectory.length > 1:
            for i in range(len(trajectory.states) - 1):
                state = trajectory.states[i]
                action = trajectory.actions[i]
                reward = trajectory.rewards[i]
                next_state = trajectory.states[i + 1]

                self._options.update_from_transition(
                    state, action, reward, next_state, terminated=(i == len(trajectory.states) - 2)
                )

            results["option_updates"] = trajectory.length - 1

        if trajectory.success:
            self._successful_plans += 1

        return results

    def get_goal_tree(self) -> Optional[GoalTree]:
        """Get the current goal tree."""
        return self._decomposition.get_goal_tree()

    def get_current_plan(self) -> Optional[PlanResult]:
        """Get the current plan."""
        return self._current_plan

    def statistics(self) -> Dict[str, Any]:
        """Get integration statistics."""
        success_rate = self._successful_plans / self._total_plans if self._total_plans > 0 else 0.0

        return {
            "total_plans": self._total_plans,
            "successful_plans": self._successful_plans,
            "success_rate": success_rate,
            "current_plan_step": self._plan_step,
            "decomposition_stats": self._decomposition.statistics(),
            "options_stats": self._options.statistics(),
            "skills_stats": self._skills.statistics(),
            "subgoal_stats": self._subgoal_discovery.statistics(),
            "world_model_stats": self._world_model.statistics(),
        }
