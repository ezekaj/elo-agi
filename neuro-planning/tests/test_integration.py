"""Tests for planning integration."""

import numpy as np

from neuro.modules.planning.integration import (
    PlanningConfig,
    PlanningIntegration,
    WorldModelAdapter,
)
from neuro.modules.planning.goal_hierarchy import Goal
from neuro.modules.planning.skill_library import Skill, SkillType
from neuro.modules.planning.subgoal_discovery import Trajectory


class TestPlanningConfig:
    """Tests for PlanningConfig class."""

    def test_default_config(self):
        config = PlanningConfig()
        assert config.state_dim == 64
        assert config.action_dim == 4
        assert config.mcts_simulations == 100

    def test_custom_config(self):
        config = PlanningConfig(
            state_dim=128,
            action_dim=8,
            mcts_simulations=50,
        )
        assert config.state_dim == 128
        assert config.action_dim == 8
        assert config.mcts_simulations == 50


class TestWorldModelAdapterIntegration:
    """Tests for WorldModelAdapter in integration context."""

    def test_adapter_creation(self):
        adapter = WorldModelAdapter(random_seed=42)
        assert adapter._imagination_calls == 0

    def test_set_models(self):
        adapter = WorldModelAdapter(random_seed=42)

        adapter.set_transition_model(lambda s, a: s * 2)
        adapter.set_reward_model(lambda s, a, ns: 1.0)
        adapter.set_terminal_model(lambda s: False)
        adapter.set_uncertainty_model(lambda s, a: 0.5)

        state = np.array([1.0, 2.0])
        next_state = adapter.predict_transition(state, 0)
        reward = adapter.predict_reward(state, 0, next_state)
        terminal = adapter.predict_terminal(next_state)
        uncertainty = adapter.predict_uncertainty(state, 0)

        np.testing.assert_array_almost_equal(next_state, np.array([2.0, 4.0]))
        assert reward == 1.0
        assert not terminal
        assert uncertainty == 0.5

    def test_imagine_trajectory(self):
        adapter = WorldModelAdapter(
            transition_model=lambda s, a: s + 1,
            reward_model=lambda s, a, ns: 1.0,
            terminal_model=lambda s: s[0] > 5,
            random_seed=42,
        )

        state = np.array([0.0])
        actions = [0] * 10

        states, rewards, terminal = adapter.imagine_trajectory(state, actions)

        assert len(states) > 1
        assert terminal

    def test_statistics(self):
        adapter = WorldModelAdapter(random_seed=42)

        adapter.predict_transition(np.array([1.0]), 0)
        adapter.predict_transition(np.array([2.0]), 0)

        stats = adapter.statistics()
        assert stats["imagination_calls"] == 2


class TestPlanningIntegration:
    """Tests for PlanningIntegration class."""

    def test_creation(self):
        integration = PlanningIntegration()
        assert integration is not None

    def test_creation_with_config(self):
        config = PlanningConfig(state_dim=32, random_seed=42)
        integration = PlanningIntegration(config=config)
        assert integration.config.state_dim == 32

    def test_set_world_model(self):
        integration = PlanningIntegration()

        integration.set_world_model(
            transition_fn=lambda s, a: s + a,
            reward_fn=lambda s, a, ns: 1.0,
        )

        assert integration._world_model._transition_model is not None
        assert integration._world_model._reward_model is not None

    def test_set_goal(self):
        integration = PlanningIntegration()

        goal = Goal(name="test_goal", description="A test goal")
        tree = integration.set_goal(goal)

        assert tree is not None
        assert tree.root.goal.name == "test_goal"

    def test_set_goal_with_decomposition(self):
        integration = PlanningIntegration()

        goal = Goal(name="main_goal")
        decomposition = {
            "main_goal": ["subtask1", "subtask2"],
            "subtask1": ["action1"],
        }

        tree = integration.set_goal(goal, decomposition)

        assert tree.get_node("subtask1") is not None
        assert tree.get_node("action1") is not None

    def test_plan(self):
        config = PlanningConfig(
            state_dim=4,
            action_dim=4,
            mcts_simulations=10,
            random_seed=42,
        )
        integration = PlanningIntegration(config=config)

        integration.set_world_model(
            transition_fn=lambda s, a: s + 0.1,
            reward_fn=lambda s, a, ns: -np.sum(np.abs(ns)),
        )

        state = np.zeros(4)
        result = integration.plan(state)

        assert result is not None
        assert result.simulations_run == 10

    def test_plan_with_goal_state(self):
        config = PlanningConfig(
            state_dim=4,
            mcts_simulations=10,
            random_seed=42,
        )
        integration = PlanningIntegration(config=config)

        state = np.zeros(4)
        goal_state = np.ones(4)
        result = integration.plan(state, goal_state=goal_state)

        assert result is not None

    def test_get_action(self):
        config = PlanningConfig(mcts_simulations=10, random_seed=42)
        integration = PlanningIntegration(config=config)

        state = np.zeros(4)
        integration.plan(state)

        action = integration.get_action(state)
        assert action is not None or integration._plan_step >= len(
            integration._current_plan.actions
        )

    def test_get_action_no_plan(self):
        integration = PlanningIntegration()
        action = integration.get_action(np.zeros(4))
        assert action is None

    def test_should_replan_no_plan(self):
        integration = PlanningIntegration()
        assert integration.should_replan(np.zeros(4))

    def test_should_replan_deviation(self):
        config = PlanningConfig(mcts_simulations=10, random_seed=42)
        integration = PlanningIntegration(config=config)

        state = np.zeros(4)
        integration.plan(state)

        expected = np.zeros(4)
        actual = np.ones(4)

        assert integration.should_replan(actual, expected, threshold=0.5)

    def test_register_option(self):
        integration = PlanningIntegration()

        option = integration.register_option(
            name="test_option",
            initiation=lambda s: True,
            policy=lambda s: 0,
            termination=lambda s: True,
        )

        assert option is not None
        assert option.name == "test_option"

    def test_register_skill(self):
        integration = PlanningIntegration()

        skill = Skill(
            name="test_skill",
            skill_type=SkillType.PRIMITIVE,
            preconditions=lambda s: True,
            effects=lambda s: s,
            policy=lambda s, p: 0,
            termination=lambda s, p: True,
        )

        name = integration.register_skill(skill)
        assert name == "test_skill"

    def test_discover_subgoals(self):
        config = PlanningConfig(state_dim=4, random_seed=42)
        integration = PlanningIntegration(config=config)

        trajectories = []
        for _ in range(10):
            states = [np.random.randn(4) for _ in range(5)]
            traj = Trajectory(
                states=states,
                actions=[0] * 4,
                rewards=[0.0] * 4,
                success=True,
            )
            trajectories.append(traj)

        subgoals = integration.discover_subgoals(trajectories)
        assert isinstance(subgoals, list)

    def test_learn_from_trajectory(self):
        config = PlanningConfig(state_dim=4, random_seed=42)
        integration = PlanningIntegration(config=config)

        integration.register_option(
            name="opt1",
            initiation=lambda s: True,
            policy=lambda s: 0,
            termination=lambda s: False,
        )

        traj = Trajectory(
            states=[np.random.randn(4) for _ in range(5)],
            actions=[0] * 4,
            rewards=[1.0] * 4,
            success=True,
        )

        results = integration.learn_from_trajectory(traj)

        assert "trajectory_length" in results
        assert results["trajectory_success"]

    def test_get_goal_tree(self):
        integration = PlanningIntegration()

        goal = Goal(name="test")
        integration.set_goal(goal)

        tree = integration.get_goal_tree()
        assert tree is not None

    def test_get_current_plan(self):
        config = PlanningConfig(mcts_simulations=10, random_seed=42)
        integration = PlanningIntegration(config=config)

        state = np.zeros(4)
        integration.plan(state)

        plan = integration.get_current_plan()
        assert plan is not None

    def test_statistics(self):
        config = PlanningConfig(mcts_simulations=10, random_seed=42)
        integration = PlanningIntegration(config=config)

        state = np.zeros(4)
        integration.plan(state)

        stats = integration.statistics()

        assert "total_plans" in stats
        assert "decomposition_stats" in stats
        assert "options_stats" in stats
        assert "skills_stats" in stats


class TestEndToEndPlanning:
    """End-to-end planning tests."""

    def test_full_planning_cycle(self):
        config = PlanningConfig(
            state_dim=2,
            action_dim=4,
            mcts_simulations=20,
            random_seed=42,
        )
        integration = PlanningIntegration(config=config)

        integration.set_world_model(
            transition_fn=lambda s, a: s + np.array([0.1 * (a - 1.5), 0]),
            reward_fn=lambda s, a, ns: -np.linalg.norm(ns - np.array([1.0, 0.0])),
            terminal_fn=lambda s: np.linalg.norm(s - np.array([1.0, 0.0])) < 0.1,
        )

        goal = Goal(name="reach_target")
        integration.set_goal(goal)

        state = np.array([0.0, 0.0])
        goal_state = np.array([1.0, 0.0])

        plan = integration.plan(state, goal_state=goal_state)
        assert plan is not None

        for _ in range(5):
            action = integration.get_action(state)
            if action is None:
                break

    def test_planning_with_skills_and_options(self):
        config = PlanningConfig(
            state_dim=4,
            mcts_simulations=10,
            random_seed=42,
        )
        integration = PlanningIntegration(config=config)

        integration.register_option(
            name="move_forward",
            initiation=lambda s: True,
            policy=lambda s: 0,
            termination=lambda s: False,
        )

        skill = Skill(
            name="dodge",
            skill_type=SkillType.PRIMITIVE,
            preconditions=lambda s: True,
            effects=lambda s: s,
            policy=lambda s, p: 1,
            termination=lambda s, p: True,
        )
        integration.register_skill(skill)

        state = np.zeros(4)
        plan = integration.plan(state, use_skills=True, use_options=True)

        assert plan is not None

    def test_subgoal_informed_planning(self):
        config = PlanningConfig(state_dim=4, random_seed=42)
        integration = PlanningIntegration(config=config)

        trajectories = []
        for _ in range(20):
            states = [np.random.randn(4) for _ in range(5)]
            states[2] = np.array([1.0, 1.0, 1.0, 1.0]) + np.random.randn(4) * 0.1
            traj = Trajectory(
                states=states,
                actions=[0] * 4,
                rewards=[0.0] * 3 + [1.0],
                success=True,
            )
            trajectories.append(traj)

        integration.discover_subgoals(trajectories)

        stats = integration.statistics()
        assert stats["subgoal_stats"]["total_subgoals"] >= 0
