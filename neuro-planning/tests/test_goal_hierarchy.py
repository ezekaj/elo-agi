"""Tests for goal hierarchy and MAXQ decomposition."""

import pytest
import numpy as np

from src.goal_hierarchy import (
    Goal,
    GoalNode,
    GoalTree,
    GoalStatus,
    MAXQDecomposition,
    CompletionFunction,
)


class TestGoal:
    """Tests for Goal class."""

    def test_goal_creation(self):
        goal = Goal(name="test_goal", description="A test goal")
        assert goal.name == "test_goal"
        assert goal.description == "A test goal"
        assert goal.priority == 1.0

    def test_goal_with_preconditions(self):
        goal = Goal(
            name="conditional_goal",
            preconditions=[lambda s: s > 0, lambda s: s < 10],
        )
        assert goal.check_preconditions(5)
        assert not goal.check_preconditions(-1)
        assert not goal.check_preconditions(15)

    def test_goal_effects(self):
        goal = Goal(
            name="effect_goal",
            effects=[lambda s: s * 2, lambda s: s + 1],
        )
        result = goal.apply_effects(5)
        assert result == 11

    def test_goal_with_deadline(self):
        goal = Goal(name="deadline_goal", deadline=100.0)
        assert goal.deadline == 100.0


class TestGoalNode:
    """Tests for GoalNode class."""

    def test_node_creation(self):
        goal = Goal(name="root")
        node = GoalNode(goal=goal)
        assert node.goal.name == "root"
        assert node.is_primitive
        assert not node.is_composite

    def test_add_child(self):
        parent = GoalNode(goal=Goal(name="parent"))
        child = GoalNode(goal=Goal(name="child"))
        parent.add_child(child)

        assert len(parent.children) == 1
        assert child.parent == parent
        assert not parent.is_primitive
        assert parent.is_composite

    def test_get_depth(self):
        root = GoalNode(goal=Goal(name="root"))
        level1 = GoalNode(goal=Goal(name="level1"))
        level2 = GoalNode(goal=Goal(name="level2"))

        root.add_child(level1)
        level1.add_child(level2)

        assert root.get_depth() == 0
        assert level1.get_depth() == 1
        assert level2.get_depth() == 2

    def test_get_ancestors(self):
        root = GoalNode(goal=Goal(name="root"))
        level1 = GoalNode(goal=Goal(name="level1"))
        level2 = GoalNode(goal=Goal(name="level2"))

        root.add_child(level1)
        level1.add_child(level2)

        ancestors = level2.get_ancestors()
        assert len(ancestors) == 2
        assert ancestors[0].goal.name == "level1"
        assert ancestors[1].goal.name == "root"

    def test_get_descendants(self):
        root = GoalNode(goal=Goal(name="root"))
        child1 = GoalNode(goal=Goal(name="child1"))
        child2 = GoalNode(goal=Goal(name="child2"))
        grandchild = GoalNode(goal=Goal(name="grandchild"))

        root.add_child(child1)
        root.add_child(child2)
        child1.add_child(grandchild)

        descendants = root.get_descendants()
        assert len(descendants) == 3


class TestGoalTree:
    """Tests for GoalTree class."""

    def test_tree_creation(self):
        root = GoalNode(goal=Goal(name="root"))
        tree = GoalTree(root=root)
        assert tree.root == root
        assert "root" in tree.nodes

    def test_get_node(self):
        root = GoalNode(goal=Goal(name="root"))
        child = GoalNode(goal=Goal(name="child"))
        root.add_child(child)
        tree = GoalTree(root=root)

        assert tree.get_node("root") == root
        assert tree.get_node("child") == child
        assert tree.get_node("nonexistent") is None

    def test_get_primitive_goals(self):
        root = GoalNode(goal=Goal(name="root"))
        child1 = GoalNode(goal=Goal(name="child1"))
        child2 = GoalNode(goal=Goal(name="child2"))
        grandchild = GoalNode(goal=Goal(name="grandchild"))

        root.add_child(child1)
        root.add_child(child2)
        child1.add_child(grandchild)

        tree = GoalTree(root=root)
        primitives = tree.get_primitive_goals()

        assert len(primitives) == 2
        names = [n.goal.name for n in primitives]
        assert "child2" in names
        assert "grandchild" in names

    def test_update_completion(self):
        root = GoalNode(goal=Goal(name="root"))
        child1 = GoalNode(goal=Goal(name="child1"))
        child2 = GoalNode(goal=Goal(name="child2"))

        root.add_child(child1)
        root.add_child(child2)

        child1.completion = 1.0
        child2.completion = 0.5

        tree = GoalTree(root=root)
        tree.update_completion()

        assert root.completion == pytest.approx(0.75)

    def test_statistics(self):
        root = GoalNode(goal=Goal(name="root"))
        child = GoalNode(goal=Goal(name="child"))
        root.add_child(child)
        tree = GoalTree(root=root)

        stats = tree.statistics()
        assert stats["total_nodes"] == 2
        assert stats["primitive_goals"] == 1


class TestCompletionFunction:
    """Tests for CompletionFunction class."""

    def test_evaluate(self):
        cf = CompletionFunction(
            goal_name="test",
            function=lambda s: s / 10.0,
        )
        assert cf.evaluate(5) == pytest.approx(0.5)
        assert cf.evaluate(10) == pytest.approx(1.0)


class TestMAXQDecomposition:
    """Tests for MAXQDecomposition class."""

    def test_decompose_simple(self):
        decomp = MAXQDecomposition(random_seed=42)
        root_goal = Goal(name="root")

        subtasks = {
            "root": ["subtask1", "subtask2"],
            "subtask1": ["action1"],
        }

        tree = decomp.decompose(root_goal, subtasks)

        assert tree is not None
        assert len(tree.nodes) == 4
        assert tree.get_node("subtask1") is not None
        assert tree.get_node("action1") is not None

    def test_get_value(self):
        decomp = MAXQDecomposition(random_seed=42)
        root_goal = Goal(name="root")
        decomp.decompose(root_goal, {"root": ["action"]})

        state = {"x": 1}
        value = decomp.get_value("root", state)
        assert isinstance(value, float)

    def test_select_subtask(self):
        decomp = MAXQDecomposition(random_seed=42)
        root_goal = Goal(name="root")
        decomp.decompose(root_goal, {"root": ["action1", "action2"]})

        state = {"x": 1}
        selected = decomp.select_subtask("root", state, epsilon=0.0)
        assert selected in ["action1", "action2"]

    def test_select_subtask_exploration(self):
        decomp = MAXQDecomposition(random_seed=42)
        root_goal = Goal(name="root")
        decomp.decompose(root_goal, {"root": ["action1", "action2"]})

        selections = set()
        for _ in range(100):
            selected = decomp.select_subtask("root", {}, epsilon=1.0)
            if selected:
                selections.add(selected)

        assert len(selections) == 2

    def test_update_value(self):
        decomp = MAXQDecomposition(random_seed=42, learning_rate=0.5)
        root_goal = Goal(name="root")
        decomp.decompose(root_goal, {"root": ["action"]})

        state1 = "s1"
        state2 = "s2"
        td_error = decomp.update_value("action", state1, 1.0, state2)

        assert isinstance(td_error, float)

    def test_get_hierarchical_policy(self):
        decomp = MAXQDecomposition(random_seed=42)
        root_goal = Goal(name="root")
        decomp.decompose(root_goal, {
            "root": ["level1"],
            "level1": ["action"],
        })

        policy = decomp.get_hierarchical_policy({})
        assert len(policy) == 2
        assert policy[0][0] == "root"
        assert policy[1][0] == "level1"

    def test_set_completion_function(self):
        decomp = MAXQDecomposition(random_seed=42)
        root_goal = Goal(name="root")
        decomp.decompose(root_goal, {"root": ["action"]})

        decomp.set_completion_function("root", lambda s: 0.5)
        completion = decomp.compute_completion_function("root", {})
        assert completion == 0.5

    def test_statistics(self):
        decomp = MAXQDecomposition(random_seed=42)
        root_goal = Goal(name="root")
        decomp.decompose(root_goal, {"root": ["action"]})

        stats = decomp.statistics()
        assert stats["total_decompositions"] == 1
        assert stats["tree_nodes"] == 2


class TestMAXQValueDecomposition:
    """Tests for MAXQ value function decomposition."""

    def test_primitive_value_learning(self):
        decomp = MAXQDecomposition(random_seed=42, learning_rate=0.1)
        root_goal = Goal(name="root")
        decomp.decompose(root_goal, {"root": ["primitive"]})

        # Track value updates
        for _ in range(10):
            td_error = decomp.update_value("primitive", "s1", 1.0, "s2")

        # Verify updates happened
        stats = decomp.statistics()
        assert stats["total_value_updates"] == 10

    def test_composite_value(self):
        decomp = MAXQDecomposition(random_seed=42, learning_rate=0.1, discount=0.9)
        root_goal = Goal(name="root")
        decomp.decompose(root_goal, {"root": ["p1", "p2"]})

        for _ in range(10):
            decomp.update_value("p1", "s", 1.0, "s2")
            decomp.update_value("p2", "s", 0.5, "s2")

        root_value = decomp.get_value("root", "s")
        assert isinstance(root_value, float)

    def test_causal_path_reasoning(self):
        decomp = MAXQDecomposition(random_seed=42)
        root_goal = Goal(name="navigate")
        decomp.decompose(root_goal, {
            "navigate": ["get_passenger", "go_to_dest", "put_passenger"],
        })

        tree = decomp.get_goal_tree()
        assert tree.get_node("get_passenger") is not None
        assert tree.get_node("go_to_dest") is not None
        assert tree.get_node("put_passenger") is not None
