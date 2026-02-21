"""Tests for planning search with Hierarchical MCTS."""

import pytest
import numpy as np

from neuro.modules.planning.planning_search import (
    SearchNode,
    NodeType,
    MCTSConfig,
    HierarchicalMCTS,
    PlanResult,
    WorldModelAdapter,
)

class TestSearchNode:
    """Tests for SearchNode class."""

    def test_node_creation(self):
        node = SearchNode(
            node_type=NodeType.STATE,
            state=np.array([1.0, 2.0]),
        )
        assert node.node_type == NodeType.STATE
        assert node.visit_count == 0
        assert node.value == 0.0

    def test_value_calculation(self):
        node = SearchNode(node_type=NodeType.ACTION)
        assert node.value == 0.0

        node.visit_count = 5
        node.value_sum = 10.0
        assert node.value == pytest.approx(2.0)

    def test_is_leaf(self):
        parent = SearchNode(node_type=NodeType.STATE)
        child = SearchNode(node_type=NodeType.ACTION)

        assert parent.is_leaf
        parent.add_child(child)
        assert not parent.is_leaf
        assert child.is_leaf

    def test_ucb_score_unvisited(self):
        node = SearchNode(node_type=NodeType.ACTION)
        score = node.ucb_score(exploration_weight=1.0, parent_visits=10)
        assert score == float("inf")

    def test_ucb_score_visited(self):
        node = SearchNode(node_type=NodeType.ACTION)
        node.visit_count = 5
        node.value_sum = 2.5

        score = node.ucb_score(exploration_weight=1.0, parent_visits=10)
        assert score > node.value

    def test_add_child(self):
        parent = SearchNode(node_type=NodeType.STATE)
        child = SearchNode(node_type=NodeType.ACTION, action="move")

        parent.add_child(child)

        assert len(parent.children) == 1
        assert child.parent == parent
        assert child.depth == 1

    def test_backpropagate(self):
        root = SearchNode(node_type=NodeType.STATE)
        child1 = SearchNode(node_type=NodeType.ACTION)
        child2 = SearchNode(node_type=NodeType.ACTION)

        root.add_child(child1)
        child1.add_child(child2)

        child2.backpropagate(value=1.0, discount=0.9)

        assert child2.visit_count == 1
        assert child2.value_sum == pytest.approx(1.0)
        assert child1.visit_count == 1
        assert child1.value_sum == pytest.approx(0.9)
        assert root.visit_count == 1
        assert root.value_sum == pytest.approx(0.81)

class TestWorldModelAdapter:
    """Tests for WorldModelAdapter class."""

    def test_creation_defaults(self):
        adapter = WorldModelAdapter(random_seed=42)
        state = np.array([1.0, 2.0])
        next_state, reward, terminal, uncertainty = adapter.imagine_step(state, 0)

        assert next_state.shape == state.shape
        assert isinstance(reward, float)
        assert isinstance(terminal, bool)
        assert isinstance(uncertainty, float)

    def test_custom_transition(self):
        adapter = WorldModelAdapter(
            transition_fn=lambda s, a: s + a,
            random_seed=42,
        )
        state = np.array([1.0, 2.0])
        next_state, _, _, _ = adapter.imagine_step(state, np.array([1.0, 1.0]))

        np.testing.assert_array_almost_equal(next_state, np.array([2.0, 3.0]))

    def test_custom_reward(self):
        adapter = WorldModelAdapter(
            reward_fn=lambda s, a, ns: np.sum(ns),
            random_seed=42,
        )
        state = np.array([1.0, 2.0])
        _, reward, _, _ = adapter.imagine_step(state, 0)

        assert reward != 0.0

    def test_custom_terminal(self):
        adapter = WorldModelAdapter(
            transition_fn=lambda s, a: s + 1,
            terminal_fn=lambda s: s[0] > 5,
            random_seed=42,
        )
        state = np.array([6.0])
        _, _, terminal, _ = adapter.imagine_step(state, 0)

        assert terminal

    def test_imagine_trajectory(self):
        adapter = WorldModelAdapter(
            transition_fn=lambda s, a: s + 1,
            reward_fn=lambda s, a, ns: 1.0,
            terminal_fn=lambda s: s[0] > 10,
            random_seed=42,
        )
        state = np.array([0.0])
        actions = [0] * 15

        states, rewards, reached_terminal = adapter.imagine_trajectory(state, actions)

        assert len(states) > 1
        assert len(rewards) == len(states) - 1
        assert reached_terminal

class TestMCTSConfig:
    """Tests for MCTSConfig class."""

    def test_default_config(self):
        config = MCTSConfig()
        assert config.num_simulations == 100
        assert config.exploration_weight == pytest.approx(1.41)
        assert config.discount == 0.99

    def test_custom_config(self):
        config = MCTSConfig(
            num_simulations=50,
            exploration_weight=2.0,
            max_depth=100,
        )
        assert config.num_simulations == 50
        assert config.exploration_weight == 2.0
        assert config.max_depth == 100

class TestHierarchicalMCTS:
    """Tests for HierarchicalMCTS class."""

    def test_creation(self):
        config = MCTSConfig(random_seed=42)
        mcts = HierarchicalMCTS(config=config)
        assert mcts.action_space == [0, 1, 2, 3]

    def test_search_basic(self):
        config = MCTSConfig(num_simulations=10, random_seed=42)
        world_model = WorldModelAdapter(
            transition_fn=lambda s, a: s + 0.1 * a,
            reward_fn=lambda s, a, ns: -np.sum(np.abs(ns)),
            terminal_fn=lambda s: np.all(np.abs(s) < 0.1),
            random_seed=42,
        )

        mcts = HierarchicalMCTS(
            config=config,
            world_model=world_model,
            action_space=[-1, 0, 1],
        )

        state = np.array([1.0, 1.0])
        result = mcts.search(state)

        assert isinstance(result, PlanResult)
        assert len(result.actions) > 0 or len(result.options) > 0
        assert result.simulations_run == 10

    def test_search_with_goal(self):
        config = MCTSConfig(num_simulations=20, random_seed=42)
        world_model = WorldModelAdapter(
            transition_fn=lambda s, a: s + np.array([a * 0.1, 0]),
            reward_fn=lambda s, a, ns: 0.0,
            random_seed=42,
        )

        mcts = HierarchicalMCTS(
            config=config,
            world_model=world_model,
            action_space=[-1, 0, 1],
        )

        state = np.array([0.0, 0.0])
        goal = np.array([1.0, 0.0])
        result = mcts.search(state, goal=goal)

        assert isinstance(result, PlanResult)

    def test_search_with_options(self):
        config = MCTSConfig(num_simulations=10, random_seed=42)

        mcts = HierarchicalMCTS(
            config=config,
            action_space=[0, 1],
            option_names=["option_a", "option_b"],
            option_policies={
                "option_a": lambda s: 0,
                "option_b": lambda s: 1,
            },
            option_terminations={
                "option_a": lambda s: True,
                "option_b": lambda s: True,
            },
        )

        state = np.array([0.0])
        result = mcts.search(state)

        assert isinstance(result, PlanResult)

    def test_get_action_values(self):
        config = MCTSConfig(num_simulations=20, random_seed=42)
        mcts = HierarchicalMCTS(config=config, action_space=[0, 1, 2])

        state = np.array([0.0])
        mcts.search(state)

        values = mcts.get_action_values()
        assert isinstance(values, dict)

    def test_get_option_values(self):
        config = MCTSConfig(num_simulations=20, random_seed=42)
        mcts = HierarchicalMCTS(
            config=config,
            action_space=[0],
            option_names=["opt"],
            option_policies={"opt": lambda s: 0},
            option_terminations={"opt": lambda s: True},
        )

        state = np.array([0.0])
        mcts.search(state)

        values = mcts.get_option_values()
        assert isinstance(values, dict)

    def test_replan(self):
        config = MCTSConfig(num_simulations=10, random_seed=42)
        mcts = HierarchicalMCTS(config=config, action_space=[0, 1])

        state1 = np.array([0.0])
        result1 = mcts.search(state1)

        state2 = np.array([0.5])
        result2 = mcts.replan(state2, result1, actions_taken=1)

        assert isinstance(result2, PlanResult)

    def test_statistics(self):
        config = MCTSConfig(num_simulations=10, random_seed=42)
        mcts = HierarchicalMCTS(config=config)

        state = np.array([0.0])
        mcts.search(state)

        stats = mcts.statistics()
        assert stats["simulations_run"] == 10
        assert stats["nodes_expanded"] > 0

class TestPlanResult:
    """Tests for PlanResult class."""

    def test_plan_result_creation(self):
        result = PlanResult(
            actions=[0, 1, 2],
            options=["opt1"],
            expected_value=5.0,
            confidence=0.8,
            tree_depth=10,
            simulations_run=100,
            nodes_expanded=50,
        )
        assert len(result.actions) == 3
        assert result.expected_value == 5.0
        assert result.confidence == 0.8

class TestMCTSExploration:
    """Tests for MCTS exploration behavior."""

    def test_exploration_vs_exploitation(self):
        config_explore = MCTSConfig(
            num_simulations=50,
            exploration_weight=10.0,
            random_seed=42,
        )
        config_exploit = MCTSConfig(
            num_simulations=50,
            exploration_weight=0.1,
            random_seed=42,
        )

        mcts_explore = HierarchicalMCTS(config=config_explore, action_space=[0, 1, 2])
        mcts_exploit = HierarchicalMCTS(config=config_exploit, action_space=[0, 1, 2])

        state = np.array([0.0])

        mcts_explore.search(state)
        mcts_exploit.search(state)

        stats_explore = mcts_explore.statistics()
        stats_exploit = mcts_exploit.statistics()

        assert stats_explore["nodes_expanded"] >= stats_exploit["nodes_expanded"] * 0.5

    def test_progressive_widening(self):
        config = MCTSConfig(
            num_simulations=50,
            progressive_widening_alpha=0.5,
            random_seed=42,
        )
        mcts = HierarchicalMCTS(
            config=config,
            action_space=list(range(100)),
        )

        state = np.array([0.0])
        mcts.search(state)

        stats = mcts.statistics()
        assert stats["nodes_expanded"] < 100 * 50

class TestMCTSWithUncertainty:
    """Tests for MCTS with uncertainty bonuses."""

    def test_uncertainty_bonus(self):
        config = MCTSConfig(
            num_simulations=20,
            use_uncertainty_bonus=True,
            uncertainty_weight=1.0,
            random_seed=42,
        )

        high_uncertainty_model = WorldModelAdapter(
            uncertainty_fn=lambda s, a: 1.0 if a == 1 else 0.1,
            random_seed=42,
        )

        mcts = HierarchicalMCTS(
            config=config,
            world_model=high_uncertainty_model,
            action_space=[0, 1, 2],
        )

        state = np.array([0.0])
        mcts.search(state)

        values = mcts.get_action_values()
        assert isinstance(values, dict)

    def test_no_uncertainty_bonus(self):
        config = MCTSConfig(
            num_simulations=20,
            use_uncertainty_bonus=False,
            random_seed=42,
        )

        mcts = HierarchicalMCTS(config=config, action_space=[0, 1])

        state = np.array([0.0])
        result = mcts.search(state)

        assert isinstance(result, PlanResult)
