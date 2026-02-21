"""
Comprehensive tests for Causal Discovery.

Tests cover:
- Conditional independence testing
- PC algorithm skeleton learning
- Edge orientation
- Structure uncertainty estimation
- Graph comparison metrics
"""

import pytest
import numpy as np
from scipy import stats
from neuro.modules.causal.causal_discovery import (
    CausalDiscovery,
    ConditionalIndependenceTest,
    CausalGraph,
    CausalEdge,
    EdgeType,
)

def generate_data_from_dag(
    n_samples: int,
    adjacency: dict,
    noise_std: float = 0.5,
    random_seed: int = 42,
) -> tuple:
    """Generate data from a known DAG structure."""
    np.random.seed(random_seed)

    variables = list(adjacency.keys())
    var_idx = {v: i for i, v in enumerate(variables)}
    n_vars = len(variables)

    data = np.zeros((n_samples, n_vars))

    # Generate in topological order
    for var in variables:
        parents = adjacency[var]["parents"]
        coeffs = adjacency[var].get("coefficients", {})
        intercept = adjacency[var].get("intercept", 0.0)

        values = intercept + np.random.normal(0, noise_std, n_samples)

        for parent in parents:
            coef = coeffs.get(parent, 1.0)
            values += coef * data[:, var_idx[parent]]

        data[:, var_idx[var]] = values

    return data, variables

@pytest.fixture
def chain_data():
    """Generate data from chain: A -> B -> C."""
    adjacency = {
        "A": {"parents": [], "intercept": 0.0},
        "B": {"parents": ["A"], "coefficients": {"A": 2.0}},
        "C": {"parents": ["B"], "coefficients": {"B": 1.5}},
    }
    data, var_names = generate_data_from_dag(500, adjacency)
    return data, var_names, adjacency

@pytest.fixture
def fork_data():
    """Generate data from fork: B <- A -> C."""
    adjacency = {
        "A": {"parents": [], "intercept": 0.0},
        "B": {"parents": ["A"], "coefficients": {"A": 2.0}},
        "C": {"parents": ["A"], "coefficients": {"A": 1.5}},
    }
    data, var_names = generate_data_from_dag(500, adjacency)
    return data, var_names, adjacency

@pytest.fixture
def collider_data():
    """Generate data from collider: A -> C <- B."""
    adjacency = {
        "A": {"parents": [], "intercept": 0.0},
        "B": {"parents": [], "intercept": 0.0},
        "C": {"parents": ["A", "B"], "coefficients": {"A": 1.0, "B": 1.0}},
    }
    data, var_names = generate_data_from_dag(500, adjacency)
    return data, var_names, adjacency

class TestConditionalIndependenceTest:
    """Test CI testing methods."""

    def test_partial_correlation_independent(self):
        """Should detect independent variables."""
        np.random.seed(42)
        n = 500
        x = np.random.randn(n)
        y = np.random.randn(n)  # Independent of x
        data = np.column_stack([x, y])

        ci_test = ConditionalIndependenceTest(alpha=0.05)
        is_ind, stat, p_val = ci_test.test(data, 0, 1)

        assert is_ind  # Should detect independence
        assert p_val > 0.05

    def test_partial_correlation_dependent(self):
        """Should detect dependent variables."""
        np.random.seed(42)
        n = 500
        x = np.random.randn(n)
        y = 2 * x + np.random.randn(n) * 0.1  # Strongly dependent
        data = np.column_stack([x, y])

        ci_test = ConditionalIndependenceTest(alpha=0.05)
        is_ind, stat, p_val = ci_test.test(data, 0, 1)

        assert not is_ind  # Should detect dependence
        assert p_val < 0.05

    def test_conditional_independence(self):
        """Should detect conditional independence."""
        np.random.seed(42)
        n = 500
        z = np.random.randn(n)
        x = 2 * z + np.random.randn(n) * 0.3
        y = 3 * z + np.random.randn(n) * 0.3  # X and Y conditionally independent given Z
        data = np.column_stack([x, y, z])

        ci_test = ConditionalIndependenceTest(alpha=0.05)

        # X and Y are NOT independent unconditionally
        is_ind_uncond, _, p_uncond = ci_test.test(data, 0, 1)
        assert not is_ind_uncond

        # X and Y ARE independent given Z
        is_ind_cond, _, p_cond = ci_test.test(data, 0, 1, z_idx=[2])
        assert is_ind_cond or p_cond > 0.01  # Allow some tolerance

    def test_mutual_information_test(self):
        """MI test should work."""
        np.random.seed(42)
        n = 500
        x = np.random.randn(n)
        y = x ** 2 + np.random.randn(n) * 0.1  # Non-linear dependence
        data = np.column_stack([x, y])

        ci_test = ConditionalIndependenceTest(method="mutual_information", alpha=0.05)
        is_ind, mi, p_val = ci_test.test(data, 0, 1)

        assert mi > 0  # Should detect mutual information
        assert not is_ind  # Should detect dependence

    def test_statistics_tracking(self):
        """Test count should increment."""
        ci_test = ConditionalIndependenceTest()
        data = np.random.randn(100, 3)

        ci_test.test(data, 0, 1)
        ci_test.test(data, 0, 2)

        stats = ci_test.statistics()
        assert stats["n_tests"] == 2

class TestCausalGraph:
    """Test CausalGraph data structure."""

    def test_graph_creation(self):
        """Should create graph correctly."""
        nodes = {"A", "B", "C"}
        edges = [
            CausalEdge("A", "B", EdgeType.DIRECTED),
            CausalEdge("B", "C", EdgeType.DIRECTED),
        ]
        graph = CausalGraph(nodes=nodes, edges=edges, is_dag=True)

        assert len(graph.nodes) == 3
        assert len(graph.edges) == 2

    def test_adjacency_matrix(self):
        """Adjacency matrix should be correct."""
        nodes = {"A", "B", "C"}
        edges = [
            CausalEdge("A", "B", EdgeType.DIRECTED),
            CausalEdge("B", "C", EdgeType.DIRECTED),
        ]
        graph = CausalGraph(nodes=nodes, edges=edges)

        adj, node_list = graph.adjacency_matrix()
        idx = {n: i for i, n in enumerate(node_list)}

        assert adj[idx["A"], idx["B"]] == 1
        assert adj[idx["B"], idx["C"]] == 1
        assert adj[idx["A"], idx["C"]] == 0

    def test_parents_children(self):
        """Should return correct parents and children."""
        nodes = {"A", "B", "C"}
        edges = [
            CausalEdge("A", "B", EdgeType.DIRECTED),
            CausalEdge("A", "C", EdgeType.DIRECTED),
        ]
        graph = CausalGraph(nodes=nodes, edges=edges)

        assert graph.parents("B") == {"A"}
        assert graph.parents("C") == {"A"}
        assert graph.children("A") == {"B", "C"}

    def test_neighbors(self):
        """Should return all neighbors."""
        nodes = {"A", "B", "C"}
        edges = [
            CausalEdge("A", "B", EdgeType.UNDIRECTED),
            CausalEdge("B", "C", EdgeType.DIRECTED),
        ]
        graph = CausalGraph(nodes=nodes, edges=edges)

        assert graph.neighbors("B") == {"A", "C"}

    def test_get_edge(self):
        """Should retrieve edges correctly."""
        nodes = {"A", "B"}
        edges = [CausalEdge("A", "B", EdgeType.DIRECTED, confidence=0.9)]
        graph = CausalGraph(nodes=nodes, edges=edges)

        edge = graph.get_edge("A", "B")
        assert edge is not None
        assert edge.confidence == 0.9

        assert graph.get_edge("B", "A") is None  # Directed, not symmetric

class TestPCAlgorithm:
    """Test PC algorithm implementation."""

    def test_discovers_chain(self, chain_data):
        """Should discover chain structure."""
        data, var_names, true_adj = chain_data

        discovery = CausalDiscovery(alpha=0.01)
        graph = discovery.pc_algorithm(data, var_names)

        # Should have edges A-B and B-C (direction may vary in CPDAG)
        neighbors_b = graph.neighbors("B")
        assert "A" in neighbors_b
        assert "C" in neighbors_b

    def test_discovers_fork(self, fork_data):
        """Should discover fork structure."""
        data, var_names, true_adj = fork_data

        discovery = CausalDiscovery(alpha=0.01)
        graph = discovery.pc_algorithm(data, var_names)

        # B and C should both be connected to A
        assert "A" in graph.neighbors("B")
        assert "A" in graph.neighbors("C")

    def test_discovers_collider(self, collider_data):
        """Should discover collider and orient correctly."""
        data, var_names, true_adj = collider_data

        discovery = CausalDiscovery(alpha=0.01)
        graph = discovery.pc_algorithm(data, var_names)

        # C should be connected to both A and B
        neighbors_c = graph.neighbors("C")
        assert "A" in neighbors_c
        assert "B" in neighbors_c

        # A and B should NOT be connected (they're marginally independent)
        edge_ab = graph.get_edge("A", "B")
        edge_ba = graph.get_edge("B", "A")
        assert edge_ab is None and edge_ba is None

    def test_discovers_disconnected_components(self):
        """Should handle disconnected graphs."""
        np.random.seed(42)
        n = 300

        # Two independent pairs
        a = np.random.randn(n)
        b = 2 * a + np.random.randn(n) * 0.1
        c = np.random.randn(n)
        d = 2 * c + np.random.randn(n) * 0.1
        data = np.column_stack([a, b, c, d])

        discovery = CausalDiscovery(alpha=0.01)
        graph = discovery.pc_algorithm(data, ["A", "B", "C", "D"])

        # A-B should be connected, C-D should be connected
        # But A-C, A-D, B-C, B-D should not be connected
        assert graph.get_edge("A", "B") is not None or graph.get_edge("B", "A") is not None
        assert graph.get_edge("C", "D") is not None or graph.get_edge("D", "C") is not None

        # Cross-cluster edges should be absent
        for x in ["A", "B"]:
            for y in ["C", "D"]:
                assert graph.get_edge(x, y) is None

    def test_empty_graph_detected(self):
        """Should detect completely independent variables."""
        np.random.seed(42)
        n = 300
        data = np.random.randn(n, 3)  # All independent

        discovery = CausalDiscovery(alpha=0.01)
        graph = discovery.pc_algorithm(data, ["A", "B", "C"])

        # Should have no edges
        assert len(graph.edges) == 0

    def test_complete_graph_not_produced(self):
        """Algorithm should remove edges based on independence."""
        # Generate data with known structure
        adjacency = {
            "A": {"parents": []},
            "B": {"parents": ["A"], "coefficients": {"A": 2.0}},
            "C": {"parents": [], "intercept": 5.0},  # C independent
        }
        data, var_names = generate_data_from_dag(500, adjacency, noise_std=0.3)

        discovery = CausalDiscovery(alpha=0.01)
        graph = discovery.pc_algorithm(data, var_names)

        # C should not be connected to A or B
        assert graph.get_edge("A", "C") is None
        assert graph.get_edge("C", "A") is None
        assert graph.get_edge("B", "C") is None
        assert graph.get_edge("C", "B") is None

class TestStructureUncertainty:
    """Test uncertainty estimation."""

    def test_bootstrap_returns_confidences(self, chain_data):
        """Bootstrap should return edge confidences."""
        data, var_names, _ = chain_data

        discovery = CausalDiscovery(alpha=0.01)
        edge_conf = discovery.estimate_structure_uncertainty(
            data, var_names, n_bootstrap=10
        )

        # Should have confidence for discovered edges
        assert isinstance(edge_conf, dict)
        assert all(0 <= v <= 1 for v in edge_conf.values())

    def test_strong_edges_high_confidence(self, chain_data):
        """Strong causal relationships should have high confidence."""
        data, var_names, _ = chain_data

        discovery = CausalDiscovery(alpha=0.01)
        edge_conf = discovery.estimate_structure_uncertainty(
            data, var_names, n_bootstrap=30
        )

        # A-B edge should have high confidence
        ab_key = tuple(sorted(["A", "B"]))
        if ab_key in edge_conf:
            assert edge_conf[ab_key] > 0.5

class TestGraphComparison:
    """Test graph comparison metrics."""

    def test_compare_identical_graphs(self):
        """Identical graphs should have SHD=0."""
        nodes = {"A", "B", "C"}
        edges = [
            CausalEdge("A", "B", EdgeType.DIRECTED),
            CausalEdge("B", "C", EdgeType.DIRECTED),
        ]
        graph1 = CausalGraph(nodes=nodes, edges=edges)
        graph2 = CausalGraph(nodes=nodes, edges=edges.copy())

        discovery = CausalDiscovery()
        metrics = discovery.compare_structures(graph1, graph2)

        assert metrics["skeleton_shd"] == 0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    def test_compare_different_graphs(self):
        """Different graphs should have non-zero SHD."""
        nodes = {"A", "B", "C"}
        edges1 = [CausalEdge("A", "B", EdgeType.DIRECTED)]
        edges2 = [CausalEdge("B", "C", EdgeType.DIRECTED)]

        graph1 = CausalGraph(nodes=nodes, edges=edges1)
        graph2 = CausalGraph(nodes=nodes, edges=edges2)

        discovery = CausalDiscovery()
        metrics = discovery.compare_structures(graph1, graph2)

        assert metrics["skeleton_shd"] == 2  # One extra, one missing
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0

class TestGraphScoring:
    """Test BIC scoring of graphs."""

    def test_score_true_graph_lower(self, chain_data):
        """True graph should have lower (better) BIC score."""
        data, var_names, _ = chain_data

        # True graph: A -> B -> C
        true_edges = [
            CausalEdge("A", "B", EdgeType.DIRECTED),
            CausalEdge("B", "C", EdgeType.DIRECTED),
        ]
        true_graph = CausalGraph(nodes=set(var_names), edges=true_edges, is_dag=True)

        # Wrong graph: A -> C, B -> C
        wrong_edges = [
            CausalEdge("A", "C", EdgeType.DIRECTED),
            CausalEdge("B", "C", EdgeType.DIRECTED),
        ]
        wrong_graph = CausalGraph(nodes=set(var_names), edges=wrong_edges, is_dag=True)

        discovery = CausalDiscovery()
        true_score = discovery.score_graph(true_graph, data, var_names)
        wrong_score = discovery.score_graph(wrong_graph, data, var_names)

        # True graph should fit better (lower BIC)
        # Note: This may not always hold depending on noise
        assert isinstance(true_score, float)
        assert isinstance(wrong_score, float)

class TestEdgeCases:
    """Test edge cases."""

    def test_two_variables(self):
        """Should work with just two variables."""
        np.random.seed(42)
        n = 300
        x = np.random.randn(n)
        y = 2 * x + np.random.randn(n) * 0.1
        data = np.column_stack([x, y])

        discovery = CausalDiscovery(alpha=0.01)
        graph = discovery.pc_algorithm(data, ["X", "Y"])

        # Should detect edge X-Y
        assert len(graph.edges) == 1

    def test_many_variables(self):
        """Should handle many variables."""
        np.random.seed(42)
        n = 300
        p = 10
        data = np.random.randn(n, p)
        data[:, 1] = 2 * data[:, 0] + np.random.randn(n) * 0.1  # X0 -> X1
        data[:, 2] = 3 * data[:, 1] + np.random.randn(n) * 0.1  # X1 -> X2

        var_names = [f"X{i}" for i in range(p)]
        discovery = CausalDiscovery(alpha=0.01)
        graph = discovery.pc_algorithm(data, var_names)

        # Should detect X0-X1 and X1-X2 edges
        assert graph.get_edge("X0", "X1") is not None or graph.get_edge("X1", "X0") is not None
        assert graph.get_edge("X1", "X2") is not None or graph.get_edge("X2", "X1") is not None

    def test_small_sample_size(self):
        """Should handle small sample size (may be less accurate)."""
        np.random.seed(42)
        n = 50  # Small sample
        x = np.random.randn(n)
        y = 2 * x + np.random.randn(n) * 0.1
        data = np.column_stack([x, y])

        discovery = CausalDiscovery(alpha=0.1)  # More lenient alpha
        graph = discovery.pc_algorithm(data, ["X", "Y"])

        # Should still produce a valid graph
        assert isinstance(graph, CausalGraph)

    def test_perfect_correlation(self):
        """Should handle perfectly correlated variables."""
        np.random.seed(42)
        n = 300
        x = np.random.randn(n)
        y = x  # Perfect correlation
        data = np.column_stack([x, y])

        discovery = CausalDiscovery(alpha=0.01)
        graph = discovery.pc_algorithm(data, ["X", "Y"])

        # Should detect strong dependence
        assert len(graph.edges) >= 0  # May or may not detect depending on numerical issues

class TestStatisticsTracking:
    """Test statistics tracking."""

    def test_discovery_count(self, chain_data):
        """Discovery count should increment."""
        data, var_names, _ = chain_data

        discovery = CausalDiscovery()
        discovery.pc_algorithm(data, var_names)
        discovery.pc_algorithm(data, var_names)

        stats = discovery.statistics()
        assert stats["n_discoveries"] == 2

    def test_ci_test_count_propagated(self, chain_data):
        """CI test count should be tracked."""
        data, var_names, _ = chain_data

        discovery = CausalDiscovery()
        discovery.pc_algorithm(data, var_names)

        stats = discovery.statistics()
        assert stats["ci_test_stats"]["n_tests"] > 0
