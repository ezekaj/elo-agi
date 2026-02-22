"""
Comprehensive tests for SymbolicBinder.

Tests cover:
- Symbol registration and retrieval
- Role-filler binding
- Tensor product and HRR composition
- Unbinding and retrieval
- Analogical binding
"""

import pytest
import numpy as np
from neuro.modules.abstract.symbolic_binder import (
    SymbolicBinder,
    RoleType,
    HRROperations,
    TPROperations,
)


class TestHRROperations:
    """Test Holographic Reduced Representation operations."""

    def test_circular_convolution_shape(self):
        """Convolution should preserve shape."""
        a = np.random.randn(100)
        b = np.random.randn(100)
        result = HRROperations.circular_convolution(a, b)
        assert result.shape == a.shape

    def test_circular_convolution_commutative(self):
        """Convolution should be commutative."""
        a = np.random.randn(100)
        b = np.random.randn(100)
        ab = HRROperations.circular_convolution(a, b)
        ba = HRROperations.circular_convolution(b, a)
        np.testing.assert_allclose(ab, ba, rtol=1e-10)

    def test_correlation_retrieval(self):
        """Correlation should approximately retrieve bound content."""
        np.random.seed(42)
        role = np.random.randn(100)
        role = HRROperations.normalize(role)
        filler = np.random.randn(100)
        filler = HRROperations.normalize(filler)

        # Bind
        bound = HRROperations.circular_convolution(role, filler)

        # Unbind
        retrieved = HRROperations.circular_correlation(role, bound)

        # Should be similar to original filler
        similarity = np.dot(retrieved / np.linalg.norm(retrieved), filler)
        assert similarity > 0.5

    def test_normalize(self):
        """Normalize should produce unit vectors."""
        v = np.array([3, 4, 0])
        normalized = HRROperations.normalize(v)
        np.testing.assert_almost_equal(np.linalg.norm(normalized), 1.0)

    def test_normalize_zero_vector(self):
        """Normalize should handle zero vectors."""
        v = np.zeros(10)
        normalized = HRROperations.normalize(v)
        np.testing.assert_array_equal(normalized, v)


class TestTPROperations:
    """Test Tensor Product Representation operations."""

    def test_tensor_product_shape(self):
        """Tensor product should produce correct shape."""
        role = np.random.randn(10)
        filler = np.random.randn(20)
        result = TPROperations.tensor_product(role, filler)
        assert result.shape == (200,)  # 10 * 20

    def test_sum_tpr(self):
        """Sum should combine multiple TPRs."""
        bindings = [np.random.randn(100) for _ in range(5)]
        result = TPROperations.sum_tpr(bindings)
        assert result.shape == (100,)

    def test_sum_tpr_empty(self):
        """Sum of empty list should be empty."""
        result = TPROperations.sum_tpr([])
        assert len(result) == 0


class TestSymbolicBinder:
    """Test SymbolicBinder class."""

    @pytest.fixture
    def binder(self):
        """Create a binder for testing."""
        return SymbolicBinder(
            embedding_dim=128,
            role_dim=32,
            random_seed=42,
        )

    def test_register_symbol(self, binder):
        """Should register and retrieve symbols."""
        rep = binder.register_symbol("dog")
        assert rep is not None
        assert rep.shape == (binder.embedding_dim,)

        retrieved = binder.get_symbol("dog")
        np.testing.assert_array_equal(rep, retrieved)

    def test_register_with_custom_rep(self, binder):
        """Should register with custom representation."""
        custom = np.random.randn(binder.embedding_dim)
        binder.register_symbol("cat", custom)

        retrieved = binder.get_symbol("cat")
        # Should be normalized version of custom
        expected = custom / np.linalg.norm(custom)
        np.testing.assert_allclose(retrieved, expected)

    def test_bind_simple(self, binder):
        """Should create simple binding."""
        binding = binder.bind("chase")

        assert binding.symbol == "chase"
        assert binding.neural_rep is not None

    def test_bind_with_roles(self, binder):
        """Should create binding with role-filler pairs."""
        binder.register_symbol("dog")
        binder.register_symbol("cat")

        binding = binder.bind(
            "chase",
            roles={
                RoleType.AGENT: ("dog", None),
                RoleType.PATIENT: ("cat", None),
            },
        )

        assert binding.symbol == "chase"
        assert RoleType.AGENT in binding.role_bindings
        assert RoleType.PATIENT in binding.role_bindings
        assert binding.get_filler(RoleType.AGENT) == "dog"
        assert binding.get_filler(RoleType.PATIENT) == "cat"

    def test_bind_structure(self, binder):
        """Should bind predicate with named arguments."""
        binding = binder.bind_structure(
            predicate="loves",
            args={
                "agent": ("john", None),
                "patient": ("mary", None),
            },
        )

        assert binding.symbol == "loves"
        assert binding.get_filler(RoleType.AGENT) == "john"
        assert binding.get_filler(RoleType.PATIENT) == "mary"

    def test_compose_single(self, binder):
        """Compose single binding should return representation."""
        binding = binder.bind("concept")
        composed = binder.compose([binding])

        assert composed.shape == (binder.embedding_dim,)
        assert np.linalg.norm(composed) > 0.99  # Should be normalized

    def test_compose_multiple(self, binder):
        """Compose multiple bindings should combine them."""
        b1 = binder.bind("concept1")
        b2 = binder.bind("concept2")

        composed = binder.compose([b1, b2])
        assert composed.shape == (binder.embedding_dim,)

    def test_compose_with_roles(self, binder):
        """Compose should include role-filler information."""
        binding = binder.bind(
            "chase",
            roles={
                RoleType.AGENT: ("dog", None),
                RoleType.PATIENT: ("cat", None),
            },
        )

        composed = binder.compose([binding])

        # Should be able to retrieve role fillers
        agent_retrieved = binder.retrieve_by_role(composed, RoleType.AGENT)
        matches = binder.retrieve_nearest_symbol(agent_retrieved)

        # Dog should be among top matches
        symbols = [m[0] for m in matches]
        assert "dog" in symbols

    def test_retrieve_by_role(self, binder):
        """Should retrieve filler for given role."""
        binder.register_symbol("john")
        binder.register_symbol("mary")

        binding = binder.bind(
            "loves",
            roles={
                RoleType.AGENT: ("john", None),
                RoleType.PATIENT: ("mary", None),
            },
        )

        composed = binder.compose([binding])

        agent_rep = binder.retrieve_by_role(composed, RoleType.AGENT)
        assert agent_rep.shape == (binder.embedding_dim,)

    def test_retrieve_nearest_symbol(self, binder):
        """Should find nearest registered symbol."""
        binder.register_symbol("apple")
        binder.register_symbol("orange")
        binder.register_symbol("car")

        # Query with apple's representation
        apple_rep = binder.get_symbol("apple")
        matches = binder.retrieve_nearest_symbol(apple_rep, top_k=3)

        assert len(matches) == 3
        assert matches[0][0] == "apple"  # Exact match should be first
        assert matches[0][1] > 0.99  # Should have high similarity

    def test_unbind(self, binder):
        """Should unbind to recover role-filler pairs."""
        binder.register_symbol("dog")
        binder.register_symbol("cat")

        binding = binder.bind(
            "chase",
            roles={
                RoleType.AGENT: ("dog", None),
                RoleType.PATIENT: ("cat", None),
            },
        )

        unbound = binder.unbind(binding)

        # Should have both roles
        assert RoleType.AGENT in unbound
        assert RoleType.PATIENT in unbound

    def test_similarity(self, binder):
        """Should compute similarity between bindings."""
        b1 = binder.bind("chase", roles={RoleType.AGENT: ("dog", None)})
        b2 = binder.bind("chase", roles={RoleType.AGENT: ("dog", None)})
        b3 = binder.bind("love", roles={RoleType.AGENT: ("cat", None)})

        sim_same = binder.similarity(b1, b2)
        sim_diff = binder.similarity(b1, b3)

        assert sim_same > sim_diff

    def test_analogical_bind(self, binder):
        """Should create analogical binding with mapped elements."""
        binder.register_symbol("dog")
        binder.register_symbol("cat")
        binder.register_symbol("wolf")
        binder.register_symbol("lion")

        source = binder.bind(
            "chase",
            roles={
                RoleType.AGENT: ("dog", None),
                RoleType.PATIENT: ("cat", None),
            },
        )

        # Create analogy: dog -> wolf, cat -> lion
        mapping = {"dog": "wolf", "cat": "lion"}
        target = binder.analogical_bind(source, mapping)

        assert target.get_filler(RoleType.AGENT) == "wolf"
        assert target.get_filler(RoleType.PATIENT) == "lion"

    def test_verify_binding_consistency(self, binder):
        """Should verify binding can be recovered."""
        binder.register_symbol("john")
        binder.register_symbol("ball")

        binding = binder.bind(
            "kicks",
            roles={
                RoleType.AGENT: ("john", None),
                RoleType.THEME: ("ball", None),
            },
        )

        is_consistent, avg_sim = binder.verify_binding_consistency(binding)

        # Should be reasonably consistent
        assert avg_sim > 0  # Some similarity expected

    def test_role_vectors_orthogonal(self, binder):
        """Role vectors should be approximately orthogonal."""
        agent = binder.get_role_vector(RoleType.AGENT)
        patient = binder.get_role_vector(RoleType.PATIENT)

        dot = np.dot(agent, patient)
        assert abs(dot) < 0.5  # Should be roughly orthogonal

    def test_list_symbols(self, binder):
        """Should list all registered symbols."""
        binder.register_symbol("a")
        binder.register_symbol("b")
        binder.register_symbol("c")

        symbols = binder.list_symbols()
        assert set(symbols) == {"a", "b", "c"}

    def test_statistics(self, binder):
        """Should track statistics."""
        binder.bind("test")
        stats = binder.statistics()

        assert stats["n_symbols"] > 0
        assert stats["n_bindings"] == 1


class TestHRRVsTPR:
    """Compare HRR and TPR binding methods."""

    def test_both_methods_work(self):
        """Both binding methods should produce valid results."""
        for method in ["hrr", "tpr"]:
            binder = SymbolicBinder(
                embedding_dim=128,
                binding_method=method,
                random_seed=42,
            )

            binding = binder.bind("test", roles={RoleType.AGENT: ("subject", None)})

            composed = binder.compose([binding])
            assert composed.shape == (128,)
            assert np.linalg.norm(composed) > 0.99


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_composition(self):
        """Empty binding list should return zeros."""
        binder = SymbolicBinder(embedding_dim=64, random_seed=42)
        composed = binder.compose([])
        assert composed.shape == (64,)
        np.testing.assert_array_equal(composed, np.zeros(64))

    def test_binding_without_roles(self):
        """Binding without roles should work."""
        binder = SymbolicBinder(random_seed=42)
        binding = binder.bind("singleton")

        assert binding.symbol == "singleton"
        assert len(binding.role_bindings) == 0

    def test_get_nonexistent_symbol(self):
        """Getting nonexistent symbol should return None."""
        binder = SymbolicBinder(random_seed=42)
        result = binder.get_symbol("nonexistent")
        assert result is None

    def test_binding_get_nonexistent_role(self):
        """Getting nonexistent role from binding should return None."""
        binder = SymbolicBinder(random_seed=42)
        binding = binder.bind("test")

        assert binding.get_filler(RoleType.AGENT) is None
        assert binding.get_neural_rep(RoleType.AGENT) is None
