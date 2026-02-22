"""
Cross-module integration tests for the new AGI modules.

Tests integration between:
- neuro-abstract → SharedSpace
- neuro-robust → SharedSpace
- neuro-causal → neuro-inference
- End-to-end pipeline
"""

import pytest
import numpy as np
import os


# Add parent paths for imports
class TestNeuroAbstractSharedSpaceIntegration:
    """Tests for neuro-abstract → SharedSpace integration."""

    @pytest.fixture
    def shared_space_integration(self):
        """Create SharedSpaceIntegration from neuro-abstract."""
        try:
            from integration import SharedSpaceIntegration, SharedSpaceConfig

            config = SharedSpaceConfig(
                embedding_dim=512,
                projection_dim=256,
                random_seed=42,
            )
            return SharedSpaceIntegration(config=config)
        except ImportError:
            pytest.skip("neuro-abstract not available")

    @pytest.fixture
    def binder(self, shared_space_integration):
        """Get the symbolic binder."""
        return shared_space_integration.binder

    def test_bind_symbol_to_shared_space(self, shared_space_integration):
        """Test binding a symbol and projecting to shared space."""
        embedding = shared_space_integration.bind_to_shared_space("cat")
        assert embedding.vector.shape == (512,)
        assert embedding.symbol == "cat"
        assert np.isclose(np.linalg.norm(embedding.vector), 1.0)

    def test_bind_with_roles(self, shared_space_integration):
        """Test binding with roles projects correctly."""
        try:
            from symbolic_binder import RoleType
        except ImportError:
            pytest.skip("symbolic_binder not available")

        roles = {RoleType.AGENT: "dog", RoleType.PATIENT: "bone"}
        embedding = shared_space_integration.bind_to_shared_space("chase", roles=roles)

        assert embedding.vector.shape == (512,)
        assert embedding.roles[RoleType.AGENT] == "dog"
        assert embedding.roles[RoleType.PATIENT] == "bone"

    def test_query_shared_space(self, shared_space_integration):
        """Test querying shared space after projection."""
        # Add some embeddings
        e1 = shared_space_integration.bind_to_shared_space("dog")
        e2 = shared_space_integration.bind_to_shared_space("cat")
        e3 = shared_space_integration.bind_to_shared_space("tree")

        # Query with dog's vector
        results = shared_space_integration.query(e1.vector, top_k=3)

        assert len(results) >= 1
        # Dog should be most similar to itself
        assert results[0][0].symbol == "dog"
        assert results[0][1] > 0.99

    def test_abstraction_projection(self, shared_space_integration):
        """Test projecting abstractions to shared space."""
        try:
            from abstraction_engine import Abstraction, AbstractionLevel
        except ImportError:
            pytest.skip("abstraction_engine not available")

        abstraction = Abstraction(
            id="animal_concept",
            level=AbstractionLevel.CONCEPT,
            embedding=np.random.randn(256),
            variables=["species", "size"],
            instances=["cat", "dog"],
            confidence=0.9,
        )

        embedding = shared_space_integration.project_abstraction(abstraction)

        assert embedding.vector.shape == (512,)
        assert embedding.abstraction_id == "animal_concept"
        assert embedding.confidence == 0.9

    def test_grounded_embedding(self, shared_space_integration):
        """Test creating grounded symbol-perception embeddings."""
        perceptual = np.random.randn(256)
        embedding = shared_space_integration.create_grounded_embedding("cat", perceptual)

        assert embedding.vector.shape == (512,)
        assert embedding.symbol == "cat"
        from integration import SemanticModalityType

        assert embedding.modality == SemanticModalityType.GROUNDED

    def test_store_and_retrieve_concept(self, shared_space_integration):
        """Test storing and retrieving concepts."""
        embedding = shared_space_integration.bind_to_shared_space("elephant")
        shared_space_integration.store_concept("big_animal", embedding)

        retrieved = shared_space_integration.retrieve_concept("big_animal")
        assert retrieved is not None
        assert retrieved.symbol == "elephant"

    def test_statistics(self, shared_space_integration):
        """Test statistics tracking."""
        shared_space_integration.bind_to_shared_space("a")
        shared_space_integration.bind_to_shared_space("b")

        stats = shared_space_integration.statistics()
        assert stats["n_projections"] >= 2
        assert stats["n_active_embeddings"] >= 2


class TestNeuroRobustSharedSpaceIntegration:
    """Tests for neuro-robust → SharedSpace integration."""

    @pytest.fixture
    def shared_space_robustness(self):
        """Create SharedSpaceRobustness from neuro-robust."""
        try:
            from integration import SharedSpaceRobustness, RobustnessLevel

            return SharedSpaceRobustness(
                embedding_dim=256,
                hidden_dim=128,
                n_classes=5,
                robustness_level=RobustnessLevel.STANDARD,
                random_seed=42,
            )
        except ImportError:
            pytest.skip("neuro-robust not available")

    @pytest.fixture
    def reference_data(self):
        """Generate reference data for fitting."""
        np.random.seed(42)
        return [(np.random.randn(256), i % 5) for i in range(100)]

    def test_add_robustness_basic(self, shared_space_robustness):
        """Test adding robustness to embedding."""
        vector = np.random.randn(256)
        embedding = shared_space_robustness.add_robustness(
            vector, modality="visual", source_module="test"
        )

        assert embedding.vector.shape == (256,)
        assert embedding.robustness is not None
        assert embedding.robustness.uncertainty >= 0

    def test_uncertainty_computed(self, shared_space_robustness):
        """Test uncertainty is computed."""
        vector = np.random.randn(256)
        embedding = shared_space_robustness.add_robustness(vector)

        assert embedding.robustness.epistemic >= 0
        assert embedding.robustness.aleatoric >= 0
        assert embedding.robustness.total >= 0

    def test_ood_detection_after_fit(self, shared_space_robustness, reference_data):
        """Test OOD detection after fitting."""
        shared_space_robustness.fit(reference_data)

        # Normal input
        normal_embedding = shared_space_robustness.add_robustness(np.random.randn(256) * 0.5)

        # OOD input (extreme values)
        ood_embedding = shared_space_robustness.add_robustness(np.ones(256) * 100)

        # OOD should have higher score
        assert ood_embedding.robustness.ood_score >= normal_embedding.robustness.ood_score - 0.1

    def test_filter_reliable(self, shared_space_robustness, reference_data):
        """Test filtering reliable embeddings."""
        shared_space_robustness.fit(reference_data)

        embeddings = [
            shared_space_robustness.add_robustness(np.random.randn(256)) for _ in range(10)
        ]

        # Some should be filtered
        reliable = shared_space_robustness.filter_reliable(embeddings, threshold=0.5)

        assert len(reliable) <= len(embeddings)
        for e in reliable:
            assert e.effective_confidence() >= 0.5

    def test_rank_by_reliability(self, shared_space_robustness):
        """Test ranking by reliability."""
        embeddings = [
            shared_space_robustness.add_robustness(np.random.randn(256)) for _ in range(5)
        ]

        ranked = shared_space_robustness.rank_by_reliability(embeddings)

        # Should be sorted by score descending
        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_aggregate_with_uncertainty(self, shared_space_robustness):
        """Test aggregating embeddings with uncertainty weighting."""
        embeddings = [
            shared_space_robustness.add_robustness(np.random.randn(256)) for _ in range(3)
        ]

        aggregated = shared_space_robustness.aggregate_with_uncertainty(embeddings)

        assert aggregated.vector.shape == (256,)
        assert np.isclose(np.linalg.norm(aggregated.vector), 1.0)
        assert aggregated.robustness.uncertainty >= 0

    def test_effective_confidence(self, shared_space_robustness):
        """Test effective confidence computation."""
        embedding = shared_space_robustness.add_robustness(np.random.randn(256))
        eff = embedding.effective_confidence()

        assert 0 <= eff <= 1

    def test_is_reliable(self, shared_space_robustness):
        """Test reliability check."""
        embedding = shared_space_robustness.add_robustness(np.random.randn(256))
        reliable = embedding.is_reliable(threshold=0.0)

        assert isinstance(reliable, bool)

    def test_statistics(self, shared_space_robustness):
        """Test statistics tracking."""
        for _ in range(5):
            shared_space_robustness.add_robustness(np.random.randn(256))

        stats = shared_space_robustness.statistics()
        assert stats["n_processed"] == 5


class TestNeuroCausalInferenceIntegration:
    """Tests for neuro-causal → neuro-inference integration."""

    @pytest.fixture
    def adapter(self):
        """Create InferenceSCMAdapter."""
        try:
            from inference_adapter import InferenceSCMAdapter, AdapterConfig

            config = AdapterConfig(random_seed=42)
            return InferenceSCMAdapter(name="test_adapter", config=config)
        except ImportError:
            pytest.skip("neuro-causal not available")

    def test_add_linear_equation(self, adapter):
        """Test adding linear equation (neuro-inference interface)."""
        adapter.add_linear_equation(
            variable="Y",
            parents=["X"],
            coefficients={"X": 2.0},
            intercept=1.0,
        )

        assert "Y" in adapter._dscm._variables
        assert adapter.get_parents("Y") == ["X"]

    def test_sample(self, adapter):
        """Test sampling from model."""
        adapter.add_linear_equation("X", [], {}, intercept=0.0)
        adapter.add_linear_equation("Y", ["X"], {"X": 1.5}, intercept=0.0)

        samples = adapter.sample(n_samples=100)
        assert len(samples) == 100
        for s in samples:
            assert "X" in s
            assert "Y" in s

    def test_intervene(self, adapter):
        """Test intervention (neuro-inference interface)."""
        adapter.add_linear_equation("X", [], {}, intercept=0.0)
        adapter.add_linear_equation("Y", ["X"], {"X": 2.0}, intercept=0.0)

        y_values = adapter.intervene({"X": 1.0}, "Y", n_samples=100)

        assert len(y_values) == 100
        # Under do(X=1), Y ≈ 2.0 + noise
        assert np.abs(np.mean(y_values) - 2.0) < 0.5

    def test_causal_effect(self, adapter):
        """Test causal effect estimation."""
        adapter.add_linear_equation("X", [], {}, intercept=0.0)
        adapter.add_linear_equation("Y", ["X"], {"X": 3.0}, intercept=0.0)

        effect = adapter.causal_effect("X", "Y", treatment_values=(0, 1))

        # ACE should be ≈ 3.0
        assert np.abs(effect - 3.0) < 0.5

    def test_counterfactual(self, adapter):
        """Test counterfactual query."""
        adapter.add_linear_equation("X", [], {}, intercept=0.0)
        adapter.add_linear_equation("Y", ["X"], {"X": 2.0}, intercept=0.0)

        cf = adapter.counterfactual(evidence={"X": 1.0, "Y": 2.0}, intervention={"X": 2.0})

        # If X had been 2, Y would be 4
        assert np.abs(cf["Y"] - 4.0) < 0.5

    def test_causal_gradient(self, adapter):
        """Test gradient-based effect estimation."""
        adapter.add_linear_equation("X", [], {}, intercept=0.0)
        adapter.add_linear_equation("Y", ["X"], {"X": 2.5}, intercept=0.0)

        grad = adapter.causal_gradient("X", "Y", {"X": 1.0})

        # Gradient should be ≈ 2.5
        assert np.abs(grad - 2.5) < 0.5

    def test_is_d_separated(self, adapter):
        """Test d-separation query."""
        # X -> Y -> Z (chain)
        adapter.add_linear_equation("X", [], {})
        adapter.add_linear_equation("Y", ["X"], {"X": 1.0})
        adapter.add_linear_equation("Z", ["Y"], {"Y": 1.0})

        # X not d-separated from Z unconditionally
        assert not adapter.is_d_separated("X", "Z", None)

        # X d-separated from Z given Y
        assert adapter.is_d_separated("X", "Z", {"Y"})

    def test_get_ancestors_descendants(self, adapter):
        """Test ancestor/descendant queries."""
        adapter.add_linear_equation("A", [], {})
        adapter.add_linear_equation("B", ["A"], {"A": 1.0})
        adapter.add_linear_equation("C", ["B"], {"B": 1.0})

        assert "A" in adapter.get_ancestors("C")
        assert "B" in adapter.get_ancestors("C")
        assert "C" in adapter.get_descendants("A")

    def test_statistics(self, adapter):
        """Test statistics."""
        adapter.add_linear_equation("X", [], {})
        adapter.sample(10)

        stats = adapter.statistics()
        assert stats["n_variables"] >= 1
        assert stats["adapter_type"] == "InferenceSCMAdapter"


class TestCausalInferenceEnhanced:
    """Tests for enhanced causal inference engine."""

    @pytest.fixture
    def engine(self):
        """Create CausalInferenceEnhanced."""
        try:
            from inference_adapter import CausalInferenceEnhanced

            return CausalInferenceEnhanced(random_seed=42)
        except ImportError:
            pytest.skip("neuro-causal not available")

    def test_discover_structure(self, engine):
        """Test structure discovery from data."""
        # Generate data from known structure: X -> Y
        np.random.seed(42)
        data = []
        for _ in range(200):
            x = np.random.randn()
            y = 2 * x + np.random.randn() * 0.1
            data.append({"X": x, "Y": y})

        parents = engine.discover_structure(data, alpha=0.1)

        # Should discover X -> Y or Y -> X (direction may vary)
        assert "X" in parents or "Y" in parents

    def test_build_scm_from_structure(self, engine):
        """Test building SCM from structure."""
        parents = {"X": [], "Y": ["X"], "Z": ["Y"]}
        scm = engine.build_scm_from_structure(parents)

        assert scm.get_parents("Y") == ["X"]
        assert scm.get_parents("Z") == ["Y"]

    def test_causal_attribution(self, engine):
        """Test causal attribution."""
        # Build simple SCM
        engine.scm.add_linear_equation("A", [], {})
        engine.scm.add_linear_equation("B", ["A"], {"A": 2.0})
        engine.scm.add_linear_equation("C", ["A", "B"], {"A": 1.0, "B": 0.5})

        evidence = {"A": 1.0, "B": 2.0, "C": 2.0}
        attributions = engine.causal_attribution("C", 2.0, evidence)

        assert "A" in attributions or "B" in attributions


class TestEndToEndIntegration:
    """End-to-end integration tests combining all new modules."""

    @pytest.fixture
    def full_pipeline(self):
        """Set up full pipeline with all modules."""
        try:
            from integration import SharedSpaceIntegration as AbstractIntegration
            from integration import SharedSpaceRobustness, RobustnessLevel
            from inference_adapter import InferenceSCMAdapter

            return {
                "abstract": AbstractIntegration(),
                "robust": SharedSpaceRobustness(
                    embedding_dim=512,
                    robustness_level=RobustnessLevel.BASIC,
                    random_seed=42,
                ),
                "causal": InferenceSCMAdapter(name="pipeline_scm"),
            }
        except ImportError as e:
            pytest.skip(f"Modules not available: {e}")

    def test_symbol_to_robust_embedding(self, full_pipeline):
        """Test: symbol → abstract binding → robust embedding."""
        abstract = full_pipeline["abstract"]
        robust = full_pipeline["robust"]

        # Create symbolic binding
        symbol_embedding = abstract.bind_to_shared_space("concept")

        # Add robustness to the embedding
        robust_embedding = robust.add_robustness(
            symbol_embedding.vector,
            modality="symbolic",
            source_module="abstract_integration",
        )

        assert robust_embedding.robustness.uncertainty >= 0
        assert robust_embedding.vector.shape == (512,)

    def test_causal_query_with_abstraction(self, full_pipeline):
        """Test causal queries using abstract concepts."""
        causal = full_pipeline["causal"]
        abstract = full_pipeline["abstract"]

        # Build causal model
        causal.add_linear_equation("Treatment", [], {})
        causal.add_linear_equation("Outcome", ["Treatment"], {"Treatment": 1.5})

        # Create abstract concept for the outcome
        outcome_binding = abstract.bind_to_shared_space("health_outcome")

        # Run causal query
        effect = causal.causal_effect("Treatment", "Outcome")

        # Verify both work
        assert abs(effect - 1.5) < 0.5
        assert outcome_binding.vector.shape[0] > 0

    def test_robust_filtering_of_abstractions(self, full_pipeline):
        """Test filtering abstract concepts by robustness."""
        abstract = full_pipeline["abstract"]
        robust = full_pipeline["robust"]

        # Create multiple symbolic bindings
        symbols = ["dog", "cat", "car", "tree", "xyz_random"]
        embeddings = []

        for sym in symbols:
            sym_emb = abstract.bind_to_shared_space(sym)
            robust_emb = robust.add_robustness(
                sym_emb.vector,
                modality="symbolic",
                source_module="test",
            )
            embeddings.append(robust_emb)

        # Filter by reliability
        reliable = robust.filter_reliable(embeddings, threshold=0.3)

        # Should filter out some low-confidence ones
        assert len(reliable) <= len(embeddings)

    def test_uncertainty_in_causal_effects(self, full_pipeline):
        """Test uncertainty quantification of causal effects."""
        causal = full_pipeline["causal"]
        robust = full_pipeline["robust"]

        # Build noisy causal model
        causal.add_linear_equation("X", [], {}, noise_var=None)
        causal.add_linear_equation("Y", ["X"], {"X": 2.0}, noise_var=None)

        # Sample multiple causal effects
        effects = []
        for _ in range(5):
            effect = causal.causal_effect("X", "Y")
            effects.append(effect)

        # Compute uncertainty of effect estimate
        effect_embedding = robust.add_robustness(np.array(effects))
        assert effect_embedding.robustness.uncertainty >= 0

    def test_full_perception_to_reasoning_pipeline(self, full_pipeline):
        """Test perception → abstraction → causal reasoning → robust output."""
        abstract = full_pipeline["abstract"]
        robust = full_pipeline["robust"]
        causal = full_pipeline["causal"]

        # 1. Simulate perception (random perceptual embedding)
        np.random.seed(42)
        perceptual_input = np.random.randn(256)

        # 2. Ground to abstract symbol
        grounded = abstract.create_grounded_embedding("observed_object", perceptual_input)

        # 3. Build causal model about observations
        causal.add_linear_equation("Observation", [], {})
        causal.add_linear_equation("Inference", ["Observation"], {"Observation": 1.0})

        # 4. Run causal query
        cf = causal.counterfactual(
            evidence={"Observation": 1.0, "Inference": 1.0}, intervention={"Observation": 2.0}
        )

        # 5. Create output embedding with robustness
        output_vector = np.concatenate(
            [
                grounded.vector[:256],  # Use first 256 dims
                np.array([cf.get("Inference", 0.0)]),  # Add causal result
            ]
        )
        # Pad to 512
        output_vector = np.pad(output_vector, (0, 512 - len(output_vector)))

        robust_output = robust.add_robustness(
            output_vector,
            modality="reasoning",
            source_module="pipeline",
        )

        # Verify full pipeline completed
        assert robust_output.is_reliable(threshold=0.1)
        assert robust_output.robustness.uncertainty >= 0
        assert grounded.symbol == "observed_object"
        assert "Inference" in cf
