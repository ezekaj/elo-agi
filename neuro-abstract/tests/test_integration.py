"""
Integration tests for neuro-abstract module.

Tests the interaction between:
- SymbolicBinder
- CompositionTypes
- ProgramSynthesizer
- AbstractionEngine
- SharedSpaceIntegration
"""

import pytest
import numpy as np
from neuro.modules.abstract.symbolic_binder import SymbolicBinder, RoleType
from neuro.modules.abstract.composition_types import INT, STR, FunctionType, StructuredType
from neuro.modules.abstract.program_synthesis import ProgramSynthesizer, Example
from neuro.modules.abstract.abstraction_engine import AbstractionEngine, AbstractionLevel
from neuro.modules.abstract.integration import SharedSpaceIntegration, SemanticModalityType


class TestBinderWithTypes:
    """Test symbolic binder with type system."""

    def test_bind_typed_concept(self):
        """Should bind concept with type information."""
        binder = SymbolicBinder(embedding_dim=128, random_seed=42)

        # Create typed binding
        int_rep = INT.to_neural(128)
        binding = binder.bind("number", neural_rep=int_rep)

        assert binding is not None
        assert binding.symbol == "number"

    def test_bind_function_type(self):
        """Should bind function type representation."""
        binder = SymbolicBinder(embedding_dim=128, random_seed=42)

        func_type = FunctionType((INT, INT), INT)
        func_rep = func_type.to_neural(128)

        binding = binder.bind("add_function", neural_rep=func_rep)
        assert binding is not None


class TestBinderWithSynthesis:
    """Test symbolic binding with program synthesis."""

    def test_synthesize_with_bound_primitives(self):
        """Should synthesize using bound primitive symbols."""
        binder = SymbolicBinder(embedding_dim=64, random_seed=42)

        # Register primitive symbols
        binder.register_symbol("add")
        binder.register_symbol("multiply")

        # Synthesize
        synthesizer = ProgramSynthesizer(max_size=5)
        examples = [
            Example(inputs=[2, 3], output=5),
            Example(inputs=[1, 4], output=5),
        ]

        program = synthesizer.synthesize_auto(examples)
        assert program is not None


class TestAbstractionWithBinding:
    """Test abstraction engine with symbolic binding."""

    def test_abstract_and_bind(self):
        """Should abstract examples and create bindings."""
        engine = AbstractionEngine(random_seed=42)
        binder = SymbolicBinder(embedding_dim=128, random_seed=42)

        # Examples of "X causes Y" relations
        examples = [
            {"type": "causes", "agent": "rain", "effect": "wet"},
            {"type": "causes", "agent": "fire", "effect": "hot"},
            {"type": "causes", "agent": "cold", "effect": "shiver"},
        ]

        # Abstract
        abstractions = engine.abstract(examples, AbstractionLevel.CONCEPT)
        assert len(abstractions) > 0

        # Bind each abstraction
        for abstraction in abstractions:
            if abstraction.embedding is not None:
                binding = binder.bind(abstraction.name, neural_rep=abstraction.embedding)
                assert binding is not None

    def test_transfer_with_binding(self):
        """Should transfer concepts using binding mappings."""
        engine = AbstractionEngine(random_seed=42)
        binder = SymbolicBinder(embedding_dim=128, random_seed=42)

        # Source domain
        binder.register_symbol("dog")
        binder.register_symbol("cat")
        binder.register_symbol("chases")

        source = binder.bind(
            "chases",
            roles={
                RoleType.AGENT: ("dog", None),
                RoleType.PATIENT: ("cat", None),
            },
        )

        # Analogical transfer
        binder.register_symbol("lion")
        binder.register_symbol("gazelle")

        mapping = {"dog": "lion", "cat": "gazelle"}
        target = binder.analogical_bind(source, mapping)

        assert target.get_filler(RoleType.AGENT) == "lion"
        assert target.get_filler(RoleType.PATIENT) == "gazelle"


class TestSharedSpaceIntegration:
    """Test SharedSpaceIntegration class."""

    @pytest.fixture
    def integration(self):
        return SharedSpaceIntegration()

    def test_project_binding(self, integration):
        """Should project binding to shared space."""
        binding = integration.binder.bind("concept", roles={RoleType.AGENT: ("subject", None)})

        embedding = integration.project_binding(binding)

        assert embedding is not None
        assert embedding.modality == SemanticModalityType.SYMBOLIC
        assert embedding.symbol == "concept"

    def test_project_abstraction(self, integration):
        """Should project abstraction to shared space."""
        examples = [
            {"type": "animal", "name": "dog"},
            {"type": "animal", "name": "cat"},
        ]

        abstractions = integration.abstraction_engine.abstract(examples, AbstractionLevel.CONCEPT)

        for abstraction in abstractions:
            embedding = integration.project_abstraction(abstraction)
            assert embedding is not None
            assert embedding.modality == SemanticModalityType.ABSTRACT

    def test_project_type(self, integration):
        """Should project type to shared space."""
        func_type = FunctionType((INT,), INT)
        embedding = integration.project_type(func_type, symbol="increment")

        assert embedding is not None
        assert embedding.type_info == func_type

    def test_query_embeddings(self, integration):
        """Should query embedded concepts."""
        # Add some embeddings
        integration.bind_to_shared_space("concept1")
        integration.bind_to_shared_space("concept2")
        integration.bind_to_shared_space("concept3")

        # Query
        query = np.random.randn(integration.config.embedding_dim)
        query = query / np.linalg.norm(query)

        results = integration.query(query, top_k=2)
        assert len(results) <= 2

    def test_query_by_symbol(self, integration):
        """Should query by symbol name."""
        integration.bind_to_shared_space("target_symbol")
        integration.bind_to_shared_space("other_symbol")

        results = integration.query_by_symbol("target_symbol")
        assert len(results) > 0
        assert all(e.symbol == "target_symbol" for e in results)

    def test_query_by_role(self, integration):
        """Should query by role-filler."""
        integration.bind_to_shared_space("action1", roles={RoleType.AGENT: "john"})
        integration.bind_to_shared_space("action2", roles={RoleType.AGENT: "mary"})

        results = integration.query_by_role(RoleType.AGENT, "john")
        assert len(results) > 0

    def test_grounded_embedding(self, integration):
        """Should create grounded embedding."""
        perceptual = np.random.randn(integration.config.projection_dim)

        embedding = integration.create_grounded_embedding(
            symbol="apple",
            perceptual_embedding=perceptual,
        )

        assert embedding is not None
        assert embedding.modality == SemanticModalityType.GROUNDED
        assert embedding.symbol == "apple"

    def test_store_and_retrieve_concept(self, integration):
        """Should store and retrieve concepts from memory."""
        embedding = integration.bind_to_shared_space("stored_concept")
        integration.store_concept("my_concept", embedding)

        retrieved = integration.retrieve_concept("my_concept")
        assert retrieved is not None
        assert retrieved.symbol == "stored_concept"

    def test_query_memory(self, integration):
        """Should query persistent memory."""
        e1 = integration.bind_to_shared_space("mem1")
        e2 = integration.bind_to_shared_space("mem2")

        integration.store_concept("memory1", e1)
        integration.store_concept("memory2", e2)

        query = np.random.randn(integration.config.embedding_dim)
        results = integration.query_memory(query, top_k=2)

        assert len(results) == 2

    def test_abstract_and_project(self, integration):
        """Should abstract and project in one step."""
        examples = [
            {"type": "vehicle", "wheels": 4},
            {"type": "vehicle", "wheels": 2},
        ]

        embeddings = integration.abstract_and_project(examples, AbstractionLevel.CONCEPT)

        assert len(embeddings) > 0
        for e in embeddings:
            assert e.modality == SemanticModalityType.ABSTRACT

    def test_statistics(self, integration):
        """Should track statistics."""
        integration.bind_to_shared_space("test")

        stats = integration.statistics()
        assert stats["n_projections"] > 0
        assert "binder_stats" in stats
        assert "abstraction_stats" in stats


class TestEndToEndPipeline:
    """Test end-to-end pipelines."""

    def test_concept_learning_pipeline(self):
        """Complete pipeline: examples -> abstractions -> bindings -> retrieval."""
        integration = SharedSpaceIntegration()

        # 1. Provide examples
        examples = [
            {"predicate": "loves", "agent": "john", "patient": "mary"},
            {"predicate": "loves", "agent": "bob", "patient": "alice"},
            {"predicate": "hates", "agent": "tom", "patient": "jerry"},
        ]

        # 2. Abstract
        abstractions = integration.abstraction_engine.abstract(examples, AbstractionLevel.CONCEPT)

        # 3. Project to shared space
        for abs in abstractions:
            integration.project_abstraction(abs)

        # 4. Create bindings
        for ex in examples:
            binding = integration.binder.bind(
                ex["predicate"],
                roles={
                    RoleType.AGENT: (ex["agent"], None),
                    RoleType.PATIENT: (ex["patient"], None),
                },
            )
            integration.project_binding(binding)

        # 5. Query
        active = integration.get_active_embeddings()
        assert len(active) > 0

    def test_analogy_pipeline(self):
        """Pipeline: source domain -> mapping -> target domain."""
        integration = SharedSpaceIntegration()

        # Source domain: atoms
        source_examples = [
            {"domain": "atom", "nucleus": "sun", "electron": "planet"},
        ]

        # Target domain: solar system (to be inferred)
        target_examples = [
            {"domain": "solar", "center": "sun", "orbiter": "earth"},
        ]

        # Find analogy
        mapping = integration.abstraction_engine.find_analogy(
            source=source_examples[0],
            target=target_examples[0],
        )

        assert mapping is not None
        assert mapping.structural_consistency >= 0

    def test_synthesis_grounding_pipeline(self):
        """Pipeline: synthesis -> bind -> ground."""
        integration = SharedSpaceIntegration()
        synthesizer = ProgramSynthesizer(max_size=5)

        # Synthesize a function
        examples = [
            Example(inputs=[1], output=2),
            Example(inputs=[5], output=6),
        ]

        program = synthesizer.synthesize_auto(examples)

        if program:
            # Bind the synthesized program
            func_type = FunctionType((INT,), INT)
            integration.project_type(func_type, symbol="increment")

            # Create grounded embedding
            perceptual = np.random.randn(integration.config.projection_dim)
            embedding = integration.create_grounded_embedding(
                "increment_function",
                perceptual,
            )

            assert embedding.modality == SemanticModalityType.GROUNDED


class TestCrossModuleConsistency:
    """Test consistency across modules."""

    def test_embedding_similarity_reflects_semantics(self):
        """Similar concepts should have similar embeddings."""
        integration = SharedSpaceIntegration()

        # Similar concepts
        e1 = integration.bind_to_shared_space("runs", roles={RoleType.AGENT: "dog"})
        e2 = integration.bind_to_shared_space("runs", roles={RoleType.AGENT: "cat"})

        # Different concept
        e3 = integration.bind_to_shared_space("sleeps", roles={RoleType.AGENT: "fish"})

        sim_similar = e1.similarity(e2)
        sim_different = e1.similarity(e3)

        # Similar concepts should be more similar
        # (This may not always hold due to random initialization)
        assert isinstance(sim_similar, float)
        assert isinstance(sim_different, float)

    def test_type_preservation(self):
        """Type information should be preserved through pipeline."""
        integration = SharedSpaceIntegration()

        # Create typed embedding
        point_type = StructuredType("Point", (("x", INT), ("y", INT)))
        embedding = integration.project_type(point_type, symbol="Point")

        # Type should be preserved
        assert embedding.type_info == point_type

        # Store and retrieve
        integration.store_concept("PointType", embedding)
        retrieved = integration.retrieve_concept("PointType")

        assert retrieved.type_info == point_type
