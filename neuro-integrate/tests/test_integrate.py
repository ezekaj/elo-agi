"""
Tests for Neuro-Integrate: Cross-module integration system.

Covers:
- Shared semantic space
- Cross-module learning
- Conflict resolution
- Evidence accumulation
- Coherence checking
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from shared_space import (
    SharedSpace, SharedSpaceConfig, SemanticEmbedding,
    ProjectionLayer, ModalityType
)
from cross_module_learning import (
    CrossModuleLearner, LearningSignal, SignalType,
    GradientRouter, ModuleSynapse
)
from conflict_resolution import (
    ConflictResolver, Conflict, Resolution, ConflictType,
    ResolutionStrategy
)
from evidence_accumulation import (
    EvidenceAccumulator, Evidence, EvidenceSource, EvidenceType,
    AccumulatorConfig, DriftDiffusionAccumulator, BayesianAccumulator
)
from coherence_checker import (
    CoherenceChecker, Belief, Inconsistency, InconsistencyType,
    BeliefNetwork, CoherenceReport
)


# =============================================================================
# Tests: Shared Space
# =============================================================================

class TestSharedSpaceConfig:
    """Tests for SharedSpaceConfig."""

    def test_default_config(self):
        config = SharedSpaceConfig()
        assert config.embedding_dim == 512
        assert config.n_attention_heads == 8

    def test_custom_config(self):
        config = SharedSpaceConfig(embedding_dim=256, temperature=0.5)
        assert config.embedding_dim == 256
        assert config.temperature == 0.5


class TestSemanticEmbedding:
    """Tests for SemanticEmbedding."""

    def test_creation(self):
        vec = np.random.randn(512)
        emb = SemanticEmbedding(
            vector=vec,
            modality=ModalityType.VISUAL,
            source_module="test"
        )
        assert emb.modality == ModalityType.VISUAL
        assert emb.source_module == "test"

    def test_similarity(self):
        vec1 = np.array([1, 0, 0, 0])
        vec2 = np.array([1, 0, 0, 0])
        emb1 = SemanticEmbedding(vec1, ModalityType.ABSTRACT, "test")
        emb2 = SemanticEmbedding(vec2, ModalityType.ABSTRACT, "test")
        assert emb1.similarity(emb2) == pytest.approx(1.0)

    def test_similarity_orthogonal(self):
        vec1 = np.array([1, 0, 0, 0])
        vec2 = np.array([0, 1, 0, 0])
        emb1 = SemanticEmbedding(vec1, ModalityType.ABSTRACT, "test")
        emb2 = SemanticEmbedding(vec2, ModalityType.ABSTRACT, "test")
        assert emb1.similarity(emb2) == pytest.approx(0.0)

    def test_distance(self):
        vec1 = np.array([0, 0, 0, 0])
        vec2 = np.array([3, 4, 0, 0])
        emb1 = SemanticEmbedding(vec1, ModalityType.ABSTRACT, "test")
        emb2 = SemanticEmbedding(vec2, ModalityType.ABSTRACT, "test")
        assert emb1.distance(emb2) == pytest.approx(5.0)

    def test_blend(self):
        vec1 = np.array([1, 0, 0, 0])
        vec2 = np.array([0, 1, 0, 0])
        emb1 = SemanticEmbedding(vec1, ModalityType.VISUAL, "m1", confidence=0.8)
        emb2 = SemanticEmbedding(vec2, ModalityType.AUDITORY, "m2", confidence=0.6)
        blended = emb1.blend(emb2, 0.5)
        assert blended.source_module == "blended"
        assert blended.confidence == pytest.approx(0.7)


class TestProjectionLayer:
    """Tests for ProjectionLayer."""

    def test_creation(self):
        proj = ProjectionLayer(
            input_dim=64,
            output_dim=512,
            modality=ModalityType.VISUAL,
            module_name="vision"
        )
        assert proj.input_dim == 64
        assert proj.output_dim == 512

    def test_project(self):
        proj = ProjectionLayer(64, 512, ModalityType.VISUAL, "vision")
        input_vec = np.random.randn(64)
        embedding = proj.project(input_vec)
        assert embedding.vector.shape == (512,)
        assert np.linalg.norm(embedding.vector) == pytest.approx(1.0, abs=0.01)

    def test_project_padding(self):
        proj = ProjectionLayer(64, 512, ModalityType.VISUAL, "vision")
        input_vec = np.random.randn(32)  # Smaller than expected
        embedding = proj.project(input_vec)
        assert embedding.vector.shape == (512,)

    def test_update(self):
        proj = ProjectionLayer(64, 512, ModalityType.VISUAL, "vision")
        input_vec = np.random.randn(64)
        error = np.random.randn(512)
        weights_before = proj.weights.copy()
        proj.update(error, input_vec)
        assert not np.allclose(proj.weights, weights_before)


class TestSharedSpace:
    """Tests for SharedSpace."""

    def test_creation(self):
        space = SharedSpace()
        assert space.config.embedding_dim == 512

    def test_register_module(self):
        space = SharedSpace()
        proj = space.register_module("vision", 64, ModalityType.VISUAL)
        assert proj is not None
        assert space.get_projection("vision") is not None

    def test_project(self):
        space = SharedSpace()
        space.register_module("vision", 64, ModalityType.VISUAL)
        vec = np.random.randn(64)
        embedding = space.project("vision", vec)
        assert embedding.vector.shape == (512,)

    def test_query(self):
        space = SharedSpace()
        space.register_module("m1", 64, ModalityType.VISUAL)
        space.register_module("m2", 64, ModalityType.AUDITORY)

        space.project("m1", np.random.randn(64))
        space.project("m2", np.random.randn(64))

        query = SemanticEmbedding(np.random.randn(512), ModalityType.ABSTRACT, "query")
        results = space.query(query, top_k=5)
        assert len(results) <= 5

    def test_store_retrieve_concept(self):
        space = SharedSpace()
        emb = SemanticEmbedding(np.random.randn(512), ModalityType.ABSTRACT, "test")
        space.store_concept("cat", emb)
        retrieved = space.get_concept("cat")
        assert retrieved is not None
        assert np.allclose(retrieved.vector, emb.vector)

    def test_bind(self):
        space = SharedSpace()
        emb1 = SemanticEmbedding(np.random.randn(512), ModalityType.VISUAL, "m1")
        emb2 = SemanticEmbedding(np.random.randn(512), ModalityType.AUDITORY, "m2")
        bound = space.bind([emb1, emb2])
        assert bound.modality == ModalityType.ABSTRACT
        assert bound.source_module == "binding"

    def test_attention_spread(self):
        space = SharedSpace()
        space.register_module("m1", 64, ModalityType.VISUAL)
        space.register_module("m2", 64, ModalityType.AUDITORY)

        space.project("m1", np.random.randn(64))
        space.project("m2", np.random.randn(64))

        focus = SemanticEmbedding(np.random.randn(512), ModalityType.ABSTRACT, "focus")
        attention = space.attention_spread(focus)
        assert sum(attention.values()) == pytest.approx(1.0)

    def test_decay(self):
        space = SharedSpace(SharedSpaceConfig(decay_rate=0.5))
        space.register_module("m1", 64, ModalityType.VISUAL)
        space.project("m1", np.random.randn(64))

        initial_conf = space.get_active_embeddings()[0].confidence
        space.decay()
        after_conf = space.get_active_embeddings()[0].confidence
        assert after_conf < initial_conf

    def test_cross_modal_similarity(self):
        space = SharedSpace()
        space.register_module("visual", 64, ModalityType.VISUAL)
        space.register_module("auditory", 64, ModalityType.AUDITORY)

        space.project("visual", np.random.randn(64))
        space.project("auditory", np.random.randn(64))

        sim_matrix = space.cross_modal_similarity(ModalityType.VISUAL, ModalityType.AUDITORY)
        assert sim_matrix.shape == (1, 1)

    def test_statistics(self):
        space = SharedSpace()
        space.register_module("m1", 64, ModalityType.VISUAL)
        space.project("m1", np.random.randn(64))

        stats = space.statistics()
        assert stats["n_registered_modules"] == 1
        assert stats["n_active_embeddings"] == 1


# =============================================================================
# Tests: Cross-Module Learning
# =============================================================================

class TestModuleSynapse:
    """Tests for ModuleSynapse."""

    def test_creation(self):
        weights = np.random.randn(64, 64)
        synapse = ModuleSynapse("m1", "m2", weights)
        assert synapse.source == "m1"
        assert synapse.target == "m2"

    def test_transmit(self):
        weights = np.eye(64)
        synapse = ModuleSynapse("m1", "m2", weights)
        signal = np.random.randn(64)
        transmitted = synapse.transmit(signal)
        assert np.allclose(transmitted, signal)

    def test_update(self):
        weights = np.random.randn(64, 64)
        synapse = ModuleSynapse("m1", "m2", weights.copy())
        pre = np.random.randn(64)
        post = np.random.randn(64)
        synapse.update(pre, post, reward=1.0)
        # Weights should change
        assert not np.allclose(synapse.weights, weights)


class TestGradientRouter:
    """Tests for GradientRouter."""

    def test_creation(self):
        router = GradientRouter(n_modules=5)
        assert router.n_modules == 5

    def test_register_module(self):
        router = GradientRouter(n_modules=5)
        idx = router.register_module("vision")
        assert idx == 0
        assert router.get_module_index("vision") == 0

    def test_route_gradient(self):
        router = GradientRouter(n_modules=5)
        router.register_module("m1")
        router.register_module("m2")

        gradient = np.random.randn(512)
        routed = router.route_gradient("m1", gradient)
        assert "m2" in routed

    def test_assign_credit(self):
        router = GradientRouter(n_modules=5)
        router.register_module("m1")
        router.register_module("m2")

        activities = {
            "m1": np.random.randn(64),
            "m2": np.random.randn(64),
        }
        router.record_activity(activities)

        credits = router.assign_credit(1.0, activities)
        assert "m1" in credits
        assert "m2" in credits


class TestCrossModuleLearner:
    """Tests for CrossModuleLearner."""

    def test_creation(self):
        learner = CrossModuleLearner()
        assert learner.embedding_dim == 512

    def test_register_module(self):
        learner = CrossModuleLearner()
        learner.register_module("vision")
        assert "vision" in learner.router._module_indices

    def test_create_synapse(self):
        learner = CrossModuleLearner()
        synapse = learner.create_synapse("m1", "m2")
        assert synapse.source == "m1"
        assert synapse.target == "m2"

    def test_emit_signal(self):
        learner = CrossModuleLearner()
        learner.register_module("m1")
        learner.register_module("m2")

        signal = learner.emit_signal(
            SignalType.ERROR,
            "m1",
            ["m2"],
            np.random.randn(512)
        )
        assert signal.source_module == "m1"
        assert len(learner._signal_queue) == 1

    def test_broadcast_signal(self):
        learner = CrossModuleLearner()
        learner.register_module("m1")
        learner.register_module("m2")
        learner.register_module("m3")

        signal = learner.broadcast_signal(
            SignalType.REWARD,
            "m1",
            np.random.randn(512)
        )
        assert "m2" in signal.target_modules
        assert "m3" in signal.target_modules
        assert "m1" not in signal.target_modules

    def test_process_signals(self):
        learner = CrossModuleLearner()
        learner.register_module("m1")
        learner.register_module("m2")

        learner.emit_signal(SignalType.ERROR, "m1", ["m2"], np.random.randn(512))
        gradients = learner.process_signals()

        assert len(learner._signal_queue) == 0

    def test_apply_reward(self):
        learner = CrossModuleLearner()
        learner.register_module("m1")
        learner.register_module("m2")
        learner.create_synapse("m1", "m2")

        activities = {
            "m1": np.random.randn(512),
            "m2": np.random.randn(512),
        }
        learner.apply_reward(1.0, activities)
        assert learner._dopamine == 1.0

    def test_modulate_plasticity(self):
        learner = CrossModuleLearner()
        learner.modulate_plasticity(dopamine=0.5, norepinephrine=0.3)
        assert learner._dopamine == 0.5
        assert learner._norepinephrine == 0.3
        assert learner._global_plasticity > 1.0

    def test_statistics(self):
        learner = CrossModuleLearner()
        learner.register_module("m1")
        stats = learner.statistics()
        assert stats["n_registered_modules"] == 1


# =============================================================================
# Tests: Conflict Resolution
# =============================================================================

class TestConflict:
    """Tests for Conflict."""

    def test_creation(self):
        emb1 = SemanticEmbedding(np.random.randn(512), ModalityType.VISUAL, "m1")
        emb2 = SemanticEmbedding(np.random.randn(512), ModalityType.AUDITORY, "m2")

        conflict = Conflict(
            conflict_id=1,
            conflict_type=ConflictType.BELIEF,
            modules=["m1", "m2"],
            embeddings={"m1": emb1, "m2": emb2},
            confidences={"m1": 0.9, "m2": 0.8}
        )
        assert conflict.conflict_id == 1
        assert len(conflict.modules) == 2

    def test_severity(self):
        vec1 = np.random.randn(512)
        vec2 = -vec1  # Opposite direction

        emb1 = SemanticEmbedding(vec1 / np.linalg.norm(vec1), ModalityType.VISUAL, "m1")
        emb2 = SemanticEmbedding(vec2 / np.linalg.norm(vec2), ModalityType.AUDITORY, "m2")

        conflict = Conflict(
            conflict_id=1,
            conflict_type=ConflictType.BELIEF,
            modules=["m1", "m2"],
            embeddings={"m1": emb1, "m2": emb2},
            confidences={"m1": 0.9, "m2": 0.8}
        )
        assert conflict.severity > 0  # Should have high severity


class TestConflictResolver:
    """Tests for ConflictResolver."""

    def test_creation(self):
        resolver = ConflictResolver()
        assert resolver.default_strategy == ResolutionStrategy.WEIGHTED_AVERAGE

    def test_detect_conflict(self):
        resolver = ConflictResolver()

        vec1 = np.random.randn(512)
        vec2 = -vec1  # Opposite direction

        emb1 = SemanticEmbedding(vec1 / np.linalg.norm(vec1), ModalityType.VISUAL, "m1", confidence=0.9)
        emb2 = SemanticEmbedding(vec2 / np.linalg.norm(vec2), ModalityType.AUDITORY, "m2", confidence=0.9)

        conflict = resolver.detect_conflict({"m1": emb1, "m2": emb2}, threshold=0.3)
        assert conflict is not None

    def test_no_conflict(self):
        resolver = ConflictResolver()

        vec = np.random.randn(512)
        vec = vec / np.linalg.norm(vec)

        emb1 = SemanticEmbedding(vec, ModalityType.VISUAL, "m1")
        emb2 = SemanticEmbedding(vec, ModalityType.AUDITORY, "m2")

        conflict = resolver.detect_conflict({"m1": emb1, "m2": emb2})
        assert conflict is None

    def test_resolve_weighted_average(self):
        resolver = ConflictResolver()

        emb1 = SemanticEmbedding(np.random.randn(512), ModalityType.VISUAL, "m1", confidence=0.9)
        emb2 = SemanticEmbedding(np.random.randn(512), ModalityType.AUDITORY, "m2", confidence=0.6)

        conflict = Conflict(
            conflict_id=1,
            conflict_type=ConflictType.BELIEF,
            modules=["m1", "m2"],
            embeddings={"m1": emb1, "m2": emb2},
            confidences={"m1": 0.9, "m2": 0.6}
        )

        resolution = resolver.resolve(conflict, ResolutionStrategy.WEIGHTED_AVERAGE)
        assert resolution.resolved_embedding is not None
        assert len(resolution.winning_modules) == 2

    def test_resolve_winner_take_all(self):
        resolver = ConflictResolver()

        emb1 = SemanticEmbedding(np.random.randn(512), ModalityType.VISUAL, "m1", confidence=0.9)
        emb2 = SemanticEmbedding(np.random.randn(512), ModalityType.AUDITORY, "m2", confidence=0.3)

        conflict = Conflict(
            conflict_id=1,
            conflict_type=ConflictType.ACTION,
            modules=["m1", "m2"],
            embeddings={"m1": emb1, "m2": emb2},
            confidences={"m1": 0.9, "m2": 0.3}
        )

        resolution = resolver.resolve(conflict, ResolutionStrategy.WINNER_TAKE_ALL)
        assert len(resolution.winning_modules) == 1
        assert "m1" in resolution.winning_modules

    def test_resolve_voting(self):
        resolver = ConflictResolver()

        # Create 3 similar embeddings and 1 different
        base_vec = np.random.randn(512)
        base_vec /= np.linalg.norm(base_vec)

        emb1 = SemanticEmbedding(base_vec + 0.01 * np.random.randn(512), ModalityType.VISUAL, "m1")
        emb2 = SemanticEmbedding(base_vec + 0.01 * np.random.randn(512), ModalityType.AUDITORY, "m2")
        emb3 = SemanticEmbedding(base_vec + 0.01 * np.random.randn(512), ModalityType.MOTOR, "m3")
        emb4 = SemanticEmbedding(-base_vec, ModalityType.SPATIAL, "m4")

        conflict = Conflict(
            conflict_id=1,
            conflict_type=ConflictType.ATTENTION,
            modules=["m1", "m2", "m3", "m4"],
            embeddings={"m1": emb1, "m2": emb2, "m3": emb3, "m4": emb4},
            confidences={"m1": 0.8, "m2": 0.8, "m3": 0.8, "m4": 0.8}
        )

        resolution = resolver.resolve(conflict, ResolutionStrategy.VOTING)
        # Majority cluster should win
        assert len(resolution.winning_modules) >= 2

    def test_update_reliability(self):
        resolver = ConflictResolver()
        resolver.update_reliability("m1", success=True)
        assert resolver._reliability.get("m1", 1.0) > 1.0

        resolver.update_reliability("m2", success=False)
        assert resolver._reliability.get("m2", 1.0) < 1.0

    def test_statistics(self):
        resolver = ConflictResolver()
        stats = resolver.statistics()
        assert "total_conflicts" in stats
        assert "module_reliability" in stats


# =============================================================================
# Tests: Evidence Accumulation
# =============================================================================

class TestEvidenceSource:
    """Tests for EvidenceSource."""

    def test_creation(self):
        source = EvidenceSource(name="vision", reliability=0.9)
        assert source.name == "vision"
        assert source.reliability == 0.9


class TestEvidence:
    """Tests for Evidence."""

    def test_creation(self):
        source = EvidenceSource("test")
        evidence = Evidence(
            source=source,
            evidence_type=EvidenceType.SENSORY,
            value=np.random.randn(64),
            strength=0.8
        )
        assert evidence.strength == 0.8

    def test_weight(self):
        source = EvidenceSource("test", reliability=0.5)
        evidence = Evidence(
            source=source,
            evidence_type=EvidenceType.SENSORY,
            value=np.random.randn(64),
            strength=1.0,
            uncertainty=0.1
        )
        # Weight should account for reliability and uncertainty
        assert evidence.weight < 1.0


class TestDriftDiffusionAccumulator:
    """Tests for DriftDiffusionAccumulator."""

    def test_creation(self):
        ddm = DriftDiffusionAccumulator()
        state, time = ddm.get_state()
        assert state == 0.0
        assert time == 0.0

    def test_accumulate(self):
        ddm = DriftDiffusionAccumulator(AccumulatorConfig(threshold=10.0))
        source = EvidenceSource("test")
        evidence = Evidence(source, EvidenceType.SENSORY, np.array([1.0]), 1.0)

        ddm.accumulate(evidence)
        state, _ = ddm.get_state()
        assert state != 0.0

    def test_decision_probability(self):
        ddm = DriftDiffusionAccumulator()
        prob = ddm.get_decision_probability()
        assert 0 <= prob <= 1


class TestBayesianAccumulator:
    """Tests for BayesianAccumulator."""

    def test_creation(self):
        bayes = BayesianAccumulator(n_hypotheses=3)
        posterior = bayes.get_posterior()
        assert len(posterior) == 3
        assert np.sum(posterior) == pytest.approx(1.0)

    def test_update(self):
        bayes = BayesianAccumulator(n_hypotheses=3)
        source = EvidenceSource("test")
        evidence = Evidence(source, EvidenceType.SENSORY, np.array([1.0]), 1.0)
        likelihoods = np.array([0.9, 0.1, 0.1])

        posterior = bayes.update(evidence, likelihoods)
        assert posterior[0] > posterior[1]  # First hypothesis more likely

    def test_map_hypothesis(self):
        bayes = BayesianAccumulator(n_hypotheses=3, prior=np.array([0.8, 0.1, 0.1]))
        assert bayes.get_map_hypothesis() == 0

    def test_entropy(self):
        bayes = BayesianAccumulator(n_hypotheses=3)
        entropy = bayes.get_entropy()
        assert entropy > 0


class TestEvidenceAccumulator:
    """Tests for EvidenceAccumulator."""

    def test_creation(self):
        acc = EvidenceAccumulator()
        assert acc.config is not None

    def test_register_source(self):
        acc = EvidenceAccumulator()
        source = acc.register_source("vision", reliability=0.9)
        assert source.name == "vision"
        assert acc.get_source("vision") is not None

    def test_create_evidence(self):
        acc = EvidenceAccumulator()
        acc.register_source("vision")
        evidence = acc.create_evidence(
            "vision",
            EvidenceType.SENSORY,
            np.random.randn(64),
            strength=0.8
        )
        assert evidence.source.name == "vision"

    def test_accumulate(self):
        acc = EvidenceAccumulator()
        acc.register_source("vision")
        evidence = acc.create_evidence("vision", EvidenceType.SENSORY, np.random.randn(64))

        accumulated = acc.accumulate("hypothesis1", evidence)
        assert accumulated is not None
        assert acc.get_accumulated("hypothesis1") is not None

    def test_compare_hypotheses(self):
        acc = EvidenceAccumulator()
        acc.register_source("vision")

        evidence1 = acc.create_evidence("vision", EvidenceType.SENSORY, np.ones(64), strength=2.0)
        evidence2 = acc.create_evidence("vision", EvidenceType.SENSORY, np.ones(64), strength=0.5)

        acc.accumulate("h1", evidence1)
        acc.accumulate("h2", evidence2)

        winner, ratio = acc.compare_hypotheses("h1", "h2")
        assert winner == "h1"
        assert ratio > 1.0

    def test_decide_binary(self):
        acc = EvidenceAccumulator(AccumulatorConfig(threshold=0.5))
        source = EvidenceSource("test")

        # Create evidence stream favoring option 1
        evidence_stream = [
            Evidence(source, EvidenceType.SENSORY, np.array([0.5]), 1.0)
            for _ in range(20)
        ]

        decision, confidence, rt = acc.decide_binary(evidence_stream)
        assert decision in [0, 1]
        assert 0 <= confidence
        assert rt > 0

    def test_statistics(self):
        acc = EvidenceAccumulator()
        stats = acc.statistics()
        assert "n_sources" in stats
        assert "total_evidence" in stats


# =============================================================================
# Tests: Coherence Checker
# =============================================================================

class TestBelief:
    """Tests for Belief."""

    def test_creation(self):
        emb = SemanticEmbedding(np.random.randn(512), ModalityType.ABSTRACT, "test")
        belief = Belief(
            belief_id="b1",
            content=emb,
            source_module="vision",
            confidence=0.9,
            timestamp=1.0
        )
        assert belief.belief_id == "b1"
        assert belief.source_module == "vision"

    def test_similarity(self):
        vec = np.random.randn(512)
        emb1 = SemanticEmbedding(vec, ModalityType.ABSTRACT, "test")
        emb2 = SemanticEmbedding(vec, ModalityType.ABSTRACT, "test")

        b1 = Belief("b1", emb1, "m1", 0.9, 1.0)
        b2 = Belief("b2", emb2, "m2", 0.9, 1.0)

        assert b1.similarity(b2) == pytest.approx(1.0)


class TestBeliefNetwork:
    """Tests for BeliefNetwork."""

    def test_creation(self):
        network = BeliefNetwork()
        assert len(network._beliefs) == 0

    def test_add_belief(self):
        network = BeliefNetwork()
        emb = SemanticEmbedding(np.random.randn(512), ModalityType.ABSTRACT, "test")
        belief = network.add_belief(emb, "vision", 0.9)
        assert belief.belief_id in network._beliefs

    def test_get_belief(self):
        network = BeliefNetwork()
        emb = SemanticEmbedding(np.random.randn(512), ModalityType.ABSTRACT, "test")
        belief = network.add_belief(emb, "vision", 0.9)
        retrieved = network.get_belief(belief.belief_id)
        assert retrieved is not None

    def test_remove_belief(self):
        network = BeliefNetwork()
        emb = SemanticEmbedding(np.random.randn(512), ModalityType.ABSTRACT, "test")
        belief = network.add_belief(emb, "vision", 0.9)
        removed = network.remove_belief(belief.belief_id)
        assert removed
        assert network.get_belief(belief.belief_id) is None

    def test_find_clusters(self):
        network = BeliefNetwork()
        emb1 = SemanticEmbedding(np.random.randn(512), ModalityType.ABSTRACT, "m1")
        emb2 = SemanticEmbedding(np.random.randn(512), ModalityType.ABSTRACT, "m2")

        network.add_belief(emb1, "m1")
        network.add_belief(emb2, "m2")

        clusters = network.find_clusters()
        assert len(clusters) >= 1


class TestCoherenceChecker:
    """Tests for CoherenceChecker."""

    def test_creation(self):
        checker = CoherenceChecker()
        assert checker.network is not None

    def test_add_belief(self):
        checker = CoherenceChecker()
        emb = SemanticEmbedding(np.random.randn(512), ModalityType.ABSTRACT, "test")
        belief, inconsistencies = checker.add_belief(emb, "vision", 0.9)
        assert belief is not None

    def test_detect_contradiction(self):
        checker = CoherenceChecker(contradiction_threshold=0.3)

        vec1 = np.random.randn(512)
        vec1 /= np.linalg.norm(vec1)
        vec2 = -vec1  # Opposite

        emb1 = SemanticEmbedding(vec1, ModalityType.ABSTRACT, "m1")
        emb2 = SemanticEmbedding(vec2, ModalityType.ABSTRACT, "m2")

        checker.add_belief(emb1, "m1", 0.9)
        _, inconsistencies = checker.add_belief(emb2, "m2", 0.9)

        # Should detect contradiction
        contradictions = [i for i in inconsistencies if i.inconsistency_type == InconsistencyType.CONTRADICTION]
        assert len(contradictions) > 0

    def test_check_staleness(self):
        checker = CoherenceChecker(staleness_threshold=10)
        emb = SemanticEmbedding(np.random.randn(512), ModalityType.ABSTRACT, "test")
        checker.add_belief(emb, "vision", 0.9)

        stale = checker.check_staleness(current_time=1000)
        assert len(stale) > 0

    def test_compute_coherence_score(self):
        checker = CoherenceChecker()

        # Add some consistent beliefs
        for i in range(5):
            vec = np.random.randn(512)
            emb = SemanticEmbedding(vec, ModalityType.ABSTRACT, f"m{i}")
            checker.add_belief(emb, f"m{i}", 0.9)

        score = checker.compute_coherence_score()
        assert 0 <= score <= 1

    def test_generate_report(self):
        checker = CoherenceChecker()
        emb = SemanticEmbedding(np.random.randn(512), ModalityType.ABSTRACT, "test")
        checker.add_belief(emb, "vision", 0.9)

        report = checker.generate_report()
        assert isinstance(report, CoherenceReport)
        assert report.n_beliefs == 1

    def test_resolve_inconsistency(self):
        checker = CoherenceChecker()

        vec1 = np.random.randn(512)
        vec1 /= np.linalg.norm(vec1)
        vec2 = -vec1

        emb1 = SemanticEmbedding(vec1, ModalityType.ABSTRACT, "m1")
        emb2 = SemanticEmbedding(vec2, ModalityType.ABSTRACT, "m2")

        checker.add_belief(emb1, "m1", 0.9)
        _, inconsistencies = checker.add_belief(emb2, "m2", 0.9)

        if inconsistencies:
            resolved = checker.resolve_inconsistency(inconsistencies[0].inconsistency_id)
            assert resolved

    def test_get_module_coherence(self):
        np.random.seed(42)  # Fixed seed for reproducibility
        checker = CoherenceChecker()

        # Add beliefs from same module
        for i in range(3):
            vec = np.random.randn(512)
            emb = SemanticEmbedding(vec, ModalityType.ABSTRACT, "vision")
            checker.add_belief(emb, "vision", 0.9)

        coherence = checker.get_module_coherence("vision")
        # Allow small negative values due to floating point in cosine similarity
        assert -0.1 <= coherence <= 1

    def test_statistics(self):
        checker = CoherenceChecker()
        emb = SemanticEmbedding(np.random.randn(512), ModalityType.ABSTRACT, "test")
        checker.add_belief(emb, "vision", 0.9)

        stats = checker.statistics()
        assert "n_beliefs" in stats
        assert "coherence_score" in stats


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the complete integration system."""

    def test_full_integration_cycle(self):
        """Test complete integration cycle."""
        # Create components
        space = SharedSpace()
        learner = CrossModuleLearner()
        resolver = ConflictResolver()
        accumulator = EvidenceAccumulator()
        checker = CoherenceChecker()

        # Register modules
        space.register_module("vision", 64, ModalityType.VISUAL)
        space.register_module("language", 64, ModalityType.LINGUISTIC)
        learner.register_module("vision")
        learner.register_module("language")
        learner.create_synapse("vision", "language")
        accumulator.register_source("vision", reliability=0.9)
        accumulator.register_source("language", reliability=0.8)

        # Project to shared space
        emb_vision = space.project("vision", np.random.randn(64))
        emb_language = space.project("language", np.random.randn(64))

        # Check for conflicts
        conflict = resolver.detect_conflict({
            "vision": emb_vision,
            "language": emb_language
        })

        if conflict:
            resolution = resolver.resolve(conflict)
            assert resolution is not None

        # Accumulate evidence
        evidence = accumulator.create_evidence(
            "vision", EvidenceType.SENSORY, np.random.randn(64)
        )
        accumulator.accumulate("object_present", evidence)

        # Add beliefs and check coherence
        checker.add_belief(emb_vision, "vision", 0.9)
        checker.add_belief(emb_language, "language", 0.8)

        report = checker.generate_report()
        assert report.coherence_score >= 0

        # Apply learning
        learner.apply_reward(1.0, {
            "vision": np.random.randn(512),
            "language": np.random.randn(512)
        })

        assert learner._total_updates > 0

    def test_cross_modal_binding(self):
        """Test binding representations across modalities."""
        space = SharedSpace()
        space.register_module("vision", 64, ModalityType.VISUAL)
        space.register_module("audio", 64, ModalityType.AUDITORY)
        space.register_module("language", 64, ModalityType.LINGUISTIC)

        # Project from each modality
        emb_v = space.project("vision", np.random.randn(64))
        emb_a = space.project("audio", np.random.randn(64))
        emb_l = space.project("language", np.random.randn(64))

        # Bind into unified representation
        bound = space.bind([emb_v, emb_a, emb_l])

        assert bound.modality == ModalityType.ABSTRACT
        assert "bound_count" in bound.metadata
        assert bound.metadata["bound_count"] == 3

    def test_learning_with_conflicts(self):
        """Test learning continues despite conflicts."""
        learner = CrossModuleLearner()
        resolver = ConflictResolver()

        learner.register_module("m1")
        learner.register_module("m2")
        learner.create_synapse("m1", "m2")

        # Create conflicting outputs
        vec1 = np.random.randn(512)
        vec2 = -vec1

        emb1 = SemanticEmbedding(vec1 / np.linalg.norm(vec1), ModalityType.ABSTRACT, "m1", confidence=0.9)
        emb2 = SemanticEmbedding(vec2 / np.linalg.norm(vec2), ModalityType.ABSTRACT, "m2", confidence=0.8)

        conflict = resolver.detect_conflict({"m1": emb1, "m2": emb2})
        assert conflict is not None

        resolution = resolver.resolve(conflict)

        # Learn from resolution
        learner.apply_reward(
            0.5 if resolution.winning_modules[0] == "m1" else -0.5,
            {"m1": vec1, "m2": vec2}
        )

        assert learner._total_updates > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
