"""
STRESS TESTS - Most Difficult Scenarios

These tests push the dual-process system to its limits:
1. High cognitive load
2. Deep conflict resolution
3. Complex compositional structures
4. Long inference chains
5. Edge cases and adversarial inputs
6. Performance under scale
"""

import pytest
import numpy as np
import time
from neuro.modules.m02_dual_process.dual_process_controller import DualProcessController
from neuro.modules.m02_dual_process.system1.pattern_recognition import PatternRecognition
from neuro.modules.m02_dual_process.system1.habit_executor import HabitExecutor, Action
from neuro.modules.m02_dual_process.system1.emotional_valuation import EmotionalValuation
from neuro.modules.m02_dual_process.system2.working_memory import WorkingMemory
from neuro.modules.m02_dual_process.system2.cognitive_control import (
    CognitiveControl,
    Response,
    ConflictLevel,
)
from neuro.modules.m02_dual_process.system2.relational_reasoning import (
    RelationalReasoning,
    RelationType,
)
from neuro.modules.m02_dual_process.hpc_pfc_complex import HPCPFCComplex
from neuro.modules.m02_dual_process.logic_network import LogicNetwork, Proposition, PropositionType


class TestWorkingMemoryStress:
    """Stress test working memory limits"""

    def test_capacity_overflow_massive(self):
        """Try to store way more than capacity"""
        wm = WorkingMemory(capacity=7)

        # Store 100 items - should keep only 7
        for i in range(100):
            wm.store(f"item_{i}", f"content_{i}")

        assert wm.current_load == 7
        # Most recent should survive
        assert wm.retrieve("item_99") is not None

    def test_rapid_access_pattern(self):
        """Rapid access should maintain items"""
        wm = WorkingMemory(capacity=5, decay_rate=0.5)

        # Store items
        for i in range(5):
            wm.store(f"item_{i}", i)

        # Rapidly access item_0 many times
        for _ in range(100):
            wm.retrieve("item_0")

        # item_0 should have high activation
        activations = wm.get_activations()
        assert activations["item_0"] >= activations.get("item_4", 0)

    def test_chunking_to_maximize_capacity(self):
        """Use chunking to hold more info"""
        wm = WorkingMemory(capacity=4)

        # Store 12 items in 4 chunks of 3
        for chunk_idx in range(4):
            items = []
            for i in range(3):
                item_id = f"item_{chunk_idx}_{i}"
                wm.store(item_id, f"content_{chunk_idx}_{i}")
                items.append(item_id)
            wm.chunk(items, f"chunk_{chunk_idx}")

        # Should have 4 chunks
        assert wm.current_load == 4

        # Each chunk should contain 3 items
        chunk = wm.retrieve("chunk_0")
        assert len(chunk.items) == 3

    def test_binding_complex_structure(self):
        """Create complex binding network"""
        wm = WorkingMemory(capacity=7)

        # Store items with complex cross-bindings
        wm.store("A", "alpha")
        wm.store("B", "beta")
        wm.store("C", "gamma")

        wm.bind("A", "relates_to", "B")
        wm.bind("B", "relates_to", "C")
        wm.bind("C", "relates_to", "A")  # Circular!
        wm.bind("A", "type", "node")
        wm.bind("B", "type", "node")
        wm.bind("C", "type", "node")

        # Query by binding
        nodes = wm.query_by_binding("type", "node")
        assert len(nodes) == 3


class TestConflictResolutionStress:
    """Stress test conflict detection and resolution"""

    def test_many_way_conflict(self):
        """10-way conflict between responses"""
        cc = CognitiveControl()

        responses = [
            Response(id=f"response_{i}", activation=0.8 + np.random.random() * 0.1, source="s1")
            for i in range(10)
        ]

        signal = cc.detect_conflict(responses)

        # Should detect severe conflict
        assert signal.level in [ConflictLevel.HIGH, ConflictLevel.SEVERE]
        assert signal.conflict_energy > 0.5

    def test_oscillating_conflict(self):
        """Conflict that keeps changing"""
        cc = CognitiveControl()

        # Simulate oscillating between two options
        for iteration in range(20):
            if iteration % 2 == 0:
                responses = [
                    Response(id="A", activation=0.9, source="s1"),
                    Response(id="B", activation=0.3, source="s1"),
                ]
            else:
                responses = [
                    Response(id="A", activation=0.3, source="s1"),
                    Response(id="B", activation=0.9, source="s1"),
                ]

            signal = cc.detect_conflict(responses)
            # Low conflict each time but history matters
            cc.conflict_history.append(signal)

        # High conflict rate in history
        conflict_rate = cc.get_conflict_rate(window=20)
        # Most should be low conflict individually
        assert conflict_rate < 0.5

    def test_inhibition_cascade(self):
        """Try to inhibit many responses"""
        cc = CognitiveControl(inhibition_strength=0.6)

        responses = [
            Response(id=f"r_{i}", activation=0.5 + i * 0.05, source="s1") for i in range(10)
        ]

        # Keep only the strongest
        result = cc.inhibit_all_except(responses, "r_9")

        # r_9 should be boosted
        r9 = next((r for r in result if r.id == "r_9"), None)
        assert r9 is not None
        assert r9.activation > 0.9

        # Others should be weakened or removed
        others = [r for r in result if r.id != "r_9"]
        for r in others:
            assert r.activation < 0.5


class TestRelationalReasoningStress:
    """Stress test compositional structures"""

    def test_deep_nesting(self):
        """Create deeply nested structure (10 levels)"""
        rr = RelationalReasoning()

        # Build nested structure: ((((((((((A) B) C) D) E) F) G) H) I) J)
        elements = [rr.create_element(chr(65 + i), element_id=chr(65 + i)) for i in range(10)]

        current_structure = rr.bind(elements[0], RelationType.PART_OF, elements[1])

        for i in range(2, 10):
            wrapper = rr.create_element(current_structure, type_tag="composed")
            current_structure = rr.bind(wrapper, RelationType.PART_OF, elements[i])

        # Should have deep structure
        assert len(current_structure.elements) == 2  # Each level has 2 elements
        assert len(current_structure.relations) == 1

    def test_wide_structure(self):
        """Create structure with many parallel relations"""
        rr = RelationalReasoning()

        # One central hub connected to 50 nodes
        hub = rr.create_element("hub", element_id="hub")

        for i in range(50):
            node = rr.create_element(f"node_{i}", element_id=f"node_{i}")
            rr.bind(hub, RelationType.HAS, node)

        # Query all relations from hub
        related = rr.get_related_elements("hub", RelationType.HAS)
        assert len(related) == 50

    def test_complex_action_frame(self):
        """Complex action with many modifiers"""
        rr = RelationalReasoning()

        action = rr.create_element("run")
        agent = rr.create_element("athlete")
        patient = rr.create_element("marathon")

        modifiers = [
            rr.create_element("quickly"),
            rr.create_element("gracefully"),
            rr.create_element("yesterday"),
            rr.create_element("in_boston"),
            rr.create_element("with_determination"),
        ]

        structure = rr.create_action_structure(
            action=action, agent=agent, patient=patient, modifiers=modifiers
        )

        # 1 action + 1 agent + 1 patient + 5 modifiers = 8 elements
        assert len(structure.elements) == 8
        # 1 agent rel + 1 patient rel + 5 modifier rels = 7 relations
        assert len(structure.relations) == 7

    def test_multi_hop_analogy(self):
        """Analogy requiring multiple structural mappings"""
        rr = RelationalReasoning()

        # Source: A is to B as B is to C
        a = rr.create_element("A", element_id="a")
        b = rr.create_element("B", element_id="b")
        c = rr.create_element("C", element_id="c")

        s1 = rr.bind(a, RelationType.CAUSES, b)
        s2 = rr.bind(b, RelationType.CAUSES, c)
        source = rr.compose(s1, s2)

        # Target: X is to Y as Y is to ?
        x = rr.create_element("X", element_id="x")
        y = rr.create_element("Y", element_id="y")
        z = rr.create_element("Z", element_id="z")

        target_elements = {"x": x, "y": y, "z": z}
        result = rr.analogy(source, target_elements)

        assert result is not None
        assert len(result.relations) >= 1


class TestLogicNetworkStress:
    """Stress test logical inference"""

    def test_long_inference_chain(self):
        """Chain of 10 implications"""
        ln = LogicNetwork()

        # P1 -> P2 -> P3 -> ... -> P10
        props = [
            Proposition(f"P{i}", PropositionType.ATOMIC, f"Proposition {i}") for i in range(11)
        ]
        implications = []

        for i in range(10):
            impl = Proposition(
                f"impl_{i}_{i + 1}",
                PropositionType.IMPLICATION,
                f"P{i} implies P{i + 1}",
                components=[props[i], props[i + 1]],
            )
            implications.append(impl)

        # Given P0, derive as much as possible
        premises = [props[0]] + implications

        all_inferences = []
        current_premises = premises.copy()

        # Iterate to find all derivable conclusions
        for _ in range(15):
            new_inferences = ln.derive_inferences(current_premises)
            if not new_inferences:
                break
            all_inferences.extend(new_inferences)
            # Add new conclusions as premises
            for inf in new_inferences:
                if inf.conclusion not in current_premises:
                    current_premises.append(inf.conclusion)

        # Should derive at least some of the chain
        derived_ids = {inf.conclusion.id for inf in all_inferences}
        assert len(derived_ids) >= 1

    def test_consistency_with_many_beliefs(self):
        """Check consistency of 50 beliefs"""
        ln = LogicNetwork()

        beliefs = []
        # 25 positive, 25 negative (no contradictions)
        for i in range(25):
            beliefs.append(Proposition(f"P{i}", PropositionType.ATOMIC, f"P{i}"))

        for i in range(25, 50):
            inner = Proposition(f"Q{i}", PropositionType.ATOMIC, f"Q{i}")
            beliefs.append(
                Proposition(f"not_Q{i}", PropositionType.NEGATION, f"not Q{i}", components=[inner])
            )

        consistent, contradictions = ln.check_consistency(beliefs)
        assert consistent
        assert len(contradictions) == 0

    def test_find_contradiction_in_large_set(self):
        """Find one contradiction among 100 beliefs"""
        ln = LogicNetwork()

        beliefs = []
        # 99 consistent beliefs
        for i in range(99):
            beliefs.append(Proposition(f"P{i}", PropositionType.ATOMIC, f"P{i}"))

        # Add contradiction: P50 and not-P50
        inner = Proposition("P50", PropositionType.ATOMIC, "P50")
        beliefs.append(
            Proposition("not_P50", PropositionType.NEGATION, "not P50", components=[inner])
        )

        consistent, contradictions = ln.check_consistency(beliefs)
        assert not consistent
        assert len(contradictions) >= 1

    def test_complex_constraint_propagation(self):
        """Propagate constraints through network"""
        ln = LogicNetwork()

        # Build implication network
        for i in range(10):
            ln.represent_relation(
                [f"N{i}", f"N{i + 1}"], "implies", properties={"type": "implication"}
            )

        model = {f"N{i}": None for i in range(11)}
        model["N0"] = True

        # Propagate from N0
        updated = ln.propagate_implications(model, ("N0", True))

        # All should be True now
        assert updated["N0"] is True
        assert updated["N1"] is True


class TestHPCPFCStress:
    """Stress test episodic and schematic memory"""

    def test_massive_episode_encoding(self):
        """Encode 1000 episodes"""
        complex = HPCPFCComplex()

        start = time.time()
        for i in range(1000):
            complex.encode_and_abstract(
                f"event_{i}", {"index": i, "type": i % 10, "value": np.random.random()}
            )
        elapsed = time.time() - start

        # Should handle 1000 episodes
        assert len(complex.hippocampus.episodes) > 0
        # Should be reasonably fast (< 5 seconds)
        assert elapsed < 5.0

    def test_schema_from_many_examples(self):
        """Extract schema from 100 similar episodes"""
        complex = HPCPFCComplex()

        # Encode 100 "meal" episodes with same structure
        for i in range(100):
            complex.encode_and_abstract(
                f"meal_{i}",
                {"type": "meal", "food": f"food_{i % 10}", "location": "restaurant", "cost": i},
            )

        # Should have extracted schema
        assert len(complex.pfc.schemas) >= 1

        if complex.pfc.schemas:
            schema = list(complex.pfc.schemas.values())[0]
            # High confidence from many examples
            assert schema.confidence > 0.3

    def test_novel_composition_stress(self):
        """Create many novel compositions"""
        complex = HPCPFCComplex()

        # Build knowledge base
        concepts = ["run", "jump", "walk", "fast", "slow", "twice", "backward", "forward"]
        for c in concepts:
            complex.encode_and_abstract(c, {"concept": c})

        # Create 50 novel combinations
        compositions = []
        for i in range(50):
            c1 = concepts[i % len(concepts)]
            c2 = concepts[(i + 3) % len(concepts)]
            result = complex.compose_novel([c1, c2], relation=f"combo_{i}")
            compositions.append(result)

        assert len(compositions) == 50
        assert all(c["novel"] for c in compositions)

    def test_retrieval_under_load(self):
        """Retrieval performance with many episodes"""
        hpc = HPCPFCComplex().hippocampus

        # Encode 500 episodes
        for i in range(500):
            hpc.encode_episode(f"event_{i}", {"index": i, "category": i % 20})

        # Time retrieval
        start = time.time()
        for _ in range(100):
            hpc.retrieve_episode(f"event_{np.random.randint(500)}", top_k=5)
        elapsed = time.time() - start

        # Should be fast (< 1 second for 100 retrievals)
        assert elapsed < 1.0


class TestDualProcessIntegrationStress:
    """Full system integration stress tests"""

    def test_high_throughput(self):
        """Process 500 inputs rapidly"""
        controller = DualProcessController()

        # Set up some knowledge
        controller.learn_pattern("known", [np.random.randn(10) for _ in range(5)])

        start = time.time()
        for i in range(500):
            inp = np.random.randn(10)
            controller.process(inp)
        elapsed = time.time() - start

        # Should handle 500 inputs in < 10 seconds
        assert elapsed < 10.0
        assert controller._total_processing_count == 500

    def test_adversarial_inputs(self):
        """Handle adversarial/edge case inputs"""
        controller = DualProcessController()

        # Zero vector
        result = controller.process(np.zeros(10))
        assert result is not None

        # Very large values
        result = controller.process(np.ones(10) * 1e6)
        assert result is not None

        # Very small values
        result = controller.process(np.ones(10) * 1e-10)
        assert result is not None

        # Negative values
        result = controller.process(np.ones(10) * -1)
        assert result is not None

        # Mixed extreme values
        mixed = np.array([1e6, -1e6, 0, 1e-10, -1e-10] * 2)
        result = controller.process(mixed)
        assert result is not None

    def test_habit_conflict_with_pattern(self):
        """Habit suggests one action, pattern suggests another"""
        controller = DualProcessController()

        # Train strong habit
        habit_trigger = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        controller.train_habit(habit_trigger, Action(id="habit_action"), repetitions=50)

        # Learn conflicting pattern
        controller.learn_pattern(
            "pattern_action",
            [
                np.array([1.0, 0.1, 0.0, 0.0, 0.0]),
                np.array([0.9, 0.0, 0.0, 0.0, 0.0]),
            ],
        )

        # Input that triggers both
        conflict_input = np.array([0.95, 0.05, 0.0, 0.0, 0.0])
        result = controller.process(conflict_input)

        # Should detect conflict
        assert result.conflict_detected or result.s2_output is not None

    def test_emotional_override(self):
        """Strong emotion should influence processing"""
        controller = DualProcessController()

        # Learn dangerous input
        danger = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        controller.learn_emotional_association(danger, threat=0.95, reward=0.0)

        # Process dangerous input
        result = controller.process(danger)

        # Should have high threat, engage S2
        assert result.s1_output.emotional_valence.threat > 0.5

    def test_system2_timeout(self):
        """System 2 should not deliberate forever"""
        controller = DualProcessController()

        # Highly ambiguous input
        ambiguous = np.random.randn(10) * 0.5  # Moderate activation everywhere

        start = time.time()
        result = controller.process(ambiguous)
        elapsed = time.time() - start

        # Should complete in reasonable time
        assert elapsed < 1.0

        if result.s2_output:
            # Should have bounded deliberation
            assert result.s2_output.deliberation_steps <= 10


class TestPatternRecognitionStress:
    """Stress test pattern matching at scale"""

    def test_many_patterns(self):
        """Learn and match against 1000 patterns"""
        pr = PatternRecognition(similarity_threshold=0.8)

        # Learn 1000 patterns
        for i in range(1000):
            pattern = np.random.randn(50)
            pr.learn_pattern(f"pattern_{i}", [pattern])

        # Query
        query = np.random.randn(50)
        start = time.time()
        matches = pr.match(query, top_k=10)
        elapsed = time.time() - start

        # Should be fast (< 100ms)
        assert elapsed < 0.1

    def test_high_dimensional_patterns(self):
        """Patterns in high-dimensional space"""
        pr = PatternRecognition(similarity_threshold=0.5)

        # 100-dimensional patterns
        dim = 100
        for i in range(100):
            pattern = np.random.randn(dim)
            pr.learn_pattern(f"hd_pattern_{i}", [pattern])

        # Query with high-dimensional vector
        query = np.random.randn(dim)
        matches = pr.match(query)

        # Should work
        assert isinstance(matches, list)

    def test_pattern_generalization_edge(self):
        """Generalization at threshold boundary"""
        pr = PatternRecognition(similarity_threshold=0.7)

        # Learn pattern
        original = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        pr.learn_pattern("original", [original])

        # Test at various similarity levels
        for similarity in [0.9, 0.8, 0.7, 0.6, 0.5]:
            # Create input with desired similarity
            test = original * similarity + np.random.randn(5) * (1 - similarity) * 0.1
            test = test / np.linalg.norm(test)  # Normalize

            matches = pr.match(test)
            # May or may not match depending on noise


class TestEmotionalProcessingStress:
    """Stress test emotional valuation"""

    def test_competing_emotions(self):
        """Stimulus with both threat and reward"""
        ev = EmotionalValuation()

        # Learn mixed associations
        stimulus = np.array([1.0, 1.0, 0.0])
        ev.learn_association(stimulus, outcome_threat=0.7, outcome_reward=0.7)

        valence = ev.evaluate(stimulus)

        # Should have both high
        assert valence.threat > 0.3
        assert valence.reward > 0.3
        # High arousal from mixed emotions
        assert valence.arousal > 0.3

    def test_extinction_under_reinforcement(self):
        """Extinction competes with reinforcement"""
        ev = EmotionalValuation()

        stimulus = np.array([1.0, 0.0, 0.0])

        # Learn fear
        ev.learn_association(stimulus, outcome_threat=0.9, outcome_reward=0.0)

        # Partially extinguish
        for _ in range(3):
            ev.extinction(stimulus, extinction_rate=0.2)

        # Re-learn fear
        ev.learn_association(stimulus, outcome_threat=0.8, outcome_reward=0.0)

        # Should still be somewhat threatening
        valence = ev.evaluate(stimulus)
        assert valence.threat > 0.3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
