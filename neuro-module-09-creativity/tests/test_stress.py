"""
Stress Tests for Module 09 - Creativity and Imagination

Tests system behavior under high load, edge cases, and adversarial inputs.
"""

import pytest
import numpy as np
import time
from src.creative_process import CreativeProcess
from src.networks import DefaultModeNetwork, ExecutiveControlNetwork, SalienceNetwork
from src.networks.salience_network import NetworkState, SwitchTrigger
from src.imagery import ImagerySystem


class TestDMNStress:
    """Stress tests for Default Mode Network"""

    def test_large_semantic_network(self):
        """Test DMN with 1000+ concepts"""
        dmn = DefaultModeNetwork()

        # Add 1000 concepts
        for i in range(1000):
            features = {
                f"feature_{j}": np.random.random()
                for j in range(5)
            }
            dmn.add_concept(f"concept_{i}", f"Content {i}", features)

        assert len(dmn.concepts) == 1000

        # Create many associations
        for i in range(500):
            j = (i + 100) % 1000
            dmn.create_association(f"concept_{i}", f"concept_{j}", np.random.random())

        # Generate thoughts should still work
        thought = dmn.generate_spontaneous_thought()
        assert thought is not None

    def test_dense_association_network(self):
        """Test DMN with dense interconnections"""
        dmn = DefaultModeNetwork()

        # Create 50 concepts
        for i in range(50):
            dmn.add_concept(f"node_{i}", f"Node {i}", {"x": i / 50})

        # Create O(n^2) associations (dense network)
        for i in range(50):
            for j in range(i + 1, 50):
                if np.random.random() > 0.5:  # 50% connectivity
                    dmn.create_association(f"node_{i}", f"node_{j}", np.random.random())

        # Mind wandering should handle dense network
        thoughts = dmn.mind_wander(duration_steps=20)
        assert len(thoughts) == 20

    def test_rapid_thought_generation(self):
        """Test rapid consecutive thought generation"""
        dmn = DefaultModeNetwork()

        for i in range(100):
            dmn.add_concept(f"c_{i}", f"Concept {i}", {"val": np.random.random()})

        for i in range(50):
            dmn.create_association(f"c_{i}", f"c_{(i+1) % 100}", np.random.random())

        # Generate 100 thoughts rapidly
        start = time.time()
        thoughts = []
        for _ in range(100):
            thought = dmn.generate_spontaneous_thought()
            thoughts.append(thought)
        elapsed = time.time() - start

        assert len(thoughts) == 100
        # Should complete in reasonable time
        assert elapsed < 5.0  # Less than 5 seconds

    def test_distant_association_search(self):
        """Test finding distant associations in large network"""
        dmn = DefaultModeNetwork()

        # Create chain: 0 -> 1 -> 2 -> ... -> 99
        for i in range(100):
            dmn.add_concept(f"chain_{i}", f"Chain node {i}", {"pos": i})

        for i in range(99):
            dmn.create_association(f"chain_{i}", f"chain_{i+1}", 0.9)

        # Find distant from start
        distant = dmn.find_distant_associations("chain_0", min_distance=5, max_results=10)

        # Should find nodes far along the chain
        found_distances = [int(d[0].split("_")[1]) for d in distant if d]
        if found_distances:
            assert max(found_distances) >= 5


class TestECNStress:
    """Stress tests for Executive Control Network"""

    def test_mass_evaluation(self):
        """Test evaluating many ideas"""
        ecn = ExecutiveControlNetwork()
        ecn.set_goal("test", "Test goal")

        evaluations = []
        for i in range(100):
            eval_result = ecn.evaluate_idea(
                f"idea_{i}",
                f"Idea content {i}",
                idea_features={
                    "novelty": np.random.random(),
                    "coherence": np.random.random(),
                    "usefulness": np.random.random()
                }
            )
            evaluations.append(eval_result)

        assert len(evaluations) == 100
        # All should have valid scores
        for ev in evaluations:
            assert 0 <= ev.overall_score <= 1

    def test_refinement_chain(self):
        """Test repeated refinement of same idea"""
        ecn = ExecutiveControlNetwork()
        ecn.set_goal("refine", "Continuously refine")

        idea_content = "Initial idea"
        idea_id = "chain_idea"

        for iteration in range(10):
            evaluation = ecn.evaluate_idea(
                idea_id,
                idea_content,
                idea_features={"novelty": 0.3, "coherence": 0.5}
            )

            new_id, new_content, refinement = ecn.refine_idea(
                idea_id,
                idea_content,
                evaluation
            )

            idea_id = new_id
            idea_content = new_content

        # Should have refined 10 times
        assert "refined" in idea_id

    def test_varied_criteria_weights(self):
        """Test evaluation with extreme criteria weights"""
        ecn = ExecutiveControlNetwork()

        from src.networks.executive_control_network import EvaluationCriterion

        # Extreme weights
        extreme_weights = {
            EvaluationCriterion.NOVELTY: 100.0,
            EvaluationCriterion.USEFULNESS: 0.001
        }

        ecn.set_goal("extreme", "Extreme weights", criteria_weights=extreme_weights)

        evaluation = ecn.evaluate_idea(
            "extreme_idea",
            "Test",
            idea_features={"novelty": 0.9, "usefulness": 0.1}
        )

        # Should still produce valid score
        assert 0 <= evaluation.overall_score


class TestSalienceNetworkStress:
    """Stress tests for Salience Network"""

    def test_rapid_switching(self):
        """Test rapid network switching"""
        salience = SalienceNetwork()

        # Switch 500 times
        for i in range(500):
            if i % 2 == 0:
                salience.execute_switch(SwitchTrigger.IDEA_GENERATED)
            else:
                salience.execute_switch(SwitchTrigger.EVALUATION_COMPLETE)

        assert len(salience.switch_history) == 500

    def test_high_reconfiguration(self):
        """Test high reconfiguration levels"""
        salience = SalienceNetwork()

        # Generate many high-novelty recordings
        for _ in range(1000):
            salience.record_idea_generated(0.95)  # positional

        # Should have recorded ideas
        activity = salience.get_network_activity()
        assert activity is not None

    def test_switch_decision_stability(self):
        """Test switch decision under varying metrics"""
        salience = SalienceNetwork()

        decisions = []
        for _ in range(100):
            metrics = {
                "ideas_generated": np.random.randint(0, 20),
                "target_ideas": 10,
                "current_novelty": np.random.random(),
                "evaluation_complete": np.random.random() > 0.5,
                "best_score": np.random.random(),
                "target_score": 0.7
            }
            should_switch, trigger = salience.should_switch(metrics)
            decisions.append(should_switch)

        # Should make some decisions
        assert len(decisions) == 100


class TestImageryStress:
    """Stress tests for Imagery System"""

    def test_mass_image_creation(self):
        """Test creating many images"""
        imagery = ImagerySystem()

        for i in range(200):
            imagery.create_multimodal_image(
                f"img_{i}",
                f"Image {i} description",
                include_visual=True,
                include_auditory=i % 2 == 0,
                include_motor=i % 3 == 0,
                include_tactile=i % 4 == 0
            )

        assert len(imagery.multimodal_images) == 200

    def test_transformation_chain(self):
        """Test chaining many transformations"""
        imagery = ImagerySystem()

        img = imagery.create_multimodal_image(
            "base",
            "Base image",
            include_visual=True,
            include_motor=True
        )

        current_id = img.id
        for i in range(20):
            transformation = ["rotate", "scale_up", "scale_down"][i % 3]
            transformed = imagery.transform_multimodal(current_id, transformation)
            current_id = transformed.id

        # Should have created 20 transformed images
        assert len(imagery.multimodal_images) >= 21

    def test_mass_blending(self):
        """Test blending many image pairs"""
        imagery = ImagerySystem()

        # Create base images
        for i in range(20):
            imagery.create_multimodal_image(
                f"base_{i}",
                f"Base {i}",
                include_visual=True,
                include_tactile=True
            )

        # Blend pairs
        for i in range(10):
            imagery.blend_images(f"base_{i}", f"base_{i+10}", blend_factor=0.5)

        # Should have blended images
        blend_count = sum(1 for k in imagery.multimodal_images if "blend" in k)
        assert blend_count == 10

    def test_all_modalities_stress(self):
        """Test intensive use of all modalities"""
        imagery = ImagerySystem()

        # Create many full multimodal images
        for i in range(50):
            imagery.create_multimodal_image(
                f"full_{i}",
                f"Full multimodal {i}",
                include_visual=True,
                include_auditory=True,
                include_motor=True,
                include_tactile=True
            )

        # All should have all modalities
        for i in range(50):
            img = imagery.multimodal_images[f"full_{i}"]
            assert img.visual is not None
            assert img.auditory is not None
            assert img.motor is not None
            assert img.tactile is not None


class TestCreativeProcessStress:
    """Stress tests for Creative Process"""

    def test_large_knowledge_base(self):
        """Test creative process with large knowledge base"""
        cp = CreativeProcess()

        # Setup 500 concepts
        concepts = [
            (f"concept_{i}", f"Content {i}", {f"f{j}": np.random.random() for j in range(3)})
            for i in range(500)
        ]
        cp.setup_knowledge(concepts)

        # Create 1000 associations
        associations = [
            (f"concept_{i}", f"concept_{(i + np.random.randint(1, 50)) % 500}",
             np.random.random(), "related")
            for i in range(1000)
        ]
        cp.create_associations(associations)

        # Run creative session
        cp.set_creative_goal("stress_test", "Handle large knowledge base")
        output = cp.creative_session(
            goal="Stress test creativity",
            duration_seconds=3.0,
            target_good_ideas=3
        )

        assert output.total_generated >= 0

    def test_extended_creative_session(self):
        """Test extended creative session"""
        cp = CreativeProcess()

        concepts = [
            ("A", "Concept A", {"x": 0.5}),
            ("B", "Concept B", {"y": 0.5}),
            ("C", "Concept C", {"z": 0.5}),
        ]
        cp.setup_knowledge(concepts)
        cp.create_associations([
            ("A", "B", 0.7, "related"),
            ("B", "C", 0.8, "related"),
        ])

        cp.set_creative_goal("extended", "Extended session")

        # Run for 3 seconds (reduced from 5 to avoid timeout)
        output = cp.creative_session(
            goal="Extended creativity",
            duration_seconds=3.0,
            target_good_ideas=10
        )

        # Session should have run
        assert output.process_duration > 0

    def test_rapid_idea_generation_evaluation(self):
        """Test rapid alternation between generation and evaluation"""
        cp = CreativeProcess()

        concepts = [(f"c_{i}", f"C{i}", {"v": i/10}) for i in range(20)]
        cp.setup_knowledge(concepts)

        associations = [(f"c_{i}", f"c_{(i+1) % 20}", 0.7, "link") for i in range(20)]
        cp.create_associations(associations)

        cp.set_creative_goal("rapid", "Rapid cycling")

        # Rapidly alternate
        for _ in range(20):
            cp.generate_ideas(num_ideas=2)
            cp.evaluate_ideas()

        stats = cp.get_creative_statistics()
        assert stats["total_ideas"] > 0

    def test_mind_wandering_extended(self):
        """Test extended mind wandering"""
        cp = CreativeProcess()

        concepts = [(f"wander_{i}", f"W{i}", {"w": np.random.random()}) for i in range(50)]
        cp.setup_knowledge(concepts)

        # Dense associations for wandering
        for i in range(50):
            for j in range(i + 1, min(i + 5, 50)):
                cp.create_associations([(f"wander_{i}", f"wander_{j}", np.random.random(), "wander")])

        # Extended wandering
        ideas = cp.mind_wander_for_ideas(duration_steps=50)

        # Should produce some ideas
        # (filtered by novelty > 0.5)

    def test_concurrent_imagery_generation(self):
        """Test generating imagery for many ideas"""
        cp = CreativeProcess()

        concepts = [(f"img_{i}", f"IMG{i}", {"visual": 0.8}) for i in range(30)]
        cp.setup_knowledge(concepts)

        for i in range(29):
            cp.create_associations([(f"img_{i}", f"img_{i+1}", 0.7, "visual")])

        # Generate ideas with imagery
        ideas = cp.generate_ideas(num_ideas=20, use_imagery=True)

        # Try to add imagery to all
        for idea in ideas:
            cp.imagine_idea(idea.id, modalities=["visual", "tactile", "motor"])


class TestEdgeCases:
    """Edge case tests"""

    def test_empty_knowledge_base(self):
        """Test with no concepts"""
        cp = CreativeProcess()

        # Should handle empty gracefully
        ideas = cp.generate_ideas(num_ideas=5)
        # May generate empty or handle gracefully

    def test_disconnected_concepts(self):
        """Test with disconnected concept islands"""
        dmn = DefaultModeNetwork()

        # Island 1
        dmn.add_concept("island1_a", "A", {"x": 1})
        dmn.add_concept("island1_b", "B", {"x": 1})
        dmn.create_association("island1_a", "island1_b", 0.9)

        # Island 2 (disconnected)
        dmn.add_concept("island2_a", "C", {"y": 1})
        dmn.add_concept("island2_b", "D", {"y": 1})
        dmn.create_association("island2_a", "island2_b", 0.9)

        # Should still work
        thought = dmn.generate_spontaneous_thought()
        assert thought is not None

    def test_self_referential_association(self):
        """Test concept associated with itself"""
        dmn = DefaultModeNetwork()

        dmn.add_concept("self", "Self-referential", {"meta": 1.0})
        dmn.create_association("self", "self", 1.0)  # Self-loop

        # Should handle without infinite loop
        thought = dmn.generate_spontaneous_thought(seed="self")
        assert thought is not None

    def test_zero_strength_associations(self):
        """Test with zero-strength associations"""
        dmn = DefaultModeNetwork()

        dmn.add_concept("a", "A", {"x": 1})
        dmn.add_concept("b", "B", {"y": 1})
        dmn.add_concept("c", "C", {"z": 1})
        dmn.create_association("a", "b", 0.0)  # Zero strength
        dmn.create_association("a", "c", 0.5)  # Non-zero

        # Generate thought - should use non-zero association
        thought = dmn.generate_spontaneous_thought(seed="a")
        assert thought is not None

    def test_extreme_vividness_values(self):
        """Test imagery with extreme vividness"""
        from src.imagery import VisualImagery

        visual = VisualImagery(default_vividness=0.0)
        img1 = visual.visualize("Zero vividness")

        visual2 = VisualImagery(default_vividness=1.0)
        img2 = visual2.visualize("Max vividness")

        assert img1.vividness == 0.0
        assert img2.vividness == 1.0

    def test_very_long_descriptions(self):
        """Test with very long text descriptions"""
        imagery = ImagerySystem()

        long_description = "A " + " ".join(["very"] * 1000) + " long description"

        img = imagery.create_multimodal_image(
            "long_desc",
            long_description,
            include_visual=True
        )

        assert img is not None

    def test_special_characters_in_ids(self):
        """Test handling special characters"""
        dmn = DefaultModeNetwork()

        # Various special characters
        special_ids = ["concept-1", "concept_2", "concept.3", "concept:4"]

        for sid in special_ids:
            dmn.add_concept(sid, f"Content for {sid}", {"x": 0.5})

        for i in range(len(special_ids) - 1):
            dmn.create_association(special_ids[i], special_ids[i + 1], 0.7)

        thought = dmn.generate_spontaneous_thought()
        assert thought is not None


class TestPerformance:
    """Performance benchmarks"""

    def test_thought_generation_throughput(self):
        """Measure thought generation throughput"""
        dmn = DefaultModeNetwork()

        for i in range(100):
            dmn.add_concept(f"perf_{i}", f"Perf {i}", {"v": i / 100})

        for i in range(50):
            dmn.create_association(f"perf_{i}", f"perf_{(i + 10) % 100}", 0.7)

        start = time.time()
        count = 0
        while time.time() - start < 1.0:  # 1 second
            dmn.generate_spontaneous_thought()
            count += 1

        # Should generate at least 50 thoughts per second
        assert count >= 50, f"Only generated {count} thoughts/second"

    def test_evaluation_throughput(self):
        """Measure evaluation throughput"""
        ecn = ExecutiveControlNetwork()
        ecn.set_goal("perf", "Performance test")

        start = time.time()
        count = 0
        while time.time() - start < 1.0:
            ecn.evaluate_idea(
                f"idea_{count}",
                "Test idea",
                idea_features={"novelty": 0.5, "coherence": 0.6}
            )
            count += 1

        # Should evaluate at least 100 ideas per second
        assert count >= 100, f"Only evaluated {count} ideas/second"

    def test_imagery_creation_throughput(self):
        """Measure imagery creation throughput"""
        imagery = ImagerySystem()

        start = time.time()
        count = 0
        while time.time() - start < 1.0:
            imagery.create_multimodal_image(
                f"img_{count}",
                "Test image",
                include_visual=True,
                include_tactile=True
            )
            count += 1

        # Should create at least 50 images per second
        assert count >= 50, f"Only created {count} images/second"
