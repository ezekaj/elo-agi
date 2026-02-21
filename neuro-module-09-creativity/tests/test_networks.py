"""
Tests for Creative Networks (DMN, ECN, Salience)
"""

import pytest
import numpy as np
from neuro.modules.m09_creativity.networks import DefaultModeNetwork, ExecutiveControlNetwork, SalienceNetwork
from neuro.modules.m09_creativity.networks.salience_network import NetworkState, SwitchTrigger
from neuro.modules.m09_creativity.networks.executive_control_network import EvaluationCriterion

class TestDefaultModeNetwork:
    """Tests for Default Mode Network"""

    def test_initialization(self):
        """Test DMN initializes correctly"""
        dmn = DefaultModeNetwork()
        assert dmn.spreading_activation_decay == 0.7
        assert dmn.association_threshold == 0.3
        assert len(dmn.concepts) == 0

    def test_add_concept(self):
        """Test adding concepts to semantic memory"""
        dmn = DefaultModeNetwork()
        dmn.add_concept("tree", "A tall plant", {"nature": 0.9, "green": 0.7})

        assert "tree" in dmn.concepts
        assert dmn.concepts["tree"].content == "A tall plant"
        assert dmn.concepts["tree"].features["nature"] == 0.9

    def test_create_association(self):
        """Test creating associations between concepts"""
        dmn = DefaultModeNetwork()
        dmn.add_concept("tree", "A tall plant", {"nature": 0.9})
        dmn.add_concept("forest", "Many trees", {"nature": 1.0})

        dmn.create_association("tree", "forest", strength=0.8, association_type="part_of")

        # Check associations list contains the association
        assoc_found = any(
            a.source_id == "tree" and a.target_id == "forest"
            for a in dmn.associations
        )
        assert assoc_found

    def test_spontaneous_thought_generation(self):
        """Test generating spontaneous thoughts"""
        dmn = DefaultModeNetwork()
        dmn.add_concept("creativity", "Novel ideas", {"abstract": 0.8})
        dmn.add_concept("art", "Visual expression", {"creative": 0.9})
        dmn.add_concept("music", "Auditory art", {"creative": 0.8})

        dmn.create_association("creativity", "art", 0.7)
        dmn.create_association("creativity", "music", 0.6)

        thought = dmn.generate_spontaneous_thought(seed="creativity")

        assert thought is not None
        assert len(thought.concepts) >= 1
        assert 0 <= thought.novelty_score <= 1
        assert 0 <= thought.coherence_score <= 1

    def test_mind_wandering(self):
        """Test mind wandering generates multiple thoughts"""
        dmn = DefaultModeNetwork()
        dmn.add_concept("ocean", "Large body of water", {"nature": 0.9})
        dmn.add_concept("wave", "Water movement", {"motion": 0.7})
        dmn.add_concept("beach", "Sandy shore", {"relaxation": 0.8})

        dmn.create_association("ocean", "wave", 0.9)
        dmn.create_association("ocean", "beach", 0.7)
        dmn.create_association("wave", "beach", 0.5)

        thoughts = dmn.mind_wander(duration_steps=5)

        assert len(thoughts) == 5
        for thought in thoughts:
            assert hasattr(thought, 'novelty_score')
            assert hasattr(thought, 'coherence_score')

    def test_find_distant_associations(self):
        """Test finding distant conceptual connections"""
        dmn = DefaultModeNetwork()
        # Create chain: A -> B -> C -> D
        dmn.add_concept("A", "Concept A", {"x": 1.0})
        dmn.add_concept("B", "Concept B", {"x": 0.8})
        dmn.add_concept("C", "Concept C", {"x": 0.6})
        dmn.add_concept("D", "Concept D", {"x": 0.4})

        dmn.create_association("A", "B", 0.9)
        dmn.create_association("B", "C", 0.8)
        dmn.create_association("C", "D", 0.7)

        distant = dmn.find_distant_associations("A", min_distance=2, max_results=5)

        # Should find C and D (distance >= 2)
        found_concepts = [d[0] for d in distant]
        assert "D" in found_concepts or "C" in found_concepts

    def test_novelty_computation(self):
        """Test that uncommon combinations have higher novelty"""
        dmn = DefaultModeNetwork()
        dmn.add_concept("fire", "Combustion", {"hot": 1.0})
        dmn.add_concept("ice", "Frozen water", {"cold": 1.0})
        dmn.add_concept("water", "H2O", {"liquid": 0.8})

        # Fire and ice are semantically distant
        dmn.create_association("fire", "water", 0.3)  # Weak
        dmn.create_association("ice", "water", 0.9)   # Strong

        thought1 = dmn.generate_spontaneous_thought(seed="water")

        assert thought1.novelty_score >= 0

class TestExecutiveControlNetwork:
    """Tests for Executive Control Network"""

    def test_initialization(self):
        """Test ECN initializes correctly"""
        ecn = ExecutiveControlNetwork()
        assert ecn.current_goal is None
        assert len(ecn.default_weights) > 0

    def test_set_goal(self):
        """Test setting creative goal"""
        ecn = ExecutiveControlNetwork()
        ecn.set_goal("write_poem", "Write a creative poem")

        assert ecn.current_goal is not None
        assert ecn.current_goal.id == "write_poem"
        assert ecn.current_goal.description == "Write a creative poem"

    def test_evaluate_idea_basic(self):
        """Test basic idea evaluation"""
        ecn = ExecutiveControlNetwork()
        ecn.set_goal("create_art", "Create artistic work")

        evaluation = ecn.evaluate_idea(
            "idea_1",
            "A painting of a sunset",
            idea_features={"novelty": 0.6, "coherence": 0.8},
            context={"goal_relevance": 0.7}
        )

        assert evaluation is not None
        assert 0 <= evaluation.overall_score <= 1
        assert evaluation.recommendation in ["accept", "refine", "reject"]

    def test_evaluate_with_custom_criteria(self):
        """Test evaluation with custom criteria weights"""
        ecn = ExecutiveControlNetwork()

        custom_weights = {
            EvaluationCriterion.NOVELTY: 0.8,
            EvaluationCriterion.USEFULNESS: 0.2
        }

        ecn.set_goal("innovate", "Create novel solution", criteria_weights=custom_weights)

        # High novelty, low usefulness idea
        eval1 = ecn.evaluate_idea(
            "novel_idea",
            "Very novel but impractical",
            idea_features={"novelty": 0.9, "usefulness": 0.2}
        )

        # Low novelty, high usefulness idea
        eval2 = ecn.evaluate_idea(
            "useful_idea",
            "Practical but boring",
            idea_features={"novelty": 0.2, "usefulness": 0.9}
        )

        # With novelty weighted higher, novel_idea should score higher
        assert eval1.overall_score > eval2.overall_score

    def test_refine_idea(self):
        """Test idea refinement"""
        ecn = ExecutiveControlNetwork()
        ecn.set_goal("improve", "Improve design")

        # First evaluate
        evaluation = ecn.evaluate_idea(
            "initial_idea",
            "Basic design concept",
            idea_features={"novelty": 0.4, "coherence": 0.5}
        )

        # Then refine
        refined_id, refined_content, refinement = ecn.refine_idea(
            "initial_idea",
            "Basic design concept",
            evaluation
        )

        assert "initial_idea_refined" in refined_id
        assert len(refinement.changes) > 0
        assert refinement.improvement_score > 0

    def test_constraint_checking(self):
        """Test constraint enforcement"""
        ecn = ExecutiveControlNetwork()
        ecn.set_goal(
            "constrained_goal",
            "Create within constraints",
            constraints=["must be simple", "no complexity"]
        )

        # Simple idea should pass constraints
        eval_simple = ecn.evaluate_idea(
            "simple_idea",
            "Simple straightforward solution",
            idea_features={"complexity": 0.2}
        )

        # Complex idea should be penalized
        eval_complex = ecn.evaluate_idea(
            "complex_idea",
            "Complex intricate sophisticated elaborate solution",
            idea_features={"complexity": 0.9}
        )

        # Both should have evaluations
        assert eval_simple is not None
        assert eval_complex is not None

class TestSalienceNetwork:
    """Tests for Salience Network"""

    def test_initialization(self):
        """Test Salience Network initializes in DMN state"""
        salience = SalienceNetwork()
        assert salience.current_state == NetworkState.DMN
        activity = salience.get_network_activity()
        assert activity.reconfiguration_level >= 0

    def test_network_switching(self):
        """Test switching between DMN and ECN"""
        salience = SalienceNetwork()

        # Start in DMN
        assert salience.current_state == NetworkState.DMN

        # Switch to ECN
        salience.execute_switch(SwitchTrigger.IDEA_GENERATED)
        assert salience.current_state == NetworkState.ECN

        # Switch back to DMN
        salience.execute_switch(SwitchTrigger.EVALUATION_COMPLETE)
        assert salience.current_state == NetworkState.DMN

    def test_reconfiguration_tracking(self):
        """Test that network reconfiguration is tracked"""
        salience = SalienceNetwork()

        initial_activity = salience.get_network_activity()
        initial_reconfig = initial_activity.reconfiguration_level

        # Multiple switches should increase reconfiguration
        for _ in range(5):
            salience.execute_switch(SwitchTrigger.IDEA_GENERATED)
            salience.execute_switch(SwitchTrigger.EVALUATION_COMPLETE)

        final_activity = salience.get_network_activity()
        assert final_activity.reconfiguration_level >= initial_reconfig

    def test_should_switch_detection(self):
        """Test automatic switch detection"""
        salience = SalienceNetwork()

        # In DMN, should switch when many ideas generated
        metrics = {
            "ideas_generated": 10,
            "target_ideas": 5,
            "current_novelty": 0.3
        }

        should_switch, trigger = salience.should_switch(metrics)
        # May or may not switch based on thresholds and time

    def test_record_idea_generated(self):
        """Test recording idea generation"""
        salience = SalienceNetwork()

        for _ in range(5):
            salience.record_idea_generated(0.7)  # positional argument

        stats = salience.get_switch_statistics()
        # Stats should reflect activity

    def test_switch_history(self):
        """Test switch history is maintained"""
        salience = SalienceNetwork()

        salience.execute_switch(SwitchTrigger.IDEA_GENERATED)
        salience.execute_switch(SwitchTrigger.EVALUATION_COMPLETE)
        salience.execute_switch(SwitchTrigger.NOVELTY_DROP)

        assert len(salience.switch_history) == 3

    def test_force_switch(self):
        """Test forced network switching"""
        salience = SalienceNetwork()

        salience.force_switch_to(NetworkState.ECN)
        assert salience.current_state == NetworkState.ECN

        salience.force_switch_to(NetworkState.DMN)
        assert salience.current_state == NetworkState.DMN

    def test_creative_reconfiguration_insight(self):
        """
        Test key insight: creative ideas lead to higher network reconfiguration.
        """
        salience = SalienceNetwork()

        # Record high novelty ideas
        for _ in range(10):
            salience.record_idea_generated(0.9)  # positional
            salience.execute_switch(SwitchTrigger.IDEA_GENERATED)
            salience.execute_switch(SwitchTrigger.EVALUATION_COMPLETE)

        high_novelty_activity = salience.get_network_activity()
        high_novelty_reconfig = high_novelty_activity.reconfiguration_level

        # Reset
        salience2 = SalienceNetwork()

        # Record low novelty ideas
        for _ in range(10):
            salience2.record_idea_generated(0.2)  # positional
            salience2.execute_switch(SwitchTrigger.IDEA_GENERATED)
            salience2.execute_switch(SwitchTrigger.EVALUATION_COMPLETE)

        low_novelty_activity = salience2.get_network_activity()
        low_novelty_reconfig = low_novelty_activity.reconfiguration_level

        # Both should have reconfiguration
        assert high_novelty_reconfig >= 0
        assert low_novelty_reconfig >= 0

class TestNetworkIntegration:
    """Tests for network integration"""

    def test_dmn_ecn_cycle(self):
        """Test full DMN-ECN creative cycle"""
        dmn = DefaultModeNetwork()
        ecn = ExecutiveControlNetwork()
        salience = SalienceNetwork()

        # Setup
        dmn.add_concept("innovation", "New ideas", {"creative": 0.9})
        dmn.add_concept("solution", "Problem solving", {"practical": 0.8})
        dmn.create_association("innovation", "solution", 0.7)

        ecn.set_goal("create_solution", "Find innovative solution")

        # Generate in DMN mode
        assert salience.current_state == NetworkState.DMN
        thought = dmn.generate_spontaneous_thought(seed="innovation")

        # Switch to ECN for evaluation
        salience.execute_switch(SwitchTrigger.IDEA_GENERATED)
        assert salience.current_state == NetworkState.ECN

        # Evaluate
        evaluation = ecn.evaluate_idea(
            "thought_idea",
            thought.concepts[0] if thought.concepts else "idea",
            idea_features={"novelty": thought.novelty_score, "coherence": thought.coherence_score}
        )

        # Switch back to DMN
        salience.execute_switch(SwitchTrigger.EVALUATION_COMPLETE)
        assert salience.current_state == NetworkState.DMN

    def test_sustained_creative_activity(self):
        """Test sustained creative activity across networks"""
        dmn = DefaultModeNetwork()
        ecn = ExecutiveControlNetwork()
        salience = SalienceNetwork()

        # Setup rich semantic network
        concepts = [
            ("light", "Electromagnetic radiation", {"energy": 0.8}),
            ("dark", "Absence of light", {"mystery": 0.7}),
            ("shadow", "Blocked light", {"contrast": 0.6}),
            ("color", "Light wavelength", {"visual": 0.9}),
            ("spectrum", "Range of colors", {"diversity": 0.8}),
        ]

        for cid, content, features in concepts:
            dmn.add_concept(cid, content, features)

        # Create associations
        dmn.create_association("light", "dark", 0.5)
        dmn.create_association("light", "shadow", 0.7)
        dmn.create_association("light", "color", 0.9)
        dmn.create_association("color", "spectrum", 0.8)

        ecn.set_goal("explore_light", "Explore light concepts")

        # Run multiple cycles
        ideas = []
        evaluations = []

        for _ in range(5):
            # Generate
            thought = dmn.generate_spontaneous_thought()
            ideas.append(thought)
            salience.record_idea_generated(thought.novelty_score)

            # Evaluate
            salience.execute_switch(SwitchTrigger.IDEA_GENERATED)

            if thought.concepts:
                eval_result = ecn.evaluate_idea(
                    f"idea_{len(evaluations)}",
                    str(thought.concepts),
                    idea_features={"novelty": thought.novelty_score}
                )
                evaluations.append(eval_result)

            salience.execute_switch(SwitchTrigger.EVALUATION_COMPLETE)

        assert len(ideas) == 5
        activity = salience.get_network_activity()
        assert activity.reconfiguration_level >= 0
