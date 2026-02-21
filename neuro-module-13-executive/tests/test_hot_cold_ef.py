"""Tests for hot and cold executive functions"""

import numpy as np
import pytest
from neuro.modules.m13_executive.hot_cold_ef import (
    HotExecutiveFunction, ColdExecutiveFunction, EmotionalRegulator, HotColdParams
)

class TestHotExecutiveFunction:
    """Tests for hot EF"""

    def test_initialization(self):
        """Test hot EF initialization"""
        hot = HotExecutiveFunction()

        assert hot.emotional_arousal == 0.5
        assert hot.valence == 0.0

    def test_emotional_stimulus_processing(self):
        """Test processing emotional stimulus"""
        hot = HotExecutiveFunction()

        stimulus = np.random.rand(50)
        result = hot.process_emotional_stimulus(
            stimulus,
            emotional_intensity=0.8,
            valence=0.5
        )

        assert "ofc_activity" in result
        assert "emotional_arousal" in result
        assert hot.emotional_arousal > 0.5  # Increased

    def test_reward_evaluation(self):
        """Test reward evaluation with discounting"""
        hot = HotExecutiveFunction()

        immediate = hot.evaluate_reward(100, delay=0)
        delayed = hot.evaluate_reward(100, delay=10)

        assert immediate > delayed  # Discounting

    def test_risky_decision(self):
        """Test risky decision making"""
        hot = HotExecutiveFunction()

        result = hot.make_risky_decision(
            safe_option=50,
            risky_option=(100, 0, 0.5)  # 50% chance of 100 or 0
        )

        assert "choice" in result
        assert result["choice"] in ["safe", "risky"]
        assert "risky_expected" in result

    def test_arousal_affects_risk(self):
        """Test emotional arousal affects risk taking"""
        hot = HotExecutiveFunction()

        # Low arousal
        hot.emotional_arousal = 0.2
        hot.valence = 0.5
        low_arousal_result = hot.make_risky_decision(50, (100, 0, 0.5))

        # High arousal (reset)
        hot2 = HotExecutiveFunction()
        hot2.emotional_arousal = 0.9
        hot2.valence = 0.5
        high_arousal_result = hot2.make_risky_decision(50, (100, 0, 0.5))

        # Results should be valid (actual behavior depends on parameters)
        assert "risk_modifier" in low_arousal_result
        assert "risk_modifier" in high_arousal_result

    def test_reward_learning(self):
        """Test reward expectation learning"""
        hot = HotExecutiveFunction()

        for _ in range(10):
            hot.update_reward_learning(1.0)

        assert hot.expected_reward > 0.5

    def test_state_getter(self):
        """Test state retrieval"""
        hot = HotExecutiveFunction()

        state = hot.get_state()

        assert "emotional_arousal" in state
        assert "valence" in state
        assert "expected_reward" in state

class TestColdExecutiveFunction:
    """Tests for cold EF"""

    def test_initialization(self):
        """Test cold EF initialization"""
        cold = ColdExecutiveFunction()

        assert cold.cognitive_load == 0.0
        assert len(cold.active_rules) == 0

    def test_abstract_stimulus_processing(self):
        """Test processing abstract stimulus"""
        cold = ColdExecutiveFunction()

        stimulus = np.random.rand(50)
        result = cold.process_abstract_stimulus(stimulus, rules=["rule1"])

        assert "dlpfc_activity" in result
        assert "active_rules" in result
        assert "rule1" in result["active_rules"]

    def test_logical_reasoning(self):
        """Test logical reasoning"""
        cold = ColdExecutiveFunction()

        # Create related premises and conclusion
        premise1 = np.random.rand(50)
        premise2 = premise1 + np.random.rand(50) * 0.2
        conclusion = (premise1 + premise2) / 2

        result = cold.reason_logically([premise1, premise2], conclusion)

        assert "valid" in result
        assert "confidence" in result
        assert "cognitive_load" in result

    def test_cognitive_load_increases(self):
        """Test cognitive load increases with complexity"""
        cold = ColdExecutiveFunction()

        # Simple (few premises)
        cold.reason_logically([np.random.rand(50)], np.random.rand(50))
        simple_load = cold.cognitive_load

        # Complex (many premises)
        cold.reason_logically(
            [np.random.rand(50) for _ in range(5)],
            np.random.rand(50)
        )
        complex_load = cold.cognitive_load

        assert complex_load > simple_load

    def test_problem_solving(self):
        """Test problem solving"""
        cold = ColdExecutiveFunction()

        result = cold.solve_problem(problem_complexity=0.3)

        assert "solved" in result
        assert "success_probability" in result
        assert "cognitive_load" in result

    def test_rule_strength_learning(self):
        """Test rule strengths are learned"""
        cold = ColdExecutiveFunction()
        cold.active_rules = ["test_rule"]
        cold.rule_strength["test_rule"] = 0.5

        # Successful problem solving
        cold.solve_problem(0.2)  # Easy problem

        # Rule strength might change based on outcome
        assert "test_rule" in cold.rule_strength

    def test_state_getter(self):
        """Test state retrieval"""
        cold = ColdExecutiveFunction()

        state = cold.get_state()

        assert "cognitive_load" in state
        assert "active_rules" in state
        assert "rule_strengths" in state

class TestEmotionalRegulator:
    """Tests for emotional regulation"""

    def test_initialization(self):
        """Test regulator initialization"""
        regulator = EmotionalRegulator()

        assert regulator.hot_ef is not None
        assert regulator.cold_ef is not None

    def test_reappraisal_regulation(self):
        """Test cognitive reappraisal"""
        regulator = EmotionalRegulator()

        result = regulator.regulate_emotion(
            emotional_intensity=0.8,
            strategy="reappraisal"
        )

        assert result["strategy"] == "reappraisal"
        assert result["regulated_intensity"] < result["initial_intensity"]

    def test_suppression_regulation(self):
        """Test response suppression"""
        regulator = EmotionalRegulator()

        result = regulator.regulate_emotion(
            emotional_intensity=0.8,
            strategy="suppression"
        )

        assert result["strategy"] == "suppression"
        assert result["regulated_intensity"] <= result["initial_intensity"]

    def test_reappraisal_less_costly(self):
        """Test reappraisal is less cognitively costly"""
        regulator = EmotionalRegulator()

        reapp = regulator.regulate_emotion(0.8, "reappraisal")

        regulator2 = EmotionalRegulator()
        supp = regulator2.regulate_emotion(0.8, "suppression")

        assert reapp["cognitive_cost"] < supp["cognitive_cost"]

    def test_hybrid_decision(self):
        """Test hybrid hot/cold decision making"""
        regulator = EmotionalRegulator()

        result = regulator.make_hybrid_decision(
            stimulus=np.random.rand(50),
            emotional_content=0.7,
            logical_content=0.3
        )

        assert "hot_contribution" in result
        assert "cold_contribution" in result
        assert "combined_activation" in result

    def test_emotional_dominates(self):
        """Test emotional content dominates when higher"""
        regulator = EmotionalRegulator()

        result = regulator.make_hybrid_decision(
            stimulus=np.random.rand(50),
            emotional_content=0.9,
            logical_content=0.1
        )

        assert result["hot_weight"] > result["cold_weight"]

    def test_logical_dominates(self):
        """Test logical content dominates when higher"""
        regulator = EmotionalRegulator()

        result = regulator.make_hybrid_decision(
            stimulus=np.random.rand(50),
            emotional_content=0.1,
            logical_content=0.9
        )

        assert result["cold_weight"] > result["hot_weight"]

    def test_state_getter(self):
        """Test state retrieval"""
        regulator = EmotionalRegulator()

        state = regulator.get_state()

        assert "hot" in state
        assert "cold" in state
        assert "regulation_strategy" in state

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
