"""Tests for System 1 components"""

import pytest
import numpy as np
from neuro.modules.m02_dual_process.system1.pattern_recognition import PatternRecognition, Pattern, PatternMatch
from neuro.modules.m02_dual_process.system1.habit_executor import HabitExecutor, Action, HabitStrength
from neuro.modules.m02_dual_process.system1.emotional_valuation import EmotionalValuation, ValenceType

class TestPatternRecognition:
    """Tests for parallel pattern matching"""

    def test_learn_and_match_pattern(self):
        pr = PatternRecognition(similarity_threshold=0.7)

        # Learn a pattern from examples
        examples = [
            np.array([1.0, 0.0, 1.0, 0.0]),
            np.array([0.9, 0.1, 0.9, 0.1]),
            np.array([1.1, -0.1, 1.1, -0.1]),
        ]
        pr.learn_pattern("pattern_A", examples)

        # Test match
        test_input = np.array([1.0, 0.0, 1.0, 0.0])
        matches = pr.match(test_input)

        assert len(matches) > 0
        assert matches[0].pattern_id == "pattern_A"
        assert matches[0].confidence > 0.9

    def test_parallel_matching(self):
        """Verify multiple patterns are checked simultaneously"""
        pr = PatternRecognition(similarity_threshold=0.5)

        # Learn multiple patterns
        pr.learn_pattern("A", [np.array([1, 0, 0, 0])])
        pr.learn_pattern("B", [np.array([0, 1, 0, 0])])
        pr.learn_pattern("C", [np.array([0, 0, 1, 0])])

        # Input that partially matches A and B
        test_input = np.array([0.7, 0.7, 0, 0])
        matches = pr.match(test_input)

        # Should match multiple patterns
        assert len(matches) >= 1

    def test_no_match_below_threshold(self):
        pr = PatternRecognition(similarity_threshold=0.9)
        pr.learn_pattern("strict", [np.array([1, 1, 1, 1])])

        # Very different input
        test_input = np.array([0, 0, 0, 1])
        matches = pr.match(test_input)

        assert len(matches) == 0

    def test_generalization(self):
        """Test generalization to novel input"""
        pr = PatternRecognition(similarity_threshold=0.7)
        pr.learn_pattern("learned", [np.array([1, 1, 0, 0])])

        # Novel but similar
        novel = np.array([0.8, 0.9, 0.1, 0.1])
        matches = pr.generalize(novel)

        assert len(matches) > 0

class TestHabitExecutor:
    """Tests for automatic habit execution"""

    def test_habit_formation(self):
        he = HabitExecutor(trigger_threshold=0.5)

        stimulus = np.array([1.0, 0.5, 0.0])
        action = Action(id="response_A")

        # Create weak habit
        habit = he.add_habit(stimulus, action, initial_strength=0.3)
        assert habit.strength == 0.3

        # Strengthen through repetition
        for _ in range(10):
            he.strengthen(stimulus, action)

        # Should be stronger now
        assert habit.strength > 0.5

    def test_automatic_execution(self):
        he = HabitExecutor(trigger_threshold=0.5)

        stimulus = np.array([1.0, 0.0, 0.0])
        action = Action(id="auto_response")

        # Create strong habit
        he.add_habit(stimulus, action, initial_strength=0.9)

        # Should trigger automatically
        response = he.execute(stimulus)
        assert response.triggered
        assert response.action.id == "auto_response"

    def test_extinction(self):
        he = HabitExecutor()

        stimulus = np.array([1.0, 1.0])
        action = Action(id="to_weaken")

        he.add_habit(stimulus, action, initial_strength=0.8)

        # Weaken repeatedly
        for _ in range(5):
            he.weaken(stimulus, extinction_rate=0.2)

        # Should be much weaker or gone
        response = he.execute(stimulus)
        assert not response.triggered or response.habit_strength < 0.3

    def test_context_dependency(self):
        he = HabitExecutor()

        stimulus = np.array([1.0, 1.0])
        action = Action(id="context_specific")

        # Habit only in context "work"
        he.add_habit(stimulus, action, contexts=["work"])

        # Should not trigger in different context
        response = he.execute(stimulus, context="home")
        assert not response.triggered

        # Should trigger in correct context
        response = he.execute(stimulus, context="work")
        assert response.triggered

class TestEmotionalValuation:
    """Tests for rapid threat/reward assessment"""

    def test_fast_evaluation(self):
        ev = EmotionalValuation()

        stimulus = np.array([0.5, 0.5, 0.5])
        valence = ev.evaluate(stimulus, fast_mode=True)

        # Should return quickly with some valence
        assert hasattr(valence, 'threat')
        assert hasattr(valence, 'reward')
        assert 0 <= valence.threat <= 1
        assert 0 <= valence.reward <= 1

    def test_learn_threat_association(self):
        ev = EmotionalValuation()

        dangerous = np.array([1.0, 0.0, 0.0])

        # Learn it's threatening
        ev.learn_association(dangerous, outcome_threat=0.9, outcome_reward=0.1)

        # Should now evaluate as threatening
        valence = ev.evaluate(dangerous)
        assert valence.threat > 0.5
        assert valence.valence_type == ValenceType.THREAT

    def test_learn_reward_association(self):
        ev = EmotionalValuation()

        rewarding = np.array([0.0, 1.0, 0.0])

        # Learn it's rewarding
        ev.learn_association(rewarding, outcome_threat=0.1, outcome_reward=0.9)

        # Should now evaluate as rewarding
        valence = ev.evaluate(rewarding)
        assert valence.reward > 0.5
        assert valence.valence_type == ValenceType.REWARD

    def test_generalization_from_similar(self):
        ev = EmotionalValuation(generalization_threshold=0.5)

        learned = np.array([1.0, 1.0, 0.0])
        ev.learn_association(learned, outcome_threat=0.8, outcome_reward=0.1)

        # Similar but novel stimulus
        novel = np.array([0.9, 0.9, 0.1])
        valence = ev.generalize(novel)

        # Should generalize threat
        assert valence.threat > 0.3

    def test_negativity_bias(self):
        """Threats should be weighted more heavily"""
        ev = EmotionalValuation(threat_bias=1.5)

        stimulus = np.array([0.5, 0.5])

        # Equal threat and reward learned
        ev.learn_association(stimulus, outcome_threat=0.5, outcome_reward=0.5)

        valence = ev.evaluate(stimulus)

        # With bias, threat should dominate
        assert valence.threat >= valence.reward

    def test_approach_avoid(self):
        ev = EmotionalValuation()

        threat = np.array([1.0, 0.0])
        reward = np.array([0.0, 1.0])

        ev.learn_association(threat, 0.9, 0.1)
        ev.learn_association(reward, 0.1, 0.9)

        # Threat should cause avoidance (negative)
        assert ev.get_approach_avoid(threat) < 0

        # Reward should cause approach (positive)
        assert ev.get_approach_avoid(reward) > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
