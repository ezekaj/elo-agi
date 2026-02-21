"""Tests for Theory of Mind"""

import numpy as np
import pytest
from neuro.modules.m15_social.theory_of_mind import (
    BeliefTracker, MentalStateAttribution, TheoryOfMind, ToMParams
)

class TestBeliefTracker:
    """Tests for belief tracking"""

    def test_initialization(self):
        """Test tracker initialization"""
        tracker = BeliefTracker()
        assert len(tracker.own_beliefs) == 0
        assert len(tracker.others_beliefs) == 0

    def test_update_own_belief(self):
        """Test updating own beliefs"""
        tracker = BeliefTracker()
        belief = np.random.rand(50)
        tracker.update_own_belief("location", belief)
        assert "location" in tracker.own_beliefs

    def test_attribute_belief(self):
        """Test attributing beliefs to others"""
        tracker = BeliefTracker()
        belief = np.random.rand(50)
        tracker.attribute_belief("sally", "location", belief)
        assert "sally" in tracker.others_beliefs
        assert "location" in tracker.others_beliefs["sally"]

    def test_second_order_belief(self):
        """Test second-order beliefs"""
        tracker = BeliefTracker()
        belief = np.random.rand(50)
        tracker.attribute_second_order("sally", "anne", "location", belief)
        assert "sally" in tracker.second_order
        assert "anne" in tracker.second_order["sally"]

    def test_false_belief_detection(self):
        """Test false belief detection"""
        tracker = BeliefTracker()
        reality = np.array([1.0] * 50)
        false_belief = np.array([-1.0] * 50)
        tracker.update_own_belief("location", reality)
        tracker.attribute_belief("sally", "location", false_belief)
        assert tracker.check_false_belief("sally", "location")

class TestMentalStateAttribution:
    """Tests for mental state attribution"""

    def test_initialization(self):
        """Test attribution initialization"""
        attr = MentalStateAttribution()
        assert len(attr.attributed_states) == 0

    def test_observe_behavior(self):
        """Test inferring mental states from behavior"""
        attr = MentalStateAttribution()
        behavior = np.random.rand(50)
        result = attr.observe_behavior("agent1", behavior)
        assert "belief" in result
        assert "desire" in result
        assert "intention" in result

    def test_get_attributed_state(self):
        """Test getting attributed states"""
        attr = MentalStateAttribution()
        attr.observe_behavior("agent1", np.random.rand(50))
        belief = attr.get_attributed_state("agent1", "belief")
        assert belief is not None

    def test_predict_behavior(self):
        """Test predicting behavior from mental states"""
        attr = MentalStateAttribution()
        attr.observe_behavior("agent1", np.random.rand(50))
        prediction = attr.predict_behavior("agent1")
        assert prediction is not None

class TestTheoryOfMind:
    """Tests for integrated ToM"""

    def test_initialization(self):
        """Test ToM initialization"""
        tom = TheoryOfMind()
        assert tom.belief_tracker is not None
        assert tom.attribution is not None

    def test_process_self(self):
        """Test self processing (mPFC)"""
        tom = TheoryOfMind()
        state = np.random.rand(50)
        result = tom.process_self(state)
        assert "mpfc_activity" in result

    def test_process_other(self):
        """Test other processing (TPJ)"""
        tom = TheoryOfMind()
        behavior = np.random.rand(50)
        result = tom.process_other("sally", behavior)
        assert "tpj_activity" in result
        assert "belief" in result

    def test_false_belief_task(self):
        """Test Sally-Anne style task"""
        tom = TheoryOfMind()
        reality = np.array([1.0] * 50)
        sally_belief = np.array([-1.0] * 50)
        result = tom.run_false_belief_task("sally", reality, sally_belief)
        assert result["false_belief_detected"]

    def test_update(self):
        """Test system update"""
        tom = TheoryOfMind()
        tom.process_other("sally", np.random.rand(50))
        initial = np.mean(tom.tpj_activation)
        tom.update(dt=10.0)
        assert np.mean(tom.tpj_activation) < initial

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
