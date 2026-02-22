"""Tests for social cognition network"""

import numpy as np
import pytest
from neuro.modules.m15_social.social_network import SocialBrain, SocialCognitionNetwork


class TestSocialBrain:
    """Tests for social brain model"""

    def test_initialization(self):
        """Test brain initialization"""
        brain = SocialBrain(n_features=50)
        assert len(brain.cognitive_activation) == 50
        assert len(brain.affective_activation) == 50

    def test_cognitive_processing(self):
        """Test cognitive level processing"""
        brain = SocialBrain()
        result = brain.process_cognitive(np.random.rand(50))
        assert len(result) == 50

    def test_affective_processing(self):
        """Test affective level processing"""
        brain = SocialBrain()
        result = brain.process_affective(np.random.rand(50))
        assert len(result) == 50

    def test_integration(self):
        """Test cognitive-affective integration"""
        brain = SocialBrain()
        brain.process_cognitive(np.random.rand(50))
        brain.process_affective(np.random.rand(50))
        integrated = brain.integrate()
        assert len(integrated) == 50

    def test_get_activations(self):
        """Test getting activations"""
        brain = SocialBrain()
        brain.process_cognitive(np.random.rand(50))
        activations = brain.get_activations()
        assert "cognitive" in activations
        assert "affective" in activations
        assert "integrated" in activations


class TestSocialCognitionNetwork:
    """Tests for full social cognition network"""

    def test_initialization(self):
        """Test network initialization"""
        network = SocialCognitionNetwork()
        assert network.theory_of_mind is not None
        assert network.mentalizing is not None
        assert network.empathy is not None
        assert network.perspective_taking is not None

    def test_process_social_stimulus(self):
        """Test processing social stimulus"""
        network = SocialCognitionNetwork()
        behavior = np.random.rand(50)
        result = network.process_social_stimulus("agent1", behavior)
        assert "mental_states" in result
        assert "tom" in result
        assert "social_brain" in result

    def test_process_with_emotion(self):
        """Test processing with emotional display"""
        network = SocialCognitionNetwork()
        behavior = np.random.rand(50)
        emotion = np.random.rand(50)
        result = network.process_social_stimulus(
            "agent1", behavior, emotional_display=emotion, is_distress=True
        )
        assert "empathy" in result

    def test_take_perspective(self):
        """Test perspective taking"""
        network = SocialCognitionNetwork()
        network.process_social_stimulus("agent1", np.random.rand(50))
        result = network.take_perspective("agent1", np.random.rand(50), "visual")
        assert "their_view" in result

    def test_predict_behavior(self):
        """Test behavior prediction"""
        network = SocialCognitionNetwork()
        network.process_social_stimulus("agent1", np.random.rand(50))
        prediction = network.predict_behavior("agent1")
        assert "predicted_behavior" in prediction

    def test_false_belief(self):
        """Test false belief checking"""
        network = SocialCognitionNetwork()
        reality = np.ones(50)
        false_belief = -np.ones(50)
        result = network.check_false_belief("agent1", reality, false_belief)
        assert result["false_belief_detected"]

    def test_empathic_response(self):
        """Test getting empathic response"""
        network = SocialCognitionNetwork()
        network.process_social_stimulus(
            "agent1", np.random.rand(50), emotional_display=np.ones(50), is_distress=True
        )
        response = network.get_empathic_response("agent1")
        assert "empathy_level" in response
        assert "helping_motivation" in response

    def test_process_self(self):
        """Test self processing"""
        network = SocialCognitionNetwork()
        own_state = np.random.rand(50)
        network.process_self(own_state)
        # Should update self models
        assert np.sum(np.abs(network.perspective_taking.self_other.self_representation)) > 0

    def test_update(self):
        """Test system update"""
        network = SocialCognitionNetwork()
        network.process_social_stimulus("agent1", np.random.rand(50))
        network.update(dt=1.0)
        # Should not error

    def test_get_state(self):
        """Test state retrieval"""
        network = SocialCognitionNetwork()
        network.process_social_stimulus("agent1", np.random.rand(50))
        state = network.get_state()
        assert "tom" in state
        assert "mentalizing" in state
        assert "empathy" in state
        assert "perspective" in state
        assert "social_brain" in state


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
