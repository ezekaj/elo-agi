"""Tests for empathy system"""

import numpy as np
import pytest
from neuro.modules.m15_social.empathy import (
    AffectiveSharing,
    EmpathicConcern,
    EmpathySystem,
)


class TestAffectiveSharing:
    """Tests for affective sharing"""

    def test_initialization(self):
        """Test sharing initialization"""
        sharing = AffectiveSharing()
        assert np.allclose(sharing.own_affect, 0)

    def test_observe_emotion(self):
        """Test observing another's emotion"""
        sharing = AffectiveSharing()
        emotion = np.random.rand(50)
        result = sharing.observe_emotion(emotion)
        assert "shared_affect" in result
        assert "insula_activity" in result

    def test_emotional_contagion(self):
        """Test emotions are shared"""
        sharing = AffectiveSharing()
        strong_emotion = np.ones(50)
        sharing.observe_emotion(strong_emotion)
        assert np.sum(np.abs(sharing.own_affect)) > 0

    def test_emotional_resonance(self):
        """Test resonance calculation"""
        sharing = AffectiveSharing()
        sharing.own_affect = np.ones(50)
        similar = np.ones(50) * 0.9
        resonance = sharing.get_emotional_resonance(similar)
        assert resonance > 0.5

    def test_regulate_affect(self):
        """Test affect regulation"""
        sharing = AffectiveSharing()
        sharing.own_affect = np.ones(50)
        target = np.zeros(50)
        regulation = sharing.regulate_affect(target)
        assert np.sum(np.abs(regulation)) > 0


class TestEmpathicConcern:
    """Tests for empathic concern"""

    def test_initialization(self):
        """Test concern initialization"""
        concern = EmpathicConcern()
        assert concern.prosocial_motivation == 0.0

    def test_observe_distress(self):
        """Test observing distress"""
        concern = EmpathicConcern()
        distress = np.ones(50)
        result = concern.observe_distress("person1", distress)
        assert result["concern_level"] > 0
        assert result["acc_activity"] > 0

    def test_helping_motivation(self):
        """Test helping motivation increases"""
        concern = EmpathicConcern()
        concern.observe_distress("person1", np.ones(50))
        motivation = concern.get_helping_motivation("person1")
        assert motivation > 0

    def test_relief_response(self):
        """Test relief reduces concern"""
        concern = EmpathicConcern()
        concern.observe_distress("person1", np.ones(50))
        initial_concern = concern.concern_levels["person1"]
        concern.relief_response("person1", np.zeros(50))
        assert concern.concern_levels["person1"] < initial_concern


class TestEmpathySystem:
    """Tests for integrated empathy"""

    def test_initialization(self):
        """Test system initialization"""
        system = EmpathySystem()
        assert system.empathy_level == 0.5

    def test_empathize(self):
        """Test full empathic response"""
        system = EmpathySystem()
        emotion = np.random.rand(50)
        result = system.empathize("person1", emotion)
        assert "resonance" in result
        assert "empathy_level" in result

    def test_empathize_with_distress(self):
        """Test empathy for distress"""
        system = EmpathySystem()
        distress = np.ones(50)
        result = system.empathize("person1", distress, is_distress=True)
        assert result["concern"] is not None
        assert result["helping_motivation"] > 0

    def test_perspective_enhances_empathy(self):
        """Test perspective taking enhances empathy"""
        system = EmpathySystem()
        system.take_perspective(True)
        assert system.perspective_active

    def test_regulate_empathy(self):
        """Test empathy regulation"""
        system = EmpathySystem()
        system.empathy_level = 0.9
        system.regulate_empathy(0.5)
        assert system.empathy_level == 0.5

    def test_update(self):
        """Test system update"""
        system = EmpathySystem()
        system.empathize("person1", np.ones(50), is_distress=True)
        initial_motivation = system.empathic_concern.prosocial_motivation
        system.update(dt=10.0)
        assert system.empathic_concern.prosocial_motivation < initial_motivation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
