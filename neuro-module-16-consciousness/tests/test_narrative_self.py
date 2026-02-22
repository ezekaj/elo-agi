"""Tests for narrative self"""

import numpy as np
import pytest
from neuro.modules.m16_consciousness.narrative_self import (
    AutobiographicalMemory,
    SelfConcept,
    NarrativeSelf,
    NarrativeParams,
)


class TestAutobiographicalMemory:
    """Tests for autobiographical memory"""

    def test_initialization(self):
        """Test memory initialization"""
        memory = AutobiographicalMemory()
        assert memory.get_memory_count() == 0

    def test_encode_episode(self):
        """Test encoding an episode"""
        memory = AutobiographicalMemory()
        content = np.random.rand(50)
        context = {"location": "home", "time": "morning"}

        idx = memory.encode_episode(content, context, emotional_salience=0.7)

        assert idx >= 0
        assert memory.get_memory_count() == 1

    def test_retrieve_by_cue(self):
        """Test retrieval by cue"""
        memory = AutobiographicalMemory()

        # Encode similar episodes
        content = np.random.rand(50)
        memory.encode_episode(content, {"tag": "test"})

        # Retrieve with same cue
        results = memory.retrieve_by_cue(content, n=5)
        assert len(results) > 0

    def test_retrieve_recent(self):
        """Test recent memory retrieval"""
        memory = AutobiographicalMemory()

        for i in range(5):
            memory.encode_episode(np.random.rand(50), {"index": i})

        recent = memory.retrieve_recent(n=3)
        assert len(recent) == 3

    def test_capacity_limit(self):
        """Test memory capacity limit"""
        params = NarrativeParams(memory_capacity=5)
        memory = AutobiographicalMemory(params)

        for i in range(10):
            memory.encode_episode(np.random.rand(50), {})

        assert memory.get_memory_count() <= 5


class TestSelfConcept:
    """Tests for self-concept"""

    def test_initialization(self):
        """Test self-concept initialization"""
        sc = SelfConcept()
        assert len(sc.traits) == 0
        assert len(sc.roles) == 0

    def test_update_trait(self):
        """Test trait updating"""
        sc = SelfConcept()
        sc.update_trait("extraversion", 0.7)
        assert "extraversion" in sc.traits
        assert sc.traits["extraversion"] == 0.7

    def test_add_role(self):
        """Test adding role"""
        sc = SelfConcept()
        sc.add_role("student")
        assert "student" in sc.roles

    def test_set_value(self):
        """Test setting value"""
        sc = SelfConcept()
        sc.set_value("honesty", 0.9)
        assert "honesty" in sc.values

    def test_process_self_relevant(self):
        """Test self-relevant processing"""
        sc = SelfConcept()
        stimulus = np.random.rand(50)
        result = sc.process_self_relevant(stimulus)
        assert "self_relevance" in result
        assert "mpfc_activity" in result


class TestNarrativeSelf:
    """Tests for integrated narrative self"""

    def test_initialization(self):
        """Test narrative self initialization"""
        ns = NarrativeSelf()
        assert ns.memory is not None
        assert ns.self_concept is not None

    def test_experience_event(self):
        """Test experiencing an event"""
        ns = NarrativeSelf()
        event = np.random.rand(50)
        context = {"what": "test event"}

        result = ns.experience_event(event, context, emotional_impact=0.6)

        assert "memory_encoded" in result
        assert "narrative_coherence" in result

    def test_recall_life_period(self):
        """Test recalling life period"""
        ns = NarrativeSelf()

        # Create some memories
        cue = np.random.rand(50)
        for _ in range(5):
            ns.experience_event(cue + np.random.rand(50) * 0.1, {})

        result = ns.recall_life_period(cue)
        assert "memories" in result

    def test_reflect_on_self(self):
        """Test self-reflection"""
        ns = NarrativeSelf()
        ns.experience_event(np.random.rand(50), {})

        result = ns.reflect_on_self()
        assert "self_concept" in result
        assert "narrative_coherence" in result

    def test_get_state(self):
        """Test getting state"""
        ns = NarrativeSelf()
        state = ns.get_narrative_self_state()
        assert "memory_count" in state
        assert "narrative_coherence" in state


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
