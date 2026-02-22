"""Tests for situated cognition"""

import numpy as np
import pytest
from neuro.modules.m14_embodied.situated_cognition import (
    ExternalMemory,
    SituatedContext,
    ContextualReasoner,
)


class TestExternalMemory:
    """Tests for external memory"""

    def test_initialization(self):
        """Test memory initialization"""
        memory = ExternalMemory(capacity=10)

        assert memory.capacity == 10
        assert len(memory.storage) == 0

    def test_store(self):
        """Test storing in external memory"""
        memory = ExternalMemory()

        memory.store("note1", "remember this")

        assert "note1" in memory.storage
        assert memory.storage["note1"] == "remember this"

    def test_retrieve(self):
        """Test retrieving from memory"""
        memory = ExternalMemory()
        memory.store("note1", "remember this")

        retrieved = memory.retrieve("note1")

        assert retrieved == "remember this"

    def test_capacity_limit(self):
        """Test capacity is enforced"""
        memory = ExternalMemory(capacity=3)

        for i in range(5):
            memory.store(f"note{i}", f"content{i}")

        assert len(memory.storage) <= 3

    def test_location_search(self):
        """Test searching by location"""
        memory = ExternalMemory()

        memory.store("note1", "content1", location=np.array([0, 0]))
        memory.store("note2", "content2", location=np.array([10, 10]))

        matches = memory.search_by_location(np.array([0.1, 0.1]), threshold=1.0)

        assert "note1" in matches
        assert "note2" not in matches

    def test_attention_decay(self):
        """Test attention decays"""
        memory = ExternalMemory()
        memory.store("note1", "content")

        initial = memory.attention["note1"]
        memory.decay_attention(dt=10.0, decay_rate=0.1)

        assert memory.attention["note1"] < initial


class TestSituatedContext:
    """Tests for situated context"""

    def test_initialization(self):
        """Test context initialization"""
        context = SituatedContext()

        assert len(context.context_features) > 0
        assert len(context.context_history) == 0

    def test_set_physical_context(self):
        """Test setting physical context"""
        context = SituatedContext()

        features = np.random.rand(len(context.physical_context))
        context.set_physical_context(features)

        assert not np.allclose(context.physical_context, 0)

    def test_set_social_context(self):
        """Test setting social context"""
        context = SituatedContext()

        features = np.random.rand(len(context.social_context))
        context.set_social_context(features)

        assert not np.allclose(context.social_context, 0)

    def test_modulate_processing(self):
        """Test context modulates processing"""
        context = SituatedContext()
        context.set_physical_context(np.ones(len(context.physical_context)))

        input_pattern = np.zeros(len(context.context_features))
        modulated = context.modulate_processing(input_pattern)

        # Should be different due to context influence
        assert not np.allclose(modulated, input_pattern)

    def test_context_similarity(self):
        """Test context similarity computation"""
        context = SituatedContext()
        context.set_physical_context(np.ones(len(context.physical_context)))

        same_context = context.context_features.copy()
        similarity = context.get_context_similarity(same_context)

        assert similarity > 0.9

    def test_detect_context_change(self):
        """Test context change detection"""
        context = SituatedContext()

        context.set_physical_context(np.zeros(len(context.physical_context)))
        context.set_physical_context(np.ones(len(context.physical_context)))

        assert context.detect_context_change(threshold=0.1)


class TestContextualReasoner:
    """Tests for contextual reasoning"""

    def test_initialization(self):
        """Test reasoner initialization"""
        reasoner = ContextualReasoner()

        assert reasoner.context is not None
        assert reasoner.external_memory is not None

    def test_set_context(self):
        """Test setting full context"""
        reasoner = ContextualReasoner()
        params = reasoner.params

        physical = np.random.rand(params.n_features // 3)
        social = np.random.rand(params.n_features // 3)
        task = np.random.rand(params.n_features // 3)

        reasoner.set_context(physical, social, task)

        assert not np.allclose(reasoner.context.physical_context, 0)

    def test_reason_in_context(self):
        """Test contextual reasoning"""
        reasoner = ContextualReasoner()

        problem = np.random.rand(reasoner.params.n_features)
        result = reasoner.reason_in_context(problem)

        assert "solution" in result
        assert "context_influence" in result

    def test_offload_to_environment(self):
        """Test cognitive offloading"""
        reasoner = ContextualReasoner()

        reasoner.offload_to_environment("reminder", "meeting at 3pm")

        assert "reminder" in reasoner.external_memory.storage

    def test_store_retrieve_knowledge(self):
        """Test context-specific knowledge storage"""
        reasoner = ContextualReasoner()

        knowledge = np.random.rand(50)
        reasoner.store_knowledge("work_context", knowledge)

        retrieved = reasoner.retrieve_contextual_knowledge("work_context")

        assert len(retrieved) == 1

    def test_update(self):
        """Test reasoner update"""
        reasoner = ContextualReasoner()
        reasoner.offload_to_environment("note", "content")

        initial = reasoner.external_memory.attention["note"]
        reasoner.update(dt=10.0)

        assert reasoner.external_memory.attention["note"] < initial


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
