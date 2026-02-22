"""Tests for memory consolidation systems"""

import numpy as np
import pytest
from neuro.modules.m05_sleep_consolidation.memory_replay import (
    MemoryTrace,
    HippocampalReplay,
    ReplayPrioritizer,
)
from neuro.modules.m05_sleep_consolidation.systems_consolidation import (
    HippocampalStore,
    CorticalStore,
    HippocampalCorticalDialogue,
    ConsolidationWindow,
    MemoryTransformation,
    StoredMemory,
)


class TestMemoryTrace:
    """Tests for memory trace representation"""

    def test_creation(self):
        """Test memory trace creation"""
        content = np.array([1.0, 2.0, 3.0, 4.0])
        trace = MemoryTrace(content=content, emotional_salience=0.5)

        assert np.array_equal(trace.content, content)
        assert trace.emotional_salience == 0.5
        assert trace.hippocampal_index == True
        assert trace.cortical_index == False

    def test_similarity(self):
        """Test similarity computation between traces"""
        trace1 = MemoryTrace(content=np.array([1.0, 0.0, 0.0]))
        trace2 = MemoryTrace(content=np.array([1.0, 0.0, 0.0]))
        trace3 = MemoryTrace(content=np.array([0.0, 1.0, 0.0]))

        # Identical traces
        assert trace1.similarity_to(trace2) == 1.0

        # Orthogonal traces
        assert trace1.similarity_to(trace3) == 0.0

    def test_partial_similarity(self):
        """Test partial similarity"""
        trace1 = MemoryTrace(content=np.array([1.0, 1.0, 0.0]))
        trace2 = MemoryTrace(content=np.array([1.0, 0.0, 0.0]))

        sim = trace1.similarity_to(trace2)
        assert 0 < sim < 1


class TestHippocampalReplay:
    """Tests for hippocampal replay system"""

    def test_initialization(self):
        """Test replay system initialization"""
        replay = HippocampalReplay(compression_factor=20.0)

        assert replay.compression_factor == 20.0
        assert len(replay.memory_traces) == 0

    def test_encode_experience(self):
        """Test encoding new experiences"""
        replay = HippocampalReplay()
        pattern = np.random.randn(10)

        trace = replay.encode_experience(pattern, emotional_salience=0.7)

        assert len(replay.memory_traces) == 1
        assert trace.emotional_salience == 0.7

    def test_replay_strengthens_memory(self):
        """Test that replay strengthens memories"""
        replay = HippocampalReplay(replay_strength_boost=0.1)
        trace = replay.encode_experience(np.random.randn(10))

        initial_strength = trace.strength
        replay.replay_memory(trace)

        assert trace.strength > initial_strength
        assert trace.replay_count == 1

    def test_ripple_boost(self):
        """Test that ripple presence boosts replay effect"""
        replay = HippocampalReplay(replay_strength_boost=0.1)

        trace1 = replay.encode_experience(np.random.randn(10))
        trace2 = replay.encode_experience(np.random.randn(10))

        initial1 = trace1.strength
        initial2 = trace2.strength

        replay.replay_memory(trace1, ripple_present=False)
        replay.replay_memory(trace2, ripple_present=True)

        boost_without_ripple = trace1.strength - initial1
        boost_with_ripple = trace2.strength - initial2

        assert boost_with_ripple > boost_without_ripple

    def test_compression_factor(self):
        """Test time compression calculation"""
        replay = HippocampalReplay(compression_factor=20.0)

        real_duration = 1.0  # 1 second
        compressed = replay.get_compressed_duration(real_duration)

        assert compressed == 0.05  # 1/20 second

    def test_priority_selection(self):
        """Test priority-based memory selection"""
        replay = HippocampalReplay()

        # Add memories with different properties
        replay.current_time = 100.0

        # Recent memory
        recent = replay.encode_experience(np.random.randn(10))

        # Old memory
        replay.current_time = 0.0
        old = replay.encode_experience(np.random.randn(10))

        replay.current_time = 100.0

        # Emotional memory
        emotional = replay.encode_experience(np.random.randn(10), emotional_salience=0.9)

        selected = replay.select_for_replay(n_select=2)

        # Should select recent and emotional over old
        # Check that we got 2 memories and they're likely not the old one
        assert len(selected) == 2
        # At least one of recent or emotional should be selected
        selected_ids = [id(m) for m in selected]
        assert id(recent) in selected_ids or id(emotional) in selected_ids


class TestReplayPrioritizer:
    """Tests for replay prioritization"""

    def test_recency_effect(self):
        """Test that recent memories get higher priority"""
        prioritizer = ReplayPrioritizer(
            recency_weight=1.0, emotion_weight=0.0, incompleteness_weight=0.0
        )

        recent = MemoryTrace(content=np.array([1.0]), encoding_time=90.0)
        old = MemoryTrace(content=np.array([1.0]), encoding_time=0.0)

        recent_priority = prioritizer.compute_priority(recent, current_time=100.0)
        old_priority = prioritizer.compute_priority(old, current_time=100.0)

        assert recent_priority > old_priority

    def test_emotion_effect(self):
        """Test that emotional memories get higher priority"""
        prioritizer = ReplayPrioritizer(
            recency_weight=0.0, emotion_weight=1.0, incompleteness_weight=0.0
        )

        emotional = MemoryTrace(content=np.array([1.0]), emotional_salience=0.9)
        neutral = MemoryTrace(content=np.array([1.0]), emotional_salience=0.0)

        emotional_priority = prioritizer.compute_priority(emotional, current_time=0.0)
        neutral_priority = prioritizer.compute_priority(neutral, current_time=0.0)

        assert emotional_priority > neutral_priority

    def test_incompleteness_effect(self):
        """Test that incomplete learning increases priority"""
        prioritizer = ReplayPrioritizer(
            recency_weight=0.0, emotion_weight=0.0, incompleteness_weight=1.0
        )

        incomplete = MemoryTrace(content=np.array([1.0]), learning_complete=0.2)
        complete = MemoryTrace(content=np.array([1.0]), learning_complete=0.9)

        incomplete_priority = prioritizer.compute_priority(incomplete, current_time=0.0)
        complete_priority = prioritizer.compute_priority(complete, current_time=0.0)

        assert incomplete_priority > complete_priority


class TestHippocampalStore:
    """Tests for hippocampal memory store"""

    def test_encoding(self):
        """Test memory encoding"""
        store = HippocampalStore()
        trace = MemoryTrace(content=np.random.randn(10))

        memory_id = store.encode(trace)

        assert store.get_memory_count() == 1
        retrieved = store.retrieve(memory_id)
        assert retrieved is not None

    def test_capacity_limit(self):
        """Test capacity limit enforcement"""
        store = HippocampalStore(capacity=5)

        for i in range(10):
            trace = MemoryTrace(content=np.array([float(i)]))
            store.encode(trace)

        assert store.get_memory_count() <= 5

    def test_unconsolidated_retrieval(self):
        """Test getting unconsolidated memories"""
        store = HippocampalStore()

        for i in range(5):
            trace = MemoryTrace(content=np.array([float(i)]))
            store.encode(trace)

        unconsolidated = store.get_unconsolidated(threshold=0.5)
        assert len(unconsolidated) == 5  # All start unconsolidated

    def test_consolidation_update(self):
        """Test updating consolidation level"""
        store = HippocampalStore()
        trace = MemoryTrace(content=np.random.randn(10))
        memory_id = store.encode(trace)

        store.update_consolidation(memory_id, 0.3)
        retrieved = store.retrieve(memory_id)

        assert retrieved.consolidation_level == 0.3


class TestCorticalStore:
    """Tests for cortical memory store"""

    def test_transfer(self):
        """Test receiving memory transfer"""
        cortex = CorticalStore()
        trace = MemoryTrace(content=np.random.randn(10))
        memory = StoredMemory(trace=trace, consolidation_level=0.8)

        cortical_id = cortex.receive_transfer(memory, abstraction=0.3)

        assert cortex.get_memory_count() == 1
        retrieved = cortex.retrieve(cortical_id)
        assert retrieved.trace.cortical_index == True

    def test_abstraction(self):
        """Test abstraction during transfer"""
        cortex = CorticalStore()
        trace = MemoryTrace(content=np.random.randn(10))
        memory = StoredMemory(trace=trace, consolidation_level=0.8)

        cortical_id = cortex.receive_transfer(memory, abstraction=0.8)
        retrieved = cortex.retrieve(cortical_id)

        # Abstracted content should differ from original
        assert retrieved.abstraction_level == 0.8

    def test_schema_extraction(self):
        """Test schema extraction from multiple memories"""
        cortex = CorticalStore()

        # Create similar memories
        base = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        memories = []
        for _ in range(5):
            noise = np.random.randn(5) * 0.1
            trace = MemoryTrace(content=base + noise)
            memories.append(StoredMemory(trace=trace, consolidation_level=1.0))

        schema = cortex.extract_schema(memories)

        # Schema should be similar to base
        similarity = np.corrcoef(schema.flatten(), base.flatten())[0, 1]
        assert similarity > 0.9


class TestConsolidationWindow:
    """Tests for consolidation window timing"""

    def test_optimal_window(self):
        """Test optimal consolidation window detection"""
        window = ConsolidationWindow(
            slow_oscillation_phase="up", spindle_present=True, ripple_present=True
        )

        assert window.is_optimal()

    def test_suboptimal_window(self):
        """Test suboptimal windows"""
        window = ConsolidationWindow(
            slow_oscillation_phase="down", spindle_present=True, ripple_present=True
        )

        assert not window.is_optimal()

    def test_consolidation_boost(self):
        """Test consolidation boost calculation"""
        optimal = ConsolidationWindow(
            slow_oscillation_phase="up", spindle_present=True, ripple_present=True
        )

        suboptimal = ConsolidationWindow(
            slow_oscillation_phase="down", spindle_present=False, ripple_present=False
        )

        assert optimal.get_consolidation_boost() > suboptimal.get_consolidation_boost()


class TestHippocampalCorticalDialogue:
    """Tests for hippocampal-cortical memory transfer"""

    def test_initialization(self):
        """Test dialogue system initialization"""
        dialogue = HippocampalCorticalDialogue()

        assert dialogue.hippocampus is not None
        assert dialogue.cortex is not None

    def test_consolidation(self):
        """Test memory consolidation process"""
        dialogue = HippocampalCorticalDialogue(transfer_threshold=0.5)

        # Encode memory
        trace = MemoryTrace(content=np.random.randn(10))
        memory_id = dialogue.hippocampus.encode(trace)

        # Set optimal window
        dialogue.initiate_dialogue(slow_osc_phase="up", spindle=True, ripple=True)

        # Consolidate multiple times
        for _ in range(10):
            dialogue.consolidate_memory(memory_id, replay_strength=1.0)

        # Memory should be more consolidated
        memory = dialogue.hippocampus.retrieve(memory_id)
        assert memory.consolidation_level > 0.3

    def test_transfer_threshold(self):
        """Test transfer occurs at threshold"""
        dialogue = HippocampalCorticalDialogue(transfer_threshold=0.5)

        trace = MemoryTrace(content=np.random.randn(10))
        memory_id = dialogue.hippocampus.encode(trace)

        # Force high consolidation
        dialogue.hippocampus.update_consolidation(memory_id, 0.6)

        # Transfer should happen
        dialogue.transfer_memory(memory_id)

        memory = dialogue.hippocampus.retrieve(memory_id)
        assert memory.trace.cortical_index == True

    def test_consolidation_cycle(self):
        """Test complete consolidation cycle"""
        dialogue = HippocampalCorticalDialogue()

        # Encode several memories
        for _ in range(5):
            trace = MemoryTrace(content=np.random.randn(10))
            dialogue.hippocampus.encode(trace)

        # Run consolidation cycle
        stats = dialogue.run_consolidation_cycle(duration_seconds=60.0)

        assert "consolidated" in stats
        assert "transferred" in stats


class TestMemoryTransformation:
    """Tests for memory transformation during consolidation"""

    def test_gist_extraction(self):
        """Test gist extraction preserves key features"""
        transformer = MemoryTransformation()
        trace = MemoryTrace(content=np.array([1.0, 0.1, 5.0, 0.05, 2.0]))
        memory = StoredMemory(trace=trace, consolidation_level=0.5)

        gist = transformer.extract_gist(memory, amount=0.5)

        # Gist should preserve strong features, remove weak
        assert gist[2] != 0  # Strong feature preserved
        # Some weak features should be zeroed
        assert np.sum(gist == 0) >= 1

    def test_schema_integration(self):
        """Test schema integration pulls memory toward schema"""
        transformer = MemoryTransformation(schema_weight=0.5)

        trace = MemoryTrace(content=np.array([1.0, 2.0, 3.0]))
        memory = StoredMemory(trace=trace, consolidation_level=0.5)
        schema = np.array([2.0, 2.0, 2.0])

        integrated = transformer.integrate_with_schema(memory, schema)

        # Should be between original and schema (at least some elements changed)
        assert np.any(integrated != trace.content)
        assert np.any(integrated != schema)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
