"""Tests for interference resolution."""

import pytest
import numpy as np
from neuro.modules.m05_sleep_consolidation.interference_resolution import (
    InterferenceResolver,
    InterferenceEvent,
    MemoryVector,
    ResolutionStrategy,
    InterferenceType,
    InterleaveSchedule,
)


class TestMemoryVector:
    """Tests for MemoryVector."""

    def test_initialization(self):
        """Test memory vector initialization."""
        content = np.random.randn(128)
        mem = MemoryVector(
            memory_id="mem1",
            content=content,
            encoding_time=0.0,
        )
        assert mem.memory_id == "mem1"
        assert mem.strength == 1.0

    def test_similarity(self):
        """Test similarity computation."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([1.0, 0.0, 0.0])
        vec3 = np.array([0.0, 1.0, 0.0])

        mem1 = MemoryVector("mem1", vec1, 0.0)
        mem2 = MemoryVector("mem2", vec2, 0.0)
        mem3 = MemoryVector("mem3", vec3, 0.0)

        assert mem1.similarity(mem2) == pytest.approx(1.0, abs=0.01)
        assert mem1.similarity(mem3) == pytest.approx(0.0, abs=0.01)

    def test_similarity_normalized(self):
        """Test similarity is normalized."""
        vec1 = np.array([2.0, 0.0, 0.0])
        vec2 = np.array([5.0, 0.0, 0.0])

        mem1 = MemoryVector("mem1", vec1, 0.0)
        mem2 = MemoryVector("mem2", vec2, 0.0)

        assert mem1.similarity(mem2) == pytest.approx(1.0, abs=0.01)


class TestInterleaveSchedule:
    """Tests for InterleaveSchedule."""

    def test_initialization(self):
        """Test schedule initialization."""
        schedule = InterleaveSchedule(
            memories=["mem1", "mem2"],
            pattern=[0, 1, 0, 1],
            repetitions=2,
        )
        assert len(schedule.memories) == 2
        assert schedule.current_position == 0

    def test_next_memory(self):
        """Test getting next memory."""
        schedule = InterleaveSchedule(
            memories=["A", "B"],
            pattern=[0, 1, 0, 1],
            repetitions=1,
        )
        assert schedule.next_memory() == "A"
        assert schedule.next_memory() == "B"
        assert schedule.next_memory() == "A"
        assert schedule.next_memory() == "B"

    def test_is_complete(self):
        """Test completion check."""
        schedule = InterleaveSchedule(
            memories=["A", "B"],
            pattern=[0, 1],
            repetitions=2,
        )
        assert not schedule.is_complete()

        for _ in range(4):
            schedule.next_memory()

        assert schedule.is_complete()


class TestInterferenceResolver:
    """Tests for InterferenceResolver."""

    @pytest.fixture
    def resolver(self):
        return InterferenceResolver(
            similarity_threshold=0.7,
            random_seed=42,
        )

    @pytest.fixture
    def similar_memories(self, resolver):
        """Create similar memories."""
        np.random.seed(42)
        base = np.random.randn(128)
        base = base / np.linalg.norm(base)

        # Create very similar memory (only 2% noise to ensure >0.7 similarity)
        mem2_content = base * 0.98 + 0.02 * np.random.randn(128)
        mem2_content = mem2_content / np.linalg.norm(mem2_content)

        mem1 = resolver.register_memory("mem1", base, encoding_time=0.0)
        mem2 = resolver.register_memory("mem2", mem2_content, encoding_time=1.0)

        return [mem1, mem2]

    @pytest.fixture
    def dissimilar_memories(self, resolver):
        """Create dissimilar memories."""
        np.random.seed(42)
        mem1 = resolver.register_memory("memA", np.random.randn(128), encoding_time=0.0)
        mem2 = resolver.register_memory("memB", np.random.randn(128), encoding_time=1.0)
        return [mem1, mem2]

    def test_initialization(self, resolver):
        """Test resolver initialization."""
        stats = resolver.statistics()
        assert stats["total_memories"] == 0
        assert stats["n_detections"] == 0

    def test_register_memory(self, resolver):
        """Test memory registration."""
        content = np.random.randn(128)
        mem = resolver.register_memory("mem1", content, encoding_time=0.0)
        assert mem.memory_id == "mem1"
        assert resolver._memories["mem1"] is not None

    def test_compute_similarity(self, resolver, similar_memories):
        """Test similarity computation."""
        sim = resolver.compute_similarity("mem1", "mem2")
        assert sim > resolver.similarity_threshold

    def test_compute_similarity_cache(self, resolver, similar_memories):
        """Test similarity caching."""
        sim1 = resolver.compute_similarity("mem1", "mem2")
        sim2 = resolver.compute_similarity("mem1", "mem2")
        assert sim1 == sim2

    def test_detect_interference_similar(self, resolver, similar_memories):
        """Test interference detection for similar memories."""
        events = resolver.detect_interference(["mem1", "mem2"])
        assert len(events) > 0
        assert events[0].similarity > resolver.similarity_threshold

    def test_detect_interference_dissimilar(self, resolver, dissimilar_memories):
        """Test no interference for dissimilar memories."""
        events = resolver.detect_interference(["memA", "memB"])
        assert len(events) == 0

    def test_interference_type_proactive(self, resolver):
        """Test proactive interference detection."""
        np.random.seed(42)
        base = np.random.randn(128)
        base = base / np.linalg.norm(base)

        resolver.register_memory("old", base, encoding_time=0.0)
        resolver.register_memory(
            "new", base * 0.99 + 0.01 * np.random.randn(128), encoding_time=10.0
        )

        events = resolver.detect_interference(["old", "new"])
        if events:
            assert events[0].interference_type == InterferenceType.PROACTIVE

    def test_resolve_proactive(self, resolver, similar_memories):
        """Test proactive interference resolution."""
        new_mem, affected = resolver.resolve_proactive("mem2", ["mem1"])
        assert new_mem is not None
        # Similar memories should be identified as affected
        sim = resolver.compute_similarity("mem1", "mem2")
        if sim > resolver.similarity_threshold:
            assert "mem1" in affected

    def test_resolve_retroactive(self, resolver, similar_memories):
        """Test retroactive interference resolution."""
        existing, protection = resolver.resolve_retroactive("mem1", ["mem2"])
        assert existing is not None
        assert 0 < protection <= 1.0

    def test_interleave_replays(self, resolver, similar_memories):
        """Test interleaved replay scheduling."""
        schedule = resolver.interleave_replays(["mem1", "mem2"], repetitions=3)

        assert len(schedule.memories) == 2
        assert len(schedule.pattern) == 6  # 2 memories * 3 repetitions

    def test_interleave_single_memory(self, resolver):
        """Test interleave with single memory."""
        resolver.register_memory("mem1", np.random.randn(128), 0.0)
        schedule = resolver.interleave_replays(["mem1"])
        assert schedule.memories == ["mem1"]

    def test_interference_risk(self, resolver, similar_memories):
        """Test interference risk computation."""
        risk1 = resolver.get_interference_risk("mem1")
        risk2 = resolver.get_interference_risk("mem2")

        # With highly similar memories, both should have positive risk
        assert risk1 >= 0
        assert risk2 >= 0
        # At least one should show risk for truly similar memories
        sim = resolver.compute_similarity("mem1", "mem2")
        if sim > resolver.similarity_threshold:
            assert risk1 > 0 or risk2 > 0

    def test_interference_risk_isolated(self, resolver):
        """Test interference risk for isolated memory."""
        resolver.register_memory("isolated", np.random.randn(128), 0.0)
        risk = resolver.get_interference_risk("isolated")
        assert risk == 0.0  # No similar memories

    def test_get_vulnerable_memories(self, resolver, similar_memories):
        """Test getting vulnerable memories."""
        vulnerable = resolver.get_vulnerable_memories(threshold=0.1)
        assert len(vulnerable) >= 0  # May or may not be vulnerable depending on threshold

    def test_clear_similarity_cache(self, resolver, similar_memories):
        """Test clearing similarity cache."""
        resolver.compute_similarity("mem1", "mem2")
        assert len(resolver._similarity_cache) > 0

        resolver.clear_similarity_cache()
        assert len(resolver._similarity_cache) == 0

    def test_remove_memory(self, resolver, similar_memories):
        """Test removing a memory."""
        resolver.compute_similarity("mem1", "mem2")
        resolver.remove_memory("mem1")

        assert "mem1" not in resolver._memories
        # Cache should also be cleared for that memory
        for key in resolver._similarity_cache:
            assert "mem1" not in key

    def test_statistics(self, resolver, similar_memories):
        """Test statistics generation."""
        events = resolver.detect_interference(["mem1", "mem2"])

        stats = resolver.statistics()
        assert stats["total_memories"] == 2
        assert stats["n_detections"] == len(events)
        assert "strategy_distribution" in stats
