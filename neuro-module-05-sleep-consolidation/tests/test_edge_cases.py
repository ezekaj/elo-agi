"""Tests for edge cases and boundary conditions.

These tests verify the system handles extreme, unusual, or boundary
conditions gracefully without crashing or producing invalid results.
"""

import pytest
import numpy as np
from neuro.modules.m05_sleep_consolidation.spaced_repetition import (
    SpacedRepetitionScheduler,
    ReviewQuality,
)
from neuro.modules.m05_sleep_consolidation.meta_learning import (
    MetaLearningController,
    MemoryType,
    LearningCurve,
)
from neuro.modules.m05_sleep_consolidation.interference_resolution import (
    InterferenceResolver,
    MemoryVector,
)
from neuro.modules.m05_sleep_consolidation.schema_refinement import SchemaRefiner, Schema


class TestZeroVectorHandling:
    """Tests for zero vector edge cases."""

    def test_interference_zero_vector_similarity(self):
        """Similarity with zero vector should not crash or return NaN."""
        resolver = InterferenceResolver(random_seed=42)

        resolver.register_memory("zero", np.zeros(128), 0.0)
        resolver.register_memory("normal", np.random.randn(128), 0.0)

        sim = resolver.compute_similarity("zero", "normal")
        assert np.isfinite(sim), "Similarity should be finite, not NaN/inf"
        assert sim == 0.0, "Zero vector should have 0 similarity"

    def test_interference_both_zero_vectors(self):
        """Two zero vectors should have 0 similarity (not NaN)."""
        resolver = InterferenceResolver(random_seed=42)

        resolver.register_memory("zero1", np.zeros(128), 0.0)
        resolver.register_memory("zero2", np.zeros(128), 0.0)

        sim = resolver.compute_similarity("zero1", "zero2")
        assert np.isfinite(sim), "Similarity should be finite"
        assert sim == 0.0

    def test_schema_zero_prototype(self):
        """Schema with zero prototype should handle similarity gracefully."""
        refiner = SchemaRefiner(random_seed=42)

        schema = refiner.create_schema("zero", np.zeros(64), instances=[])
        sim = schema.similarity_to(np.random.randn(64))

        assert np.isfinite(sim), "Similarity should be finite"
        assert sim == 0.0


class TestEmptyInputs:
    """Tests for empty input handling."""

    def test_scheduler_no_memories(self):
        """Scheduler with no memories should work correctly."""
        scheduler = SpacedRepetitionScheduler(random_seed=42)

        due = scheduler.get_due_memories()
        assert due == []

        stats = scheduler.statistics()
        assert stats["total_memories"] == 0

    def test_controller_no_memories(self):
        """Controller with no memories should work correctly."""
        controller = MetaLearningController(random_seed=42)

        needs_replay = controller.get_memories_needing_replay()
        assert needs_replay == []

        stats = controller.statistics()
        assert stats["total_memories"] == 0

    def test_resolver_empty_detect(self):
        """Detecting interference on empty list should return empty."""
        resolver = InterferenceResolver(random_seed=42)

        events = resolver.detect_interference([])
        assert events == []

    def test_resolver_single_memory_detect(self):
        """Detecting interference with single memory returns empty."""
        resolver = InterferenceResolver(random_seed=42)
        resolver.register_memory("solo", np.random.randn(128), 0.0)

        events = resolver.detect_interference(["solo"])
        assert events == []

    def test_schema_no_instances_coverage(self):
        """Schema coverage with no instances should be 0."""
        refiner = SchemaRefiner(random_seed=42)
        refiner.create_schema("empty", np.random.randn(64), instances=[])

        coverage = refiner.compute_coverage("empty")
        assert coverage == 0.0

    def test_learning_curve_no_history(self):
        """Learning curve with no history should handle velocity/efficiency."""
        curve = LearningCurve(
            memory_id="test",
            memory_type=MemoryType.EPISODIC,
        )

        assert curve.get_learning_velocity() == 0.0
        assert curve.get_efficiency() == 0.0


class TestExtremeValues:
    """Tests for extreme value handling."""

    def test_scheduler_huge_interval(self):
        """System should handle memories not reviewed for a long time."""
        scheduler = SpacedRepetitionScheduler(random_seed=42)
        scheduler.register_memory("old")

        # Simulate 10 years passing
        scheduler.set_night(3650)

        schedule = scheduler.get_schedule("old")
        assert schedule.is_due(3650)

        days_overdue = schedule.days_until_due(3650)
        assert days_overdue == 0  # Already past due

    def test_scheduler_very_long_streak(self):
        """Long streaks should be tracked correctly."""
        scheduler = SpacedRepetitionScheduler(random_seed=42)
        scheduler.register_memory("mem1")

        for _ in range(1000):
            scheduler.schedule_review("mem1", ReviewQuality.PERFECT)

        streak = scheduler.get_schedule("mem1").streak
        assert streak == 1000

    def test_controller_many_memories(self):
        """Controller should handle many memories."""
        controller = MetaLearningController(random_seed=42)

        # Register 1000 memories
        for i in range(1000):
            controller.register_memory(f"mem{i}", MemoryType.EPISODIC, 0.1)

        stats = controller.statistics()
        assert stats["total_memories"] == 1000

        # Should still be able to query
        needs = controller.get_memories_needing_replay(top_k=10)
        assert len(needs) <= 10

    def test_resolver_high_dimension_vectors(self):
        """Resolver should handle high-dimensional vectors."""
        resolver = InterferenceResolver(random_seed=42)

        np.random.seed(42)
        vec1 = np.random.randn(10000)  # 10k dimensions
        vec2 = np.random.randn(10000)

        resolver.register_memory("large1", vec1, 0.0)
        resolver.register_memory("large2", vec2, 0.0)

        sim = resolver.compute_similarity("large1", "large2")
        assert np.isfinite(sim)

    def test_very_small_strength_values(self):
        """Controller should handle very small strength values."""
        controller = MetaLearningController(random_seed=42)
        controller.register_memory("tiny", MemoryType.EPISODIC, 1e-10)

        outcome = controller.track_consolidation_success("tiny", 1e-10, 1e-9, 1)
        assert np.isfinite(outcome.efficiency)


class TestNonexistentReferences:
    """Tests for operations on nonexistent items."""

    def test_scheduler_get_nonexistent(self):
        """Getting schedule for nonexistent memory returns None."""
        scheduler = SpacedRepetitionScheduler(random_seed=42)

        schedule = scheduler.get_schedule("nonexistent")
        assert schedule is None

    def test_controller_get_nonexistent_curve(self):
        """Getting learning curve for nonexistent memory returns None."""
        controller = MetaLearningController(random_seed=42)

        curve = controller.get_learning_curve("nonexistent")
        assert curve is None

    def test_resolver_similarity_nonexistent(self):
        """Computing similarity with nonexistent memory should handle gracefully."""
        resolver = InterferenceResolver(random_seed=42)
        resolver.register_memory("exists", np.random.randn(128), 0.0)

        # This should not crash - implementation may return 0 or raise
        try:
            sim = resolver.compute_similarity("exists", "nonexistent")
            assert sim == 0.0 or sim is None
        except KeyError:
            pass  # Also acceptable

    def test_schema_get_nonexistent(self):
        """Getting nonexistent schema returns None."""
        refiner = SchemaRefiner(random_seed=42)

        schema = refiner.get_schema("nonexistent")
        assert schema is None


class TestBoundaryConditions:
    """Tests for boundary conditions."""

    def test_scheduler_review_at_exact_due_time(self):
        """Review at exact due time should work."""
        scheduler = SpacedRepetitionScheduler(random_seed=42)
        scheduler.register_memory("mem1")
        scheduler.schedule_review("mem1", ReviewQuality.EASY_CORRECT)

        next_review = scheduler.get_schedule("mem1").next_review
        scheduler.set_night(next_review)  # Exact due time

        assert scheduler.get_schedule("mem1").is_due(next_review)

    def test_controller_strength_exactly_one(self):
        """Strength of exactly 1.0 should be handled."""
        controller = MetaLearningController(random_seed=42)
        controller.register_memory("perfect", MemoryType.EPISODIC, 1.0)

        # Should be considered consolidated
        curve = controller.get_learning_curve("perfect")
        replays = controller.predict_optimal_replays("perfect")
        # Already at max strength, might need 0 replays
        assert replays >= 0

    def test_controller_strength_exactly_zero(self):
        """Strength of exactly 0.0 should be handled."""
        controller = MetaLearningController(random_seed=42)
        controller.register_memory("blank", MemoryType.EPISODIC, 0.0)

        replays = controller.predict_optimal_replays("blank")
        assert replays > 0  # Should need replays

    def test_resolver_identical_memories(self):
        """Two identical memories should have similarity 1.0."""
        resolver = InterferenceResolver(random_seed=42)

        np.random.seed(42)
        vec = np.random.randn(128)

        resolver.register_memory("copy1", vec.copy(), 0.0)
        resolver.register_memory("copy2", vec.copy(), 0.0)

        sim = resolver.compute_similarity("copy1", "copy2")
        assert sim == pytest.approx(1.0, abs=0.001)

    def test_schema_dimension_mismatch(self):
        """Schema should handle or reject mismatched dimensions."""
        refiner = SchemaRefiner(embedding_dim=64, random_seed=42)

        prototype = np.random.randn(64)
        schema = refiner.create_schema("test", prototype, instances=[])

        # Wrong dimension vector
        wrong_dim = np.random.randn(128)
        result = schema.contains(wrong_dim)
        assert result is False  # Should not contain mismatched vector


class TestNumericalStability:
    """Tests for numerical stability."""

    def test_very_small_variance(self):
        """Schema with very small variance should not cause division issues."""
        refiner = SchemaRefiner(random_seed=42)

        prototype = np.random.randn(64)
        schema = Schema(
            name="tight",
            prototype=prototype,
            variance=np.ones(64) * 1e-10,  # Very small variance
            coverage=0.5,
            instances=[],
        )

        # Should not crash due to near-zero division
        result = schema.contains(prototype + 0.1 * np.random.randn(64))
        # numpy.bool_ is a valid bool-like type
        assert result is True or result is False or isinstance(result, (bool, np.bool_))

    def test_large_consolidation_values(self):
        """Controller should handle large consolidation values."""
        controller = MetaLearningController(random_seed=42)
        controller.register_memory("big", MemoryType.EPISODIC, 0.1)

        # Large replays value
        outcome = controller.track_consolidation_success("big", 0.1, 0.5, 1000000)

        assert np.isfinite(outcome.efficiency)

    def test_similarity_near_zero_vectors(self):
        """Near-zero vectors should not cause numerical issues."""
        resolver = InterferenceResolver(random_seed=42)

        resolver.register_memory("tiny1", np.ones(128) * 1e-100, 0.0)
        resolver.register_memory("tiny2", np.ones(128) * 1e-100, 0.0)

        sim = resolver.compute_similarity("tiny1", "tiny2")
        assert np.isfinite(sim) or sim == 0.0


class TestTypeConsistency:
    """Tests for type consistency across operations."""

    def test_scheduler_interval_always_float(self):
        """Interval should always be a float."""
        scheduler = SpacedRepetitionScheduler(random_seed=42)
        scheduler.register_memory("mem1")

        for q in ReviewQuality:
            scheduler.schedule_review("mem1", q)
            interval = scheduler.get_schedule("mem1").interval
            assert isinstance(interval, (int, float))
            assert interval > 0

    def test_controller_scores_are_floats(self):
        """All scores should be floats."""
        controller = MetaLearningController(random_seed=42)
        controller.register_memory("mem1", MemoryType.EPISODIC, 0.5)

        score = controller.compute_priority_score("mem1", 0.5, 0.5, 0.5, 0.5)
        assert isinstance(score, (int, float))

        rate = controller.get_learning_rate(MemoryType.EPISODIC)
        assert isinstance(rate, (int, float))

    def test_statistics_return_dicts(self):
        """All statistics methods should return dicts."""
        scheduler = SpacedRepetitionScheduler(random_seed=42)
        controller = MetaLearningController(random_seed=42)
        resolver = InterferenceResolver(random_seed=42)
        refiner = SchemaRefiner(random_seed=42)

        assert isinstance(scheduler.statistics(), dict)
        assert isinstance(controller.statistics(), dict)
        assert isinstance(resolver.statistics(), dict)
        assert isinstance(refiner.statistics(), dict)
