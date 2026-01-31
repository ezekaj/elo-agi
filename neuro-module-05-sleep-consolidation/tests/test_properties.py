"""Property-based tests for invariants and correctness.

These tests verify mathematical properties that must always hold,
regardless of the specific inputs or sequence of operations.
"""

import pytest
import numpy as np
from src.spaced_repetition import SpacedRepetitionScheduler, ReviewQuality
from src.meta_learning import MetaLearningController, MemoryType, ReplayWeights
from src.interference_resolution import InterferenceResolver
from src.schema_refinement import SchemaRefiner


class TestSpacedRepetitionInvariants:
    """Property tests for spaced repetition."""

    def test_interval_monotonic_with_perfect_reviews(self):
        """Intervals must grow monotonically with consecutive perfect reviews (until cap)."""
        scheduler = SpacedRepetitionScheduler(random_seed=42)
        scheduler.register_memory("mem1")

        prev_interval = 0
        for i in range(15):
            scheduler.schedule_review("mem1", ReviewQuality.PERFECT)
            curr = scheduler.get_schedule("mem1").interval

            # Intervals grow until hitting max_interval cap (365 days)
            if prev_interval < scheduler.maximum_interval:
                assert curr >= prev_interval, \
                    f"Iteration {i}: interval {curr} should not decrease from {prev_interval}"
            prev_interval = curr

        # Should hit the cap
        assert scheduler.get_schedule("mem1").interval == scheduler.maximum_interval

    def test_interval_bounded_above(self):
        """Intervals should not grow beyond reasonable limits."""
        scheduler = SpacedRepetitionScheduler(random_seed=42)
        scheduler.register_memory("mem1")

        # 100 perfect reviews
        for _ in range(100):
            scheduler.schedule_review("mem1", ReviewQuality.PERFECT)

        interval = scheduler.get_schedule("mem1").interval
        # Should not exceed ~10 years (3650 days) even with perfect history
        assert interval < 3650, f"Interval {interval} is unreasonably large"

    def test_easiness_bounded(self):
        """Easiness factor must stay within SM-2 bounds [1.3, 2.5]."""
        scheduler = SpacedRepetitionScheduler(random_seed=42)
        scheduler.register_memory("mem1")

        # Mix of all quality levels
        qualities = [
            ReviewQuality.COMPLETE_BLACKOUT,
            ReviewQuality.PERFECT,
            ReviewQuality.INCORRECT,
            ReviewQuality.EASY_CORRECT,
            ReviewQuality.DIFFICULT_CORRECT,
        ]

        for _ in range(50):
            for q in qualities:
                scheduler.schedule_review("mem1", q)
                ef = scheduler.get_schedule("mem1").easiness
                assert 1.3 <= ef <= 2.5, f"Easiness {ef} out of bounds"

    def test_next_review_always_future(self):
        """Next review date must always be in the future."""
        scheduler = SpacedRepetitionScheduler(random_seed=42)
        scheduler.register_memory("mem1")

        for night in range(20):
            scheduler.set_night(night)
            scheduler.schedule_review("mem1", ReviewQuality.EASY_CORRECT)
            next_review = scheduler.get_schedule("mem1").next_review
            assert next_review > night, \
                f"Next review {next_review} not after current night {night}"

    def test_streak_consistency(self):
        """Streak must match consecutive successes."""
        scheduler = SpacedRepetitionScheduler(random_seed=42)
        scheduler.register_memory("mem1")

        # Build streak
        for i in range(5):
            scheduler.schedule_review("mem1", ReviewQuality.EASY_CORRECT)
            assert scheduler.get_schedule("mem1").streak == i + 1

        # Break streak
        scheduler.schedule_review("mem1", ReviewQuality.INCORRECT)
        assert scheduler.get_schedule("mem1").streak == 0

        # Build again
        scheduler.schedule_review("mem1", ReviewQuality.EASY_CORRECT)
        assert scheduler.get_schedule("mem1").streak == 1

    def test_total_reviews_always_increases(self):
        """Total reviews counter must always increase."""
        scheduler = SpacedRepetitionScheduler(random_seed=42)
        scheduler.register_memory("mem1")

        prev_total = 0
        for _ in range(20):
            scheduler.schedule_review("mem1", ReviewQuality.EASY_CORRECT)
            curr_total = scheduler.get_schedule("mem1").total_reviews
            assert curr_total > prev_total
            prev_total = curr_total


class TestMetaLearningInvariants:
    """Property tests for meta-learning controller."""

    def test_weights_always_normalized(self):
        """Replay weights must always sum to 1."""
        controller = MetaLearningController(random_seed=42)

        # Check initial
        w = controller.weights
        total = w.recency + w.emotional_salience + w.incompleteness + w.interference_risk
        assert total == pytest.approx(1.0, abs=0.01)

        # After many operations
        for i in range(30):
            controller.register_memory(f"mem{i}", MemoryType.EPISODIC, 0.2)
            controller.track_consolidation_success(f"mem{i}", 0.2, 0.5 + 0.01*i, 3)

        controller.adapt_replay_weights()
        w = controller.weights
        total = w.recency + w.emotional_salience + w.incompleteness + w.interference_risk
        assert total == pytest.approx(1.0, abs=0.01)

    def test_priority_score_bounded(self):
        """Priority score must be in [0, 1]."""
        controller = MetaLearningController(random_seed=42)
        controller.register_memory("mem1", MemoryType.EPISODIC)

        # Test various score combinations
        test_cases = [
            (0.0, 0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0, 1.0),
            (0.5, 0.5, 0.5, 0.5),
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
        ]

        for recency, emotional, incomp, interf in test_cases:
            score = controller.compute_priority_score(
                "mem1", recency, emotional, incomp, interf
            )
            assert 0 <= score <= 1, f"Score {score} out of bounds for inputs {(recency, emotional, incomp, interf)}"

    def test_consolidation_history_grows(self):
        """Consolidation history must grow with each tracking."""
        controller = MetaLearningController(random_seed=42)
        controller.register_memory("mem1", MemoryType.EPISODIC, 0.1)

        for i in range(10):
            controller.track_consolidation_success("mem1", 0.1 + 0.05*i, 0.15 + 0.05*i, 2)
            curve = controller.get_learning_curve("mem1")
            assert len(curve.consolidation_history) == i + 2  # +1 for initial, +1 for this

    def test_learning_rate_bounded(self):
        """Learning rates must stay in reasonable bounds."""
        controller = MetaLearningController(random_seed=42)

        # Extreme success rates
        for success_rate in [0.0, 0.5, 1.0]:
            for mem_type in MemoryType:
                rate = controller.update_learning_rates(mem_type, success_rate)
                assert 0.01 <= rate <= 0.5, f"Rate {rate} out of bounds"


class TestInterferenceInvariants:
    """Property tests for interference resolution."""

    def test_similarity_symmetric(self):
        """Similarity must be symmetric: sim(A,B) = sim(B,A)."""
        resolver = InterferenceResolver(random_seed=42)

        np.random.seed(42)
        vec1 = np.random.randn(128)
        vec2 = np.random.randn(128)

        resolver.register_memory("mem1", vec1, 0.0)
        resolver.register_memory("mem2", vec2, 0.0)

        sim_12 = resolver.compute_similarity("mem1", "mem2")
        sim_21 = resolver.compute_similarity("mem2", "mem1")

        assert sim_12 == pytest.approx(sim_21, abs=1e-10)

    def test_self_similarity_is_one(self):
        """Similarity of a vector with itself must be 1."""
        resolver = InterferenceResolver(random_seed=42)

        np.random.seed(42)
        vec = np.random.randn(128)
        vec = vec / np.linalg.norm(vec)  # Normalize

        resolver.register_memory("mem1", vec, 0.0)

        sim = resolver.compute_similarity("mem1", "mem1")
        assert sim == pytest.approx(1.0, abs=0.01)

    def test_similarity_bounded(self):
        """Similarity must be in [-1, 1]."""
        resolver = InterferenceResolver(random_seed=42)

        np.random.seed(42)
        for i in range(10):
            vec = np.random.randn(128)
            resolver.register_memory(f"mem{i}", vec, 0.0)

        for i in range(10):
            for j in range(i+1, 10):
                sim = resolver.compute_similarity(f"mem{i}", f"mem{j}")
                assert -1 <= sim <= 1, f"Similarity {sim} out of bounds"

    def test_interference_events_have_valid_similarity(self):
        """All detected interference events must have similarity above threshold."""
        resolver = InterferenceResolver(similarity_threshold=0.7, random_seed=42)

        # Create memories with known similarities
        np.random.seed(42)
        base = np.random.randn(128)
        base = base / np.linalg.norm(base)

        resolver.register_memory("base", base, 0.0)
        for i in range(5):
            vec = base + 0.1 * np.random.randn(128)
            resolver.register_memory(f"similar{i}", vec, float(i+1))

        events = resolver.detect_interference(["base"] + [f"similar{i}" for i in range(5)])

        for event in events:
            assert event.similarity >= resolver.similarity_threshold, \
                f"Event similarity {event.similarity} below threshold"


class TestSchemaInvariants:
    """Property tests for schema refinement."""

    def test_coverage_bounded(self):
        """Coverage must be in [0, 1]."""
        refiner = SchemaRefiner(random_seed=42)

        np.random.seed(42)
        for i in range(20):
            refiner.register_instance(f"inst{i}", np.random.randn(256))

        prototype = np.random.randn(256)
        refiner.create_schema("test", prototype, instances=["inst0", "inst1"])

        coverage = refiner.compute_coverage("test")
        assert 0 <= coverage <= 1, f"Coverage {coverage} out of bounds"

    def test_prototype_similarity_bounded(self):
        """Prototype similarity must be in [-1, 1]."""
        refiner = SchemaRefiner(random_seed=42)

        np.random.seed(42)
        prototype = np.random.randn(256)
        schema = refiner.create_schema("test", prototype, instances=[])

        for _ in range(10):
            test_vec = np.random.randn(256)
            sim = schema.similarity_to(test_vec)
            assert -1 <= sim <= 1, f"Similarity {sim} out of bounds"

    def test_schema_instances_tracked(self):
        """All instances added to schema must be in instance list."""
        refiner = SchemaRefiner(embedding_dim=64, random_seed=42)

        np.random.seed(42)
        prototype = np.random.randn(64)
        refiner.create_schema("test", prototype, instances=[], timestamp=0.0)

        added = []
        for i in range(10):
            inst_id = f"inst{i}"
            vec = np.random.randn(64)
            refiner.update_schema("test", inst_id, vec, timestamp=float(i))
            added.append(inst_id)

        schema = refiner.get_schema("test")
        for inst_id in added:
            assert inst_id in schema.instances, f"{inst_id} not in schema instances"


class TestDeterminism:
    """Tests for deterministic behavior with seeds."""

    def test_scheduler_deterministic(self):
        """Same seed produces identical scheduler behavior."""
        def run_scheduler(seed):
            scheduler = SpacedRepetitionScheduler(random_seed=seed)
            scheduler.register_memory("mem1")
            for q in [ReviewQuality.EASY_CORRECT, ReviewQuality.PERFECT, ReviewQuality.INCORRECT]:
                scheduler.schedule_review("mem1", q)
            return (
                scheduler.get_schedule("mem1").interval,
                scheduler.get_schedule("mem1").easiness,
            )

        result1 = run_scheduler(42)
        result2 = run_scheduler(42)
        result3 = run_scheduler(99)

        assert result1 == result2, "Same seed should produce same results"
        # Different seeds might produce same or different results

    def test_controller_deterministic(self):
        """Same seed produces identical controller behavior."""
        def run_controller(seed):
            controller = MetaLearningController(random_seed=seed)
            controller.register_memory("mem1", MemoryType.EPISODIC, 0.3)
            controller.track_consolidation_success("mem1", 0.3, 0.6, 5)
            return controller.statistics()

        result1 = run_controller(42)
        result2 = run_controller(42)

        assert result1 == result2, "Same seed should produce same statistics"

    def test_resolver_deterministic(self):
        """Same seed produces identical resolver behavior."""
        def run_resolver(seed):
            np.random.seed(seed)
            resolver = InterferenceResolver(random_seed=seed)
            vec1 = np.random.randn(128)
            vec2 = np.random.randn(128)
            resolver.register_memory("mem1", vec1, 0.0)
            resolver.register_memory("mem2", vec2, 1.0)
            return resolver.compute_similarity("mem1", "mem2")

        result1 = run_resolver(42)
        result2 = run_resolver(42)

        assert result1 == result2, "Same seed should produce same similarity"
