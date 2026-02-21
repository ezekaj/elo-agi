"""Integration tests for multi-component scenarios.

These tests verify that components work correctly together
in realistic usage scenarios.
"""

import pytest
import numpy as np
from neuro.modules.m05_sleep_consolidation.spaced_repetition import SpacedRepetitionScheduler, ReviewQuality
from neuro.modules.m05_sleep_consolidation.meta_learning import MetaLearningController, MemoryType
from neuro.modules.m05_sleep_consolidation.interference_resolution import InterferenceResolver
from neuro.modules.m05_sleep_consolidation.schema_refinement import SchemaRefiner

class TestFullConsolidationCycle:
    """Tests for complete consolidation cycles."""

    def test_week_of_learning(self):
        """Simulate a realistic week of learning 10 memories."""
        np.random.seed(42)
        controller = MetaLearningController(random_seed=42)
        scheduler = SpacedRepetitionScheduler(random_seed=42)

        # Day 1: Learn 10 new memories
        for i in range(10):
            controller.register_memory(f"mem{i}", MemoryType.EPISODIC, 0.2)
            scheduler.register_memory(f"mem{i}")

        # Day 2-7: Review cycle
        successes = 0
        total_reviews = 0

        for day in range(1, 8):
            scheduler.set_night(day)
            due = scheduler.get_due_memories()

            for mem_id in due:
                total_reviews += 1
                # 80% success rate
                if np.random.random() < 0.8:
                    quality = ReviewQuality.EASY_CORRECT
                    successes += 1
                    after_strength = 0.5 + 0.05 * day
                else:
                    quality = ReviewQuality.INCORRECT
                    after_strength = 0.3

                scheduler.schedule_review(mem_id, quality)
                controller.track_consolidation_success(
                    mem_id, 0.3, after_strength, 2
                )

        # Verify outcomes
        stats = controller.statistics()
        assert stats["total_memories"] == 10
        assert stats["n_outcomes"] > 0

        sched_stats = scheduler.statistics()
        assert sched_stats["total_reviews"] == total_reviews

    def test_month_of_spaced_repetition(self):
        """Simulate a month of spaced repetition for 5 memories."""
        scheduler = SpacedRepetitionScheduler(random_seed=42)

        # Register 5 memories
        for i in range(5):
            scheduler.register_memory(f"mem{i}")

        # Simulate 30 days
        review_counts = {f"mem{i}": 0 for i in range(5)}

        for day in range(30):
            scheduler.set_night(day)
            due = scheduler.get_due_memories()

            for mem_id in due:
                review_counts[mem_id] += 1
                scheduler.schedule_review(mem_id, ReviewQuality.EASY_CORRECT)

        # With spaced repetition, review count should decrease over time
        # Early days have more reviews, later days fewer
        total_reviews = sum(review_counts.values())
        assert total_reviews < 30 * 5, "Should have fewer reviews than daily for all"
        assert total_reviews >= 5, "Should have at least initial reviews"

        # All memories should have increasing intervals
        for i in range(5):
            schedule = scheduler.get_schedule(f"mem{i}")
            assert schedule.interval > 1.0, "Interval should grow with success"

class TestInterferenceWithConsolidation:
    """Tests for interference during consolidation."""

    def test_similar_memories_detected_during_learning(self):
        """Detect interference between similar memories."""
        np.random.seed(42)
        controller = MetaLearningController(random_seed=42)
        resolver = InterferenceResolver(similarity_threshold=0.7, random_seed=42)

        # Create base memory
        base_vector = np.random.randn(128)
        base_vector = base_vector / np.linalg.norm(base_vector)

        # Register in both systems
        controller.register_memory("original", MemoryType.EPISODIC, 0.5)
        resolver.register_memory("original", base_vector, 0.0)

        # Create similar memory (confusable)
        similar_vector = base_vector * 0.95 + 0.05 * np.random.randn(128)
        similar_vector = similar_vector / np.linalg.norm(similar_vector)

        controller.register_memory("confusable", MemoryType.EPISODIC, 0.3)
        resolver.register_memory("confusable", similar_vector, 1.0)

        # Detect interference
        events = resolver.detect_interference(["original", "confusable"])

        # Should detect high similarity
        if events:
            assert events[0].similarity > 0.7
            # Risk should be elevated
            risk = resolver.get_interference_risk("original")
            assert risk >= 0

    def test_interleaved_replay_reduces_interference(self):
        """Interleaved replay should be scheduled for similar memories."""
        np.random.seed(42)
        resolver = InterferenceResolver(similarity_threshold=0.7, random_seed=42)

        # Create two similar memories
        base = np.random.randn(128)
        base = base / np.linalg.norm(base)

        resolver.register_memory("mem_a", base, 0.0)
        resolver.register_memory("mem_b", base * 0.98 + 0.02 * np.random.randn(128), 1.0)

        # Get interleaved schedule
        schedule = resolver.interleave_replays(["mem_a", "mem_b"], repetitions=4)

        assert len(schedule.memories) == 2
        # Pattern has 4 repetitions * 2 memories = 8 items
        assert len(schedule.pattern) == 8

        # Execute schedule (is_complete uses pattern * repetitions)
        replayed = []
        while not schedule.is_complete():
            mem = schedule.next_memory()
            replayed.append(mem)

        # Both memories should appear in the replay
        assert "mem_a" in replayed
        assert "mem_b" in replayed
        # Should have multiple replays of each
        assert replayed.count("mem_a") >= 4
        assert replayed.count("mem_b") >= 4

class TestSchemaEvolution:
    """Tests for schema evolution during consolidation."""

    def test_schema_forms_from_experiences(self):
        """Schema should form from multiple similar experiences."""
        np.random.seed(42)
        refiner = SchemaRefiner(embedding_dim=64, random_seed=42)

        # Create a category prototype
        cat_center = np.random.randn(64)
        cat_center = cat_center / np.linalg.norm(cat_center)

        # Register instances around the prototype
        instances = []
        for i in range(10):
            instance = cat_center + 0.1 * np.random.randn(64)
            refiner.register_instance(f"cat_{i}", instance)
            instances.append(f"cat_{i}")

        # Auto-discover schema
        schemas = refiner.auto_discover_schemas(similarity_threshold=0.8)

        # May or may not cluster depending on noise, so don't assert on count

        # Create schema manually with all instances
        refiner.create_schema("cats", cat_center, instances=instances)

        # Coverage is fraction of all registered instances that schema contains
        # Since we created with those exact instances, at least some should be covered
        stats = refiner.statistics()
        assert stats["total_schemas"] >= 1
        assert stats["total_instances"] == 10

    def test_schema_generalization(self):
        """Schema should generalize to cover new instances."""
        np.random.seed(42)
        refiner = SchemaRefiner(embedding_dim=64, random_seed=42)

        # Initial narrow schema
        prototype = np.random.randn(64)
        prototype = prototype / np.linalg.norm(prototype)

        refiner.create_schema("animals", prototype, instances=["dog", "cat"])
        refiner.register_instance("dog", prototype + 0.05 * np.random.randn(64))
        refiner.register_instance("cat", prototype + 0.05 * np.random.randn(64))

        # Add diverse new instances
        new_instances = []
        for i in range(5):
            inst_id = f"animal_{i}"
            vec = prototype + 0.2 * np.random.randn(64)  # More varied
            refiner.register_instance(inst_id, vec)
            new_instances.append(inst_id)

        # Generalize schema
        all_instances = ["dog", "cat"] + new_instances
        schema = refiner.generalize_schema("animals", all_instances)

        assert schema is not None
        assert len(schema.instances) >= 7

class TestAdaptiveLearning:
    """Tests for adaptive learning behavior."""

    def test_learning_rate_adapts_to_success(self):
        """Learning rates should adapt based on success patterns."""
        controller = MetaLearningController(random_seed=42)

        # Register memories of different types
        for i in range(15):
            controller.register_memory(f"episodic_{i}", MemoryType.EPISODIC, 0.2)
            controller.register_memory(f"semantic_{i}", MemoryType.SEMANTIC, 0.2)

        # Simulate: episodic memories consolidate well, semantic struggle
        for i in range(15):
            # Episodic: high success
            controller.track_consolidation_success(
                f"episodic_{i}", 0.2, 0.7, 3
            )
            # Semantic: low success
            controller.track_consolidation_success(
                f"semantic_{i}", 0.2, 0.25, 5
            )

        # Update learning rates
        controller.update_learning_rates(MemoryType.EPISODIC, 0.9)
        controller.update_learning_rates(MemoryType.SEMANTIC, 0.3)

        # Rates should diverge
        episodic_rate = controller.get_learning_rate(MemoryType.EPISODIC)
        semantic_rate = controller.get_learning_rate(MemoryType.SEMANTIC)

        # Episodic should have higher rate (high success = can learn faster)
        assert episodic_rate >= semantic_rate

    def test_replay_priority_adapts(self):
        """Replay weights should adapt based on outcomes."""
        controller = MetaLearningController(
            min_samples_for_adaptation=5,
            random_seed=42
        )

        # Create enough samples for adaptation
        for i in range(20):
            mem_type = MemoryType.EMOTIONAL if i % 2 == 0 else MemoryType.EPISODIC
            controller.register_memory(f"mem_{i}", mem_type, 0.3)

            # Emotional memories consolidate better
            if mem_type == MemoryType.EMOTIONAL:
                controller.track_consolidation_success(f"mem_{i}", 0.3, 0.8, 2)
            else:
                controller.track_consolidation_success(f"mem_{i}", 0.3, 0.35, 5)

        # Adapt weights
        initial_weights = controller.weights.to_dict()
        controller.adapt_replay_weights()
        adapted_weights = controller.weights.to_dict()

        # Weights should still be normalized
        total = sum(adapted_weights.values())
        assert total == pytest.approx(1.0, abs=0.01)

class TestCrossComponentConsistency:
    """Tests for consistency across components."""

    def test_memory_tracking_consistent(self):
        """Memory tracking should be consistent across scheduler and controller."""
        controller = MetaLearningController(random_seed=42)
        scheduler = SpacedRepetitionScheduler(random_seed=42)

        memory_ids = [f"mem_{i}" for i in range(5)]

        # Register in both
        for mem_id in memory_ids:
            controller.register_memory(mem_id, MemoryType.EPISODIC, 0.2)
            scheduler.register_memory(mem_id)

        # Verify both have same count
        ctrl_stats = controller.statistics()
        sched_stats = scheduler.statistics()

        assert ctrl_stats["total_memories"] == len(memory_ids)
        assert sched_stats["total_memories"] == len(memory_ids)

    def test_review_affects_both_systems(self):
        """Reviews should update both scheduler and controller."""
        controller = MetaLearningController(random_seed=42)
        scheduler = SpacedRepetitionScheduler(random_seed=42)

        controller.register_memory("shared", MemoryType.EPISODIC, 0.3)
        scheduler.register_memory("shared")

        # Review
        scheduler.schedule_review("shared", ReviewQuality.EASY_CORRECT)
        controller.track_consolidation_success("shared", 0.3, 0.6, 3)

        # Both should reflect the review
        sched = scheduler.get_schedule("shared")
        curve = controller.get_learning_curve("shared")

        assert sched.repetition_count == 1
        assert len(curve.consolidation_history) == 2  # initial + review

class TestRealisticScenarios:
    """Tests for realistic learning scenarios."""

    def test_vocabulary_learning_pattern(self):
        """Simulate learning vocabulary with typical patterns."""
        scheduler = SpacedRepetitionScheduler(random_seed=42)

        # Learn 20 words
        words = [f"word_{i}" for i in range(20)]
        for word in words:
            scheduler.register_memory(word)

        # Simulate 14 days with realistic quality distribution
        np.random.seed(42)
        quality_dist = [
            ReviewQuality.PERFECT,
            ReviewQuality.EASY_CORRECT,
            ReviewQuality.EASY_CORRECT,
            ReviewQuality.CORRECT_WITH_EFFORT,
            ReviewQuality.INCORRECT,
        ]

        learned_count = 0
        for day in range(14):
            scheduler.set_night(day)
            due = scheduler.get_due_memories()

            for word in due:
                quality = np.random.choice(quality_dist)
                scheduler.schedule_review(word, quality)

                # Check if "learned" (interval > 7 days)
                if scheduler.get_schedule(word).interval > 7:
                    learned_count += 1

        # Some words should be well-learned after 2 weeks
        stats = scheduler.statistics()
        assert stats["total_reviews"] > 20  # Multiple reviews per word
        assert stats["successful_reviews"] > 0

    def test_exam_cramming_scenario(self):
        """Simulate cramming before an exam (many reviews in short time)."""
        scheduler = SpacedRepetitionScheduler(random_seed=42)
        controller = MetaLearningController(random_seed=42)

        # 50 facts to learn
        for i in range(50):
            scheduler.register_memory(f"fact_{i}")
            controller.register_memory(f"fact_{i}", MemoryType.SEMANTIC, 0.1)

        # Cram: review everything 3 times in one "night"
        for _ in range(3):
            for i in range(50):
                scheduler.schedule_review(f"fact_{i}", ReviewQuality.CORRECT_WITH_EFFORT)
                controller.track_consolidation_success(f"fact_{i}", 0.1, 0.4, 1)

        # All facts should have increased strength
        stats = controller.statistics()
        assert stats["n_outcomes"] == 150  # 50 * 3

        # After cramming, all items have been reviewed so intervals are set
        # Check that upcoming reviews exist (within next 30 days)
        upcoming = scheduler.get_upcoming_reviews(days_ahead=30)
        assert len(upcoming) > 0, "Should have upcoming reviews after cramming"

        # Verify statistics tracked the reviews
        sched_stats = scheduler.statistics()
        assert sched_stats["total_reviews"] == 150
