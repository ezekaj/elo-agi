"""Tests for spaced repetition scheduler."""

import pytest
from neuro.modules.m05_sleep_consolidation.spaced_repetition import (
    SpacedRepetitionScheduler,
    RepetitionSchedule,
    ReviewQuality,
)


class TestRepetitionSchedule:
    """Tests for RepetitionSchedule."""

    def test_initialization(self):
        """Test schedule initialization."""
        schedule = RepetitionSchedule(
            memory_id="mem1",
            next_review=1.0,
            interval=1.0,
            easiness=2.5,
            repetition_count=0,
            last_review=0.0,
        )
        assert schedule.memory_id == "mem1"
        assert schedule.interval == 1.0
        assert schedule.is_new()

    def test_is_due(self):
        """Test due checking."""
        schedule = RepetitionSchedule(
            memory_id="mem1",
            next_review=5.0,
            interval=1.0,
            easiness=2.5,
            repetition_count=0,
            last_review=4.0,
        )
        assert not schedule.is_due(4.0)
        assert schedule.is_due(5.0)
        assert schedule.is_due(6.0)

    def test_days_until_due(self):
        """Test days until due calculation."""
        schedule = RepetitionSchedule(
            memory_id="mem1",
            next_review=10.0,
            interval=1.0,
            easiness=2.5,
            repetition_count=0,
            last_review=9.0,
        )
        assert schedule.days_until_due(7.0) == 3.0
        assert schedule.days_until_due(10.0) == 0.0
        assert schedule.days_until_due(12.0) == 0.0

    def test_is_new(self):
        """Test new memory detection."""
        new_schedule = RepetitionSchedule(
            memory_id="mem1",
            next_review=0.0,
            interval=1.0,
            easiness=2.5,
            repetition_count=0,
            last_review=0.0,
        )
        assert new_schedule.is_new()

        reviewed_schedule = RepetitionSchedule(
            memory_id="mem2",
            next_review=1.0,
            interval=1.0,
            easiness=2.5,
            repetition_count=1,
            last_review=0.0,
        )
        assert not reviewed_schedule.is_new()


class TestSpacedRepetitionScheduler:
    """Tests for SpacedRepetitionScheduler."""

    @pytest.fixture
    def scheduler(self):
        return SpacedRepetitionScheduler(random_seed=42)

    def test_initialization(self, scheduler):
        """Test scheduler initialization."""
        assert scheduler.initial_interval == 1.0
        assert scheduler.initial_easiness == 2.5
        stats = scheduler.statistics()
        assert stats["total_memories"] == 0

    def test_register_memory(self, scheduler):
        """Test memory registration."""
        schedule = scheduler.register_memory("mem1")
        assert schedule.memory_id == "mem1"
        assert schedule.easiness == scheduler.initial_easiness

    def test_get_schedule(self, scheduler):
        """Test getting schedule."""
        scheduler.register_memory("mem1")
        schedule = scheduler.get_schedule("mem1")
        assert schedule is not None
        assert schedule.memory_id == "mem1"

    def test_advance_night(self, scheduler):
        """Test advancing night."""
        scheduler.advance_night(3.0)
        assert scheduler._current_night == 3.0

    def test_schedule_review_perfect(self, scheduler):
        """Test scheduling after perfect review."""
        scheduler.register_memory("mem1")
        schedule = scheduler.schedule_review("mem1", ReviewQuality.PERFECT)

        assert schedule.repetition_count == 1
        assert schedule.interval >= scheduler.graduating_interval
        assert schedule.streak == 1

    def test_schedule_review_failed(self, scheduler):
        """Test scheduling after failed review."""
        scheduler.register_memory("mem1")
        scheduler.schedule_review("mem1", ReviewQuality.EASY_CORRECT)
        scheduler.schedule_review("mem1", ReviewQuality.EASY_CORRECT)

        scheduler.get_schedule("mem1").repetition_count
        scheduler.schedule_review("mem1", ReviewQuality.COMPLETE_BLACKOUT)

        schedule = scheduler.get_schedule("mem1")
        assert schedule.repetition_count == 0  # Reset
        assert schedule.streak == 0
        # Interval is reset to ~initial (with small fuzz factor)
        assert schedule.interval == pytest.approx(scheduler.initial_interval, rel=0.1)

    def test_update_after_review(self, scheduler):
        """Test simplified update interface."""
        scheduler.register_memory("mem1")
        schedule = scheduler.update_after_review("mem1", success=True, confidence=0.8)

        assert schedule.repetition_count == 1

    def test_get_due_memories(self, scheduler):
        """Test getting due memories."""
        scheduler.register_memory("mem1")
        scheduler.register_memory("mem2")
        scheduler.register_memory("mem3")

        # All should be due at night 0
        due = scheduler.get_due_memories()
        assert len(due) == 3

        # Review mem1, it should no longer be due
        scheduler.schedule_review("mem1", ReviewQuality.EASY_CORRECT)
        due = scheduler.get_due_memories()
        assert "mem1" not in due

    def test_get_due_memories_limit(self, scheduler):
        """Test due memories with limit."""
        for i in range(10):
            scheduler.register_memory(f"mem{i}")

        due = scheduler.get_due_memories(limit=5)
        assert len(due) == 5

    def test_get_upcoming_reviews(self, scheduler):
        """Test upcoming reviews forecast."""
        scheduler.register_memory("mem1")
        scheduler.schedule_review("mem1", ReviewQuality.EASY_CORRECT)

        upcoming = scheduler.get_upcoming_reviews(days_ahead=7)
        assert len(upcoming) > 0

    def test_retention_estimate(self, scheduler):
        """Test retention estimate."""
        scheduler.register_memory("mem1")
        scheduler.schedule_review("mem1", ReviewQuality.EASY_CORRECT)

        # Right after review
        retention = scheduler.get_memory_retention_estimate("mem1")
        assert retention > 0.9

        # Days later
        scheduler.advance_night(10)
        retention_later = scheduler.get_memory_retention_estimate("mem1")
        assert retention_later < retention

    def test_export_import_schedule(self, scheduler):
        """Test exporting and importing schedules."""
        scheduler.register_memory("mem1")
        scheduler.schedule_review("mem1", ReviewQuality.EASY_CORRECT)

        exported = scheduler.export_schedule()
        assert "mem1" in exported

        new_scheduler = SpacedRepetitionScheduler()
        new_scheduler.import_schedule(exported)

        assert new_scheduler.get_schedule("mem1") is not None

    def test_workload_forecast(self, scheduler):
        """Test workload forecast."""
        for i in range(5):
            scheduler.register_memory(f"mem{i}")

        forecast = scheduler.get_workload_forecast(days=7)
        assert len(forecast) == 7
        assert forecast[0] == 5  # All due today

    def test_easiness_increases_with_good_reviews(self, scheduler):
        """Test easiness factor increases."""
        scheduler.register_memory("mem1")
        initial_easiness = scheduler.get_schedule("mem1").easiness

        for _ in range(3):
            scheduler.schedule_review("mem1", ReviewQuality.PERFECT)

        final_easiness = scheduler.get_schedule("mem1").easiness
        assert final_easiness >= initial_easiness

    def test_easiness_decreases_with_bad_reviews(self, scheduler):
        """Test easiness factor decreases."""
        scheduler.register_memory("mem1")
        scheduler.schedule_review("mem1", ReviewQuality.EASY_CORRECT)
        scheduler.schedule_review("mem1", ReviewQuality.EASY_CORRECT)

        mid_easiness = scheduler.get_schedule("mem1").easiness
        scheduler.schedule_review("mem1", ReviewQuality.INCORRECT)

        final_easiness = scheduler.get_schedule("mem1").easiness
        assert final_easiness < mid_easiness

    def test_statistics(self, scheduler):
        """Test statistics generation."""
        scheduler.register_memory("mem1")
        scheduler.schedule_review("mem1", ReviewQuality.EASY_CORRECT)

        stats = scheduler.statistics()
        assert stats["total_memories"] == 1
        assert stats["total_reviews"] == 1
        assert stats["successful_reviews"] == 1
        assert "workload_7day" in stats
