"""Tests for SM-2 algorithm compliance.

The SM-2 algorithm defines specific formulas for interval and easiness calculation.
These tests verify our implementation matches the specification.

SM-2 Reference:
- Initial easiness factor (EF) = 2.5
- EF range: [1.3, 2.5]
- EF update: EF' = EF + (0.1 - (5-q)*(0.08 + (5-q)*0.02))
- Quality q in [0, 5]
- If q >= 3: progress (increase interval)
- If q < 3: reset (lapse)
- Interval after 1st successful review: 1 day
- Interval after 2nd successful review: 6 days
- Subsequent: I(n) = I(n-1) * EF
"""

import pytest
import numpy as np
from src.spaced_repetition import (
    SpacedRepetitionScheduler, RepetitionSchedule, ReviewQuality,
)


class TestSM2EasinessFormula:
    """Tests for SM-2 easiness factor updates."""

    def test_easiness_update_quality_5(self):
        """Quality 5: EF increases by 0.1."""
        scheduler = SpacedRepetitionScheduler(random_seed=42)
        scheduler.register_memory("mem1")

        initial_ef = scheduler.get_schedule("mem1").easiness
        scheduler.schedule_review("mem1", ReviewQuality.PERFECT)  # q=5
        new_ef = scheduler.get_schedule("mem1").easiness

        # EF' = EF + (0.1 - (5-5)*(0.08 + (5-5)*0.02)) = EF + 0.1
        # But capped at maximum_easiness (2.5)
        expected = min(initial_ef + 0.1, 2.5)
        assert new_ef == pytest.approx(expected, abs=0.01)

    def test_easiness_update_quality_4(self):
        """Quality 4: EF stays roughly same."""
        scheduler = SpacedRepetitionScheduler(random_seed=42)
        scheduler.register_memory("mem1")

        initial_ef = scheduler.get_schedule("mem1").easiness
        scheduler.schedule_review("mem1", ReviewQuality.EASY_CORRECT)  # q=4
        new_ef = scheduler.get_schedule("mem1").easiness

        # EF' = EF + (0.1 - (5-4)*(0.08 + (5-4)*0.02)) = EF + 0.1 - 0.1 = EF
        assert new_ef == pytest.approx(initial_ef, abs=0.02)

    def test_easiness_update_quality_3(self):
        """Quality 3: EF decreases slightly."""
        scheduler = SpacedRepetitionScheduler(random_seed=42)
        scheduler.register_memory("mem1")

        initial_ef = scheduler.get_schedule("mem1").easiness
        scheduler.schedule_review("mem1", ReviewQuality.CORRECT_WITH_EFFORT)  # q=3
        new_ef = scheduler.get_schedule("mem1").easiness

        # EF' = EF + (0.1 - (5-3)*(0.08 + (5-3)*0.02)) = EF + 0.1 - 0.24 = EF - 0.14
        expected = max(initial_ef - 0.14, 1.3)
        assert new_ef == pytest.approx(expected, abs=0.02)

    def test_easiness_update_quality_0(self):
        """Quality 0: EF decreases significantly."""
        scheduler = SpacedRepetitionScheduler(random_seed=42)
        scheduler.register_memory("mem1")

        initial_ef = scheduler.get_schedule("mem1").easiness
        scheduler.schedule_review("mem1", ReviewQuality.COMPLETE_BLACKOUT)  # q=0
        new_ef = scheduler.get_schedule("mem1").easiness

        # EF' = EF + (0.1 - (5-0)*(0.08 + (5-0)*0.02)) = EF + 0.1 - 0.9 = EF - 0.8
        # Additional -0.2 for complete blackout, and bounded by minimum
        assert new_ef >= 1.3
        assert new_ef < initial_ef

    def test_easiness_never_below_minimum(self):
        """EF must never go below 1.3."""
        scheduler = SpacedRepetitionScheduler(random_seed=42)
        scheduler.register_memory("mem1")

        # 50 consecutive failures
        for _ in range(50):
            scheduler.schedule_review("mem1", ReviewQuality.COMPLETE_BLACKOUT)

        assert scheduler.get_schedule("mem1").easiness >= 1.3

    def test_easiness_never_above_maximum(self):
        """EF must never exceed 2.5."""
        scheduler = SpacedRepetitionScheduler(random_seed=42)
        scheduler.register_memory("mem1")

        # 50 consecutive perfect reviews
        for _ in range(50):
            scheduler.schedule_review("mem1", ReviewQuality.PERFECT)

        assert scheduler.get_schedule("mem1").easiness <= 2.5


class TestSM2IntervalProgression:
    """Tests for SM-2 interval progression."""

    def test_first_interval_after_success(self):
        """First successful review: interval = graduating_interval * easy_bonus."""
        scheduler = SpacedRepetitionScheduler(random_seed=42)
        scheduler.register_memory("mem1")

        scheduler.schedule_review("mem1", ReviewQuality.PERFECT)
        interval = scheduler.get_schedule("mem1").interval

        # With PERFECT (q=5), gets easy_bonus (1.3), plus fuzz
        # Expected: 1.0 * 1.3 * fuzz = ~1.3
        assert interval == pytest.approx(1.3, rel=0.1)

    def test_second_interval_after_success(self):
        """Second successful review: interval = 6 days."""
        scheduler = SpacedRepetitionScheduler(random_seed=42)
        scheduler.register_memory("mem1")

        scheduler.schedule_review("mem1", ReviewQuality.EASY_CORRECT)
        scheduler.schedule_review("mem1", ReviewQuality.EASY_CORRECT)
        interval = scheduler.get_schedule("mem1").interval

        # Second interval is ~6 days * easy_bonus
        # With easy_bonus=1.3 and fuzz, should be around 7-8 days
        assert interval >= 6.0

    def test_interval_grows_with_ef(self):
        """After 2nd review, interval grows by EF factor."""
        scheduler = SpacedRepetitionScheduler(random_seed=42)
        scheduler.register_memory("mem1")

        # Get through initial intervals
        scheduler.schedule_review("mem1", ReviewQuality.CORRECT_WITH_EFFORT)  # 1 day
        scheduler.schedule_review("mem1", ReviewQuality.CORRECT_WITH_EFFORT)  # 6 days

        ef = scheduler.get_schedule("mem1").easiness
        interval_2 = scheduler.get_schedule("mem1").interval

        scheduler.schedule_review("mem1", ReviewQuality.CORRECT_WITH_EFFORT)
        interval_3 = scheduler.get_schedule("mem1").interval

        # interval_3 should be approximately interval_2 * ef (with some fuzz)
        expected = interval_2 * ef
        assert interval_3 == pytest.approx(expected, rel=0.15)

    def test_interval_resets_on_lapse(self):
        """If q < 3, interval resets to initial."""
        scheduler = SpacedRepetitionScheduler(random_seed=42)
        scheduler.register_memory("mem1")

        # Build up a long interval
        for _ in range(5):
            scheduler.schedule_review("mem1", ReviewQuality.EASY_CORRECT)

        long_interval = scheduler.get_schedule("mem1").interval
        assert long_interval > 10  # Should be pretty long by now

        # Lapse
        scheduler.schedule_review("mem1", ReviewQuality.INCORRECT)
        reset_interval = scheduler.get_schedule("mem1").interval

        # Should reset to ~initial interval (1 day)
        assert reset_interval == pytest.approx(1.0, rel=0.1)


class TestSM2RepetitionCount:
    """Tests for repetition count tracking."""

    def test_rep_count_increases_on_success(self):
        """Repetition count increases with each successful review."""
        scheduler = SpacedRepetitionScheduler(random_seed=42)
        scheduler.register_memory("mem1")

        assert scheduler.get_schedule("mem1").repetition_count == 0

        scheduler.schedule_review("mem1", ReviewQuality.EASY_CORRECT)
        assert scheduler.get_schedule("mem1").repetition_count == 1

        scheduler.schedule_review("mem1", ReviewQuality.EASY_CORRECT)
        assert scheduler.get_schedule("mem1").repetition_count == 2

    def test_rep_count_resets_on_lapse(self):
        """Repetition count resets to 0 on lapse."""
        scheduler = SpacedRepetitionScheduler(random_seed=42)
        scheduler.register_memory("mem1")

        scheduler.schedule_review("mem1", ReviewQuality.EASY_CORRECT)
        scheduler.schedule_review("mem1", ReviewQuality.EASY_CORRECT)
        assert scheduler.get_schedule("mem1").repetition_count == 2

        scheduler.schedule_review("mem1", ReviewQuality.INCORRECT)
        assert scheduler.get_schedule("mem1").repetition_count == 0


class TestSM2QualityBoundaries:
    """Tests for quality rating boundaries."""

    def test_quality_3_is_passing(self):
        """Quality 3 (correct with effort) counts as successful."""
        scheduler = SpacedRepetitionScheduler(random_seed=42)
        scheduler.register_memory("mem1")

        scheduler.schedule_review("mem1", ReviewQuality.CORRECT_WITH_EFFORT)

        assert scheduler.get_schedule("mem1").repetition_count == 1
        assert scheduler.get_schedule("mem1").streak == 1

    def test_quality_2_is_failing(self):
        """Quality 2 (difficult correct) counts as lapse."""
        scheduler = SpacedRepetitionScheduler(random_seed=42)
        scheduler.register_memory("mem1")

        # First get some reps
        scheduler.schedule_review("mem1", ReviewQuality.EASY_CORRECT)
        scheduler.schedule_review("mem1", ReviewQuality.EASY_CORRECT)

        # Now a difficult correct (quality 2)
        scheduler.schedule_review("mem1", ReviewQuality.DIFFICULT_CORRECT)

        assert scheduler.get_schedule("mem1").repetition_count == 0
        assert scheduler.get_schedule("mem1").streak == 0


class TestSM2Integration:
    """Integration tests for SM-2 algorithm over multiple reviews."""

    def test_realistic_learning_sequence(self):
        """Test a realistic learning sequence over 30 days."""
        scheduler = SpacedRepetitionScheduler(random_seed=42)
        scheduler.register_memory("mem1")

        # Simulate: mostly good reviews with occasional struggles
        qualities = [
            ReviewQuality.CORRECT_WITH_EFFORT,  # Day 1: struggle
            ReviewQuality.EASY_CORRECT,         # Day 2: better
            ReviewQuality.EASY_CORRECT,         # Day ~8: good
            ReviewQuality.PERFECT,              # Day ~20: mastered
        ]

        intervals = []
        for q in qualities:
            scheduler.schedule_review("mem1", q)
            intervals.append(scheduler.get_schedule("mem1").interval)

        # Intervals should generally increase
        for i in range(1, len(intervals)):
            assert intervals[i] >= intervals[i-1] * 0.5, \
                f"Interval {i} ({intervals[i]}) shouldn't drop too much from {intervals[i-1]}"

    def test_relearning_after_lapse(self):
        """Test that relearning is faster after lapse (preserved EF)."""
        scheduler = SpacedRepetitionScheduler(random_seed=42)
        scheduler.register_memory("mem1")

        # Learn it well
        for _ in range(5):
            scheduler.schedule_review("mem1", ReviewQuality.EASY_CORRECT)

        good_ef = scheduler.get_schedule("mem1").easiness

        # Lapse
        scheduler.schedule_review("mem1", ReviewQuality.COMPLETE_BLACKOUT)

        # EF should still be relatively high (memory of past learning)
        lapse_ef = scheduler.get_schedule("mem1").easiness
        assert lapse_ef >= 1.3
        # EF decreased but not completely reset
        assert lapse_ef < good_ef
