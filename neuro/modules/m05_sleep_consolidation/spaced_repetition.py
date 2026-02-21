"""
Spaced Repetition Scheduler for Multi-Night Memory Consolidation.

Implements SM-2 style spaced repetition algorithm adapted for
sleep-based memory consolidation across multiple nights.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np


class ReviewQuality(Enum):
    """Quality of memory recall during review."""
    COMPLETE_BLACKOUT = 0      # No recall at all
    INCORRECT = 1              # Incorrect recall
    DIFFICULT_CORRECT = 2      # Correct with difficulty
    CORRECT_WITH_EFFORT = 3    # Correct with some effort
    EASY_CORRECT = 4           # Correct, easy recall
    PERFECT = 5                # Perfect, automatic recall


@dataclass
class RepetitionSchedule:
    """Schedule for a single memory's repetitions."""
    memory_id: str
    next_review: float          # Night index for next review
    interval: float             # Current interval (nights)
    easiness: float             # SM-2 easiness factor (1.3-2.5)
    repetition_count: int       # Number of successful reviews
    last_review: float          # Night of last review
    last_quality: ReviewQuality = ReviewQuality.CORRECT_WITH_EFFORT
    streak: int = 0             # Consecutive successful reviews
    total_reviews: int = 0
    created_night: float = 0.0

    def is_due(self, current_night: float) -> bool:
        """Check if memory is due for review."""
        return current_night >= self.next_review

    def days_until_due(self, current_night: float) -> float:
        """Get nights until next review."""
        return max(0.0, self.next_review - current_night)

    def is_new(self) -> bool:
        """Check if this is a new memory (never reviewed)."""
        return self.repetition_count == 0


@dataclass
class ReviewResult:
    """Result of a memory review."""
    memory_id: str
    night_index: float
    quality: ReviewQuality
    old_interval: float
    new_interval: float
    old_easiness: float
    new_easiness: float
    passed: bool


class SpacedRepetitionScheduler:
    """
    Scheduler for spaced repetition across multiple nights.

    Implements a modified SM-2 algorithm optimized for
    sleep-based memory consolidation.
    """

    def __init__(
        self,
        initial_interval: float = 1.0,
        initial_easiness: float = 2.5,
        minimum_easiness: float = 1.3,
        maximum_easiness: float = 2.5,
        maximum_interval: float = 365.0,  # Cap at 1 year
        graduating_interval: float = 1.0,
        easy_bonus: float = 1.3,
        random_seed: Optional[int] = None,
    ):
        self.initial_interval = initial_interval
        self.initial_easiness = initial_easiness
        self.minimum_easiness = minimum_easiness
        self.maximum_easiness = maximum_easiness
        self.maximum_interval = maximum_interval
        self.graduating_interval = graduating_interval
        self.easy_bonus = easy_bonus
        self._rng = np.random.default_rng(random_seed)

        # Schedules per memory
        self._schedules: Dict[str, RepetitionSchedule] = {}

        # Review history
        self._review_history: List[ReviewResult] = []

        # Current night index
        self._current_night: float = 0.0

        # Statistics
        self._total_reviews = 0
        self._successful_reviews = 0

    def register_memory(
        self,
        memory_id: str,
        initial_easiness: Optional[float] = None,
        created_night: Optional[float] = None,
    ) -> RepetitionSchedule:
        """Register a new memory for scheduling."""
        schedule = RepetitionSchedule(
            memory_id=memory_id,
            next_review=created_night or self._current_night,
            interval=self.initial_interval,
            easiness=initial_easiness or self.initial_easiness,
            repetition_count=0,
            last_review=created_night or self._current_night,
            created_night=created_night or self._current_night,
        )
        self._schedules[memory_id] = schedule
        return schedule

    def get_schedule(self, memory_id: str) -> Optional[RepetitionSchedule]:
        """Get schedule for a memory."""
        return self._schedules.get(memory_id)

    def advance_night(self, nights: float = 1.0) -> None:
        """Advance the current night counter."""
        self._current_night += nights

    def set_night(self, night: float) -> None:
        """Set the current night index."""
        self._current_night = night

    def schedule_review(
        self,
        memory_id: str,
        quality: ReviewQuality,
    ) -> RepetitionSchedule:
        """
        Update schedule after a review.

        Uses SM-2 algorithm with modifications:
        - Quality 0-2: Reset to beginning (lapse)
        - Quality 3-5: Progress with interval increase
        """
        self._total_reviews += 1

        schedule = self._schedules.get(memory_id)
        if schedule is None:
            schedule = self.register_memory(memory_id)

        old_interval = schedule.interval
        old_easiness = schedule.easiness

        # Update easiness factor
        quality_value = quality.value
        new_easiness = schedule.easiness + (
            0.1 - (5 - quality_value) * (0.08 + (5 - quality_value) * 0.02)
        )
        new_easiness = max(self.minimum_easiness, min(self.maximum_easiness, new_easiness))

        # Determine if review passed
        passed = quality_value >= 3

        if passed:
            self._successful_reviews += 1
            schedule.streak += 1

            if schedule.repetition_count == 0:
                # First review
                new_interval = self.graduating_interval
            elif schedule.repetition_count == 1:
                # Second review
                new_interval = 6.0
            else:
                # Subsequent reviews
                new_interval = schedule.interval * new_easiness

            # Easy bonus
            if quality_value >= 4:
                new_interval *= self.easy_bonus

            schedule.repetition_count += 1
        else:
            # Lapse - reset interval but keep some easiness
            new_interval = self.initial_interval
            schedule.repetition_count = 0
            schedule.streak = 0
            # Reduce easiness more for complete failures
            if quality_value == 0:
                new_easiness = max(self.minimum_easiness, new_easiness - 0.2)

        # Add some randomness to prevent clustering
        fuzz = self._rng.uniform(0.95, 1.05)
        new_interval = max(1.0, min(self.maximum_interval, new_interval * fuzz))

        # Update schedule
        schedule.interval = new_interval
        schedule.easiness = new_easiness
        schedule.next_review = self._current_night + new_interval
        schedule.last_review = self._current_night
        schedule.last_quality = quality
        schedule.total_reviews += 1

        # Record result
        result = ReviewResult(
            memory_id=memory_id,
            night_index=self._current_night,
            quality=quality,
            old_interval=old_interval,
            new_interval=new_interval,
            old_easiness=old_easiness,
            new_easiness=new_easiness,
            passed=passed,
        )
        self._review_history.append(result)

        return schedule

    def update_after_review(
        self,
        memory_id: str,
        success: bool,
        confidence: float = 0.5,
    ) -> RepetitionSchedule:
        """
        Simplified update interface.

        Converts success/confidence to quality rating.
        """
        if success:
            if confidence > 0.9:
                quality = ReviewQuality.PERFECT
            elif confidence > 0.7:
                quality = ReviewQuality.EASY_CORRECT
            elif confidence > 0.5:
                quality = ReviewQuality.CORRECT_WITH_EFFORT
            else:
                quality = ReviewQuality.DIFFICULT_CORRECT
        else:
            if confidence > 0.3:
                quality = ReviewQuality.INCORRECT
            else:
                quality = ReviewQuality.COMPLETE_BLACKOUT

        return self.schedule_review(memory_id, quality)

    def get_due_memories(
        self,
        night_index: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> List[str]:
        """
        Get memories due for review.

        Returns list of memory IDs sorted by urgency.
        """
        night = night_index if night_index is not None else self._current_night

        due = []
        for memory_id, schedule in self._schedules.items():
            if schedule.is_due(night):
                overdue = night - schedule.next_review
                due.append((memory_id, overdue))

        # Sort by how overdue (most overdue first)
        due.sort(key=lambda x: x[1], reverse=True)

        memory_ids = [m_id for m_id, _ in due]

        if limit is not None:
            memory_ids = memory_ids[:limit]

        return memory_ids

    def get_upcoming_reviews(
        self,
        days_ahead: float = 7.0,
    ) -> Dict[float, List[str]]:
        """Get reviews scheduled for upcoming nights."""
        upcoming = {}

        for memory_id, schedule in self._schedules.items():
            if schedule.next_review <= self._current_night + days_ahead:
                night = round(schedule.next_review)
                if night not in upcoming:
                    upcoming[night] = []
                upcoming[night].append(memory_id)

        return upcoming

    def get_memory_retention_estimate(
        self,
        memory_id: str,
        at_night: Optional[float] = None,
    ) -> float:
        """
        Estimate retention probability at a given night.

        Uses forgetting curve: R = e^(-t/S) where S is stability.
        """
        schedule = self._schedules.get(memory_id)
        if schedule is None:
            return 0.0

        night = at_night if at_night is not None else self._current_night
        days_since_review = night - schedule.last_review

        if days_since_review <= 0:
            return 1.0

        # Stability approximated by interval * easiness
        stability = schedule.interval * (schedule.easiness / 2.5)

        # Forgetting curve
        retention = np.exp(-days_since_review / stability)
        return float(np.clip(retention, 0.0, 1.0))

    def export_schedule(self) -> Dict[str, Any]:
        """Export all schedules as dictionary."""
        return {
            memory_id: {
                "next_review": s.next_review,
                "interval": s.interval,
                "easiness": s.easiness,
                "repetition_count": s.repetition_count,
                "streak": s.streak,
                "total_reviews": s.total_reviews,
            }
            for memory_id, s in self._schedules.items()
        }

    def import_schedule(self, data: Dict[str, Any]) -> None:
        """Import schedules from dictionary."""
        for memory_id, schedule_data in data.items():
            schedule = RepetitionSchedule(
                memory_id=memory_id,
                next_review=schedule_data.get("next_review", 0.0),
                interval=schedule_data.get("interval", self.initial_interval),
                easiness=schedule_data.get("easiness", self.initial_easiness),
                repetition_count=schedule_data.get("repetition_count", 0),
                last_review=schedule_data.get("next_review", 0.0) - schedule_data.get("interval", 0.0),
                streak=schedule_data.get("streak", 0),
                total_reviews=schedule_data.get("total_reviews", 0),
            )
            self._schedules[memory_id] = schedule

    def get_workload_forecast(
        self,
        days: int = 30,
    ) -> List[int]:
        """Get forecast of reviews per night for upcoming days."""
        forecast = [0] * days

        for schedule in self._schedules.values():
            # Simulate future reviews
            next_review = schedule.next_review
            interval = schedule.interval

            while next_review < self._current_night + days:
                day_idx = int(next_review - self._current_night)
                if 0 <= day_idx < days:
                    forecast[day_idx] += 1
                next_review += interval
                interval *= schedule.easiness  # Approximate future intervals

        return forecast

    def optimize_schedule(
        self,
        target_daily_reviews: int = 20,
    ) -> None:
        """
        Redistribute reviews to balance daily workload.

        Moves some reviews earlier or later to smooth workload.
        """
        forecast = self.get_workload_forecast(days=14)

        # Find overloaded days
        for day_idx, count in enumerate(forecast):
            if count > target_daily_reviews:
                # Move some reviews to adjacent days
                night = self._current_night + day_idx
                due_that_day = [
                    (m_id, s) for m_id, s in self._schedules.items()
                    if abs(s.next_review - night) < 0.5
                ]

                # Sort by flexibility (higher easiness = more flexible)
                due_that_day.sort(key=lambda x: x[1].easiness, reverse=True)

                # Move excess reviews
                excess = count - target_daily_reviews
                for i in range(min(excess, len(due_that_day))):
                    memory_id, schedule = due_that_day[i]
                    # Move by 1-2 days (preferring earlier for safety)
                    offset = self._rng.choice([-1, 1, -2, 2])
                    schedule.next_review += offset

    def statistics(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        if not self._schedules:
            return {
                "total_memories": 0,
                "current_night": self._current_night,
            }

        intervals = [s.interval for s in self._schedules.values()]
        easinesses = [s.easiness for s in self._schedules.values()]
        streaks = [s.streak for s in self._schedules.values()]

        due_now = len(self.get_due_memories())
        new_memories = sum(1 for s in self._schedules.values() if s.is_new())
        mature = sum(1 for s in self._schedules.values() if s.interval >= 21)

        return {
            "total_memories": len(self._schedules),
            "current_night": self._current_night,
            "due_now": due_now,
            "new_memories": new_memories,
            "mature_memories": mature,
            "average_interval": float(np.mean(intervals)),
            "max_interval": float(np.max(intervals)),
            "average_easiness": float(np.mean(easinesses)),
            "average_streak": float(np.mean(streaks)),
            "max_streak": int(np.max(streaks)),
            "total_reviews": self._total_reviews,
            "successful_reviews": self._successful_reviews,
            "success_rate": (
                self._successful_reviews / self._total_reviews
                if self._total_reviews > 0 else 0.0
            ),
            "workload_7day": self.get_workload_forecast(days=7),
        }


__all__ = [
    'ReviewQuality',
    'RepetitionSchedule',
    'ReviewResult',
    'SpacedRepetitionScheduler',
]
