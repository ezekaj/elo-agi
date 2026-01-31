"""
Selective Consolidation for Continual Learning

Implements performance-based memory consolidation:
- Performance gap assessment across tasks
- Consolidation prioritization
- Budget allocation for replay/rehearsal
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np


class ConsolidationStrategy(Enum):
    """Strategies for consolidation."""
    UNIFORM = "uniform"
    PERFORMANCE_WEIGHTED = "performance_weighted"
    RECENCY_WEIGHTED = "recency_weighted"
    DIFFICULTY_WEIGHTED = "difficulty_weighted"


@dataclass
class ConsolidationConfig:
    """Configuration for selective consolidation."""
    strategy: ConsolidationStrategy = ConsolidationStrategy.PERFORMANCE_WEIGHTED
    min_gap_for_consolidation: float = 0.1
    max_consolidation_budget: int = 1000
    decay_rate: float = 0.95
    priority_exponent: float = 2.0


@dataclass
class PerformanceRecord:
    """Performance record for a task."""
    task_id: str
    historical_performance: List[float]
    current_performance: float
    peak_performance: float
    last_updated: int


@dataclass
class ConsolidationPlan:
    """Plan for consolidation."""
    task_priorities: List[Tuple[str, float]]
    budget_allocation: Dict[str, int]
    total_budget: int
    strategy: ConsolidationStrategy


class SelectiveConsolidation:
    """
    Manages selective memory consolidation based on performance.

    Prioritizes consolidation for:
    - Tasks showing performance degradation
    - Tasks critical for overall system performance
    - Tasks that haven't been recently rehearsed
    """

    def __init__(
        self,
        config: Optional[ConsolidationConfig] = None,
        random_seed: Optional[int] = None,
    ):
        self.config = config or ConsolidationConfig()
        self._rng = np.random.default_rng(random_seed)

        self._performance_records: Dict[str, PerformanceRecord] = {}
        self._consolidation_history: List[Dict[str, Any]] = []

        self._timestep = 0
        self._total_consolidations = 0

    def register_task(self, task_id: str, initial_performance: float = 0.0) -> None:
        """Register a new task for tracking."""
        if task_id not in self._performance_records:
            self._performance_records[task_id] = PerformanceRecord(
                task_id=task_id,
                historical_performance=[initial_performance],
                current_performance=initial_performance,
                peak_performance=initial_performance,
                last_updated=self._timestep,
            )

    def update_performance(self, task_id: str, performance: float) -> None:
        """Update performance for a task."""
        if task_id not in self._performance_records:
            self.register_task(task_id, performance)
            return

        record = self._performance_records[task_id]
        record.historical_performance.append(performance)
        record.current_performance = performance
        record.peak_performance = max(record.peak_performance, performance)
        record.last_updated = self._timestep

    def assess_performance_gap(
        self,
        task_id: str,
        current: Optional[float] = None,
        historical: Optional[float] = None,
    ) -> float:
        """
        Assess performance gap for a task.

        Args:
            task_id: Task identifier
            current: Current performance (uses recorded if None)
            historical: Historical baseline (uses peak if None)

        Returns:
            Performance gap (positive = degradation)
        """
        if task_id not in self._performance_records:
            return 0.0

        record = self._performance_records[task_id]

        if current is None:
            current = record.current_performance
        if historical is None:
            historical = record.peak_performance

        gap = historical - current
        return max(0.0, gap)

    def prioritize_consolidation(
        self,
        gaps: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Prioritize tasks for consolidation.

        Args:
            gaps: Optional dict of task_id to performance gap

        Returns:
            List of (task_id, priority) sorted by priority
        """
        if gaps is None:
            gaps = {
                task_id: self.assess_performance_gap(task_id)
                for task_id in self._performance_records
            }

        priorities = []

        for task_id, gap in gaps.items():
            if gap < self.config.min_gap_for_consolidation:
                continue

            priority = self._compute_priority(task_id, gap)
            priorities.append((task_id, priority))

        priorities.sort(key=lambda x: x[1], reverse=True)
        return priorities

    def _compute_priority(self, task_id: str, gap: float) -> float:
        """Compute consolidation priority for a task."""
        record = self._performance_records.get(task_id)
        if record is None:
            return 0.0

        gap_factor = gap ** self.config.priority_exponent

        if self.config.strategy == ConsolidationStrategy.UNIFORM:
            return gap_factor

        elif self.config.strategy == ConsolidationStrategy.PERFORMANCE_WEIGHTED:
            importance = record.peak_performance
            return gap_factor * (importance + 0.1)

        elif self.config.strategy == ConsolidationStrategy.RECENCY_WEIGHTED:
            time_since_update = self._timestep - record.last_updated + 1
            recency_factor = self.config.decay_rate ** time_since_update
            return gap_factor * (1.0 + 1.0 - recency_factor)

        elif self.config.strategy == ConsolidationStrategy.DIFFICULTY_WEIGHTED:
            if record.historical_performance:
                variance = np.var(record.historical_performance)
                difficulty = np.sqrt(variance + 1e-8)
            else:
                difficulty = 1.0
            return gap_factor * difficulty

        return gap_factor

    def allocate_consolidation_budget(
        self,
        tasks: List[Tuple[str, float]],
        total_budget: Optional[int] = None,
    ) -> Dict[str, int]:
        """
        Allocate consolidation budget across tasks.

        Args:
            tasks: List of (task_id, priority) tuples
            total_budget: Total budget to allocate

        Returns:
            Dict mapping task_id to allocated budget
        """
        if total_budget is None:
            total_budget = self.config.max_consolidation_budget

        if not tasks:
            return {}

        total_priority = sum(p for _, p in tasks)
        if total_priority < 1e-8:
            per_task = total_budget // len(tasks)
            return {task_id: per_task for task_id, _ in tasks}

        allocation = {}
        remaining = total_budget

        for task_id, priority in tasks:
            share = int(total_budget * priority / total_priority)
            share = min(share, remaining)
            if share > 0:
                allocation[task_id] = share
                remaining -= share

        if remaining > 0 and allocation:
            top_task = tasks[0][0]
            allocation[top_task] = allocation.get(top_task, 0) + remaining

        return allocation

    def create_consolidation_plan(
        self,
        total_budget: Optional[int] = None,
    ) -> ConsolidationPlan:
        """
        Create a full consolidation plan.

        Args:
            total_budget: Total budget for consolidation

        Returns:
            ConsolidationPlan with priorities and allocation
        """
        if total_budget is None:
            total_budget = self.config.max_consolidation_budget

        priorities = self.prioritize_consolidation()
        allocation = self.allocate_consolidation_budget(priorities, total_budget)

        plan = ConsolidationPlan(
            task_priorities=priorities,
            budget_allocation=allocation,
            total_budget=total_budget,
            strategy=self.config.strategy,
        )

        self._consolidation_history.append({
            "timestep": self._timestep,
            "num_tasks": len(priorities),
            "total_budget": total_budget,
            "allocated_budget": sum(allocation.values()),
        })

        self._total_consolidations += 1
        self._timestep += 1

        return plan

    def get_tasks_needing_consolidation(
        self,
        threshold: Optional[float] = None,
    ) -> List[str]:
        """Get tasks that need consolidation."""
        if threshold is None:
            threshold = self.config.min_gap_for_consolidation

        tasks = []
        for task_id in self._performance_records:
            gap = self.assess_performance_gap(task_id)
            if gap >= threshold:
                tasks.append(task_id)

        return tasks

    def get_performance_history(
        self,
        task_id: str,
        n: Optional[int] = None,
    ) -> List[float]:
        """Get performance history for a task."""
        record = self._performance_records.get(task_id)
        if record is None:
            return []

        history = record.historical_performance
        if n is not None:
            history = history[-n:]

        return list(history)

    def get_all_tasks(self) -> List[str]:
        """Get all registered task IDs."""
        return list(self._performance_records.keys())

    def reset(self) -> None:
        """Reset all tracking."""
        self._performance_records.clear()
        self._consolidation_history.clear()
        self._timestep = 0
        self._total_consolidations = 0

    def statistics(self) -> Dict[str, Any]:
        """Get consolidation statistics."""
        gaps = [
            self.assess_performance_gap(tid)
            for tid in self._performance_records
        ]

        return {
            "total_tasks": len(self._performance_records),
            "total_consolidations": self._total_consolidations,
            "timestep": self._timestep,
            "avg_performance_gap": float(np.mean(gaps)) if gaps else 0.0,
            "max_performance_gap": float(np.max(gaps)) if gaps else 0.0,
            "tasks_needing_consolidation": len(self.get_tasks_needing_consolidation()),
            "strategy": self.config.strategy.value,
        }
