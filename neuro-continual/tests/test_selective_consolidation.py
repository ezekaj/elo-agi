"""Tests for selective consolidation."""

import pytest
import numpy as np

from src.selective_consolidation import (
    SelectiveConsolidation,
    ConsolidationConfig,
    ConsolidationStrategy,
    ConsolidationPlan,
    PerformanceRecord,
)


class TestConsolidationConfig:
    """Tests for ConsolidationConfig class."""

    def test_default_config(self):
        config = ConsolidationConfig()
        assert config.strategy == ConsolidationStrategy.PERFORMANCE_WEIGHTED
        assert config.min_gap_for_consolidation == 0.1
        assert config.max_consolidation_budget == 1000

    def test_custom_config(self):
        config = ConsolidationConfig(
            strategy=ConsolidationStrategy.UNIFORM,
            max_consolidation_budget=500,
        )
        assert config.strategy == ConsolidationStrategy.UNIFORM
        assert config.max_consolidation_budget == 500


class TestSelectiveConsolidation:
    """Tests for SelectiveConsolidation class."""

    def test_creation(self):
        sc = SelectiveConsolidation(random_seed=42)
        assert sc is not None

    def test_register_task(self):
        sc = SelectiveConsolidation(random_seed=42)

        sc.register_task("task1", initial_performance=0.5)

        assert "task1" in sc.get_all_tasks()

    def test_update_performance(self):
        sc = SelectiveConsolidation(random_seed=42)

        sc.register_task("task1", 0.5)
        sc.update_performance("task1", 0.8)

        history = sc.get_performance_history("task1")
        assert 0.8 in history

    def test_update_unregistered_task(self):
        sc = SelectiveConsolidation(random_seed=42)

        sc.update_performance("new_task", 0.7)

        assert "new_task" in sc.get_all_tasks()

    def test_assess_performance_gap_no_regression(self):
        sc = SelectiveConsolidation(random_seed=42)

        sc.register_task("task1", 0.5)
        sc.update_performance("task1", 0.8)

        gap = sc.assess_performance_gap("task1")
        assert gap == 0.0

    def test_assess_performance_gap_with_regression(self):
        sc = SelectiveConsolidation(random_seed=42)

        sc.register_task("task1", 0.8)
        sc.update_performance("task1", 0.5)

        gap = sc.assess_performance_gap("task1")
        assert gap == pytest.approx(0.3)

    def test_assess_gap_custom_values(self):
        sc = SelectiveConsolidation(random_seed=42)

        sc.register_task("task1", 0.5)

        gap = sc.assess_performance_gap("task1", current=0.3, historical=0.7)
        assert gap == pytest.approx(0.4)

    def test_prioritize_consolidation(self):
        sc = SelectiveConsolidation(random_seed=42)

        sc.register_task("task1", 0.8)
        sc.update_performance("task1", 0.5)

        sc.register_task("task2", 0.9)
        sc.update_performance("task2", 0.3)

        priorities = sc.prioritize_consolidation()

        assert len(priorities) >= 1
        assert priorities[0][0] == "task2"

    def test_prioritize_no_regression(self):
        config = ConsolidationConfig(min_gap_for_consolidation=0.1)
        sc = SelectiveConsolidation(config=config, random_seed=42)

        sc.register_task("task1", 0.5)
        sc.update_performance("task1", 0.6)

        priorities = sc.prioritize_consolidation()

        assert len(priorities) == 0

    def test_allocate_budget(self):
        sc = SelectiveConsolidation(random_seed=42)

        tasks = [("task1", 2.0), ("task2", 1.0)]

        allocation = sc.allocate_consolidation_budget(tasks, total_budget=100)

        assert "task1" in allocation
        assert "task2" in allocation
        assert sum(allocation.values()) == 100
        assert allocation["task1"] > allocation["task2"]

    def test_allocate_budget_empty(self):
        sc = SelectiveConsolidation(random_seed=42)

        allocation = sc.allocate_consolidation_budget([], total_budget=100)

        assert allocation == {}

    def test_create_consolidation_plan(self):
        sc = SelectiveConsolidation(random_seed=42)

        sc.register_task("task1", 0.8)
        sc.update_performance("task1", 0.4)

        plan = sc.create_consolidation_plan(total_budget=100)

        assert isinstance(plan, ConsolidationPlan)
        assert plan.total_budget == 100

    def test_get_tasks_needing_consolidation(self):
        sc = SelectiveConsolidation(random_seed=42)

        sc.register_task("good", 0.5)
        sc.update_performance("good", 0.6)

        sc.register_task("bad", 0.8)
        sc.update_performance("bad", 0.3)

        tasks = sc.get_tasks_needing_consolidation(threshold=0.2)

        assert "bad" in tasks
        assert "good" not in tasks

    def test_reset(self):
        sc = SelectiveConsolidation(random_seed=42)

        sc.register_task("task1", 0.5)
        sc.reset()

        assert len(sc.get_all_tasks()) == 0

    def test_statistics(self):
        sc = SelectiveConsolidation(random_seed=42)

        sc.register_task("task1", 0.8)
        sc.update_performance("task1", 0.5)

        stats = sc.statistics()

        assert "total_tasks" in stats
        assert "total_consolidations" in stats
        assert "avg_performance_gap" in stats
        assert stats["total_tasks"] == 1


class TestConsolidationStrategy:
    """Tests for ConsolidationStrategy enum."""

    def test_strategies(self):
        assert ConsolidationStrategy.UNIFORM.value == "uniform"
        assert ConsolidationStrategy.PERFORMANCE_WEIGHTED.value == "performance_weighted"
        assert ConsolidationStrategy.RECENCY_WEIGHTED.value == "recency_weighted"
        assert ConsolidationStrategy.DIFFICULTY_WEIGHTED.value == "difficulty_weighted"

    def test_different_strategies(self):
        for strategy in ConsolidationStrategy:
            config = ConsolidationConfig(strategy=strategy)
            sc = SelectiveConsolidation(config=config, random_seed=42)

            sc.register_task("task1", 0.8)
            sc.update_performance("task1", 0.4)

            priorities = sc.prioritize_consolidation()
            assert len(priorities) > 0
