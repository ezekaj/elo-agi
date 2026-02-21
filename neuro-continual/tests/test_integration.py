"""Tests for continual learning controller integration."""

import pytest
import numpy as np

from neuro.modules.continual.integration import (
    ContinualLearningController,
    ContinualLearningConfig,
)
from neuro.modules.continual.forgetting_prevention import ForgettingPreventionMethod

class TestContinualLearningConfig:
    """Tests for ContinualLearningConfig class."""

    def test_default_config(self):
        config = ContinualLearningConfig()
        assert config.task_change_threshold == 0.5
        assert config.consolidation_frequency == 100
        assert config.replay_batch_size == 32
        assert config.auto_consolidate

    def test_custom_config(self):
        config = ContinualLearningConfig(
            task_change_threshold=0.3,
            ewc_lambda=500.0,
            auto_detect_tasks=False,
        )
        assert config.task_change_threshold == 0.3
        assert config.ewc_lambda == 500.0
        assert not config.auto_detect_tasks

class TestContinualLearningController:
    """Tests for ContinualLearningController class."""

    def test_creation(self):
        controller = ContinualLearningController()
        assert controller is not None

    def test_creation_with_config(self):
        config = ContinualLearningConfig(random_seed=42)
        controller = ContinualLearningController(config=config)
        assert controller.config.random_seed == 42

    def test_observe_first(self):
        config = ContinualLearningConfig(random_seed=42, auto_detect_tasks=True)
        controller = ContinualLearningController(config=config)

        state = np.random.randn(10)
        result = controller.observe(state, 0, 1.0, state, False)

        assert "task_id" in result
        assert result["task_id"] is not None

    def test_observe_accumulates_replay(self):
        config = ContinualLearningConfig(random_seed=42, auto_detect_tasks=False)
        controller = ContinualLearningController(config=config)

        for i in range(10):
            state = np.random.randn(10)
            controller.observe(state, i, 1.0, state, False)

        assert controller.get_replay_buffer_size() == 10

    def test_observe_with_params(self):
        config = ContinualLearningConfig(random_seed=42, auto_detect_tasks=False)
        controller = ContinualLearningController(config=config)

        params = {"layer1": np.random.randn(5, 5)}

        state = np.random.randn(10)
        result = controller.observe(state, 0, 1.0, state, False, params=params)

        assert "forgetting_loss" in result
        assert isinstance(result["forgetting_loss"], float)

    def test_sample_replay(self):
        config = ContinualLearningConfig(random_seed=42, auto_detect_tasks=False)
        controller = ContinualLearningController(config=config)

        for i in range(50):
            state = np.random.randn(10)
            controller.observe(state, i, 1.0, state, False)

        batch = controller.sample_replay(batch_size=10)

        assert len(batch) == 10

    def test_sample_replay_empty(self):
        controller = ContinualLearningController()

        batch = controller.sample_replay(batch_size=10)

        assert len(batch) == 0

    def test_update_replay_priorities(self):
        config = ContinualLearningConfig(random_seed=42, auto_detect_tasks=False)
        controller = ContinualLearningController(config=config)

        for i in range(10):
            state = np.random.randn(10)
            controller.observe(state, i, 1.0, state, False)

        controller.update_replay_priorities([0, 1, 2], [5.0, 3.0, 1.0])

    def test_compute_forgetting_loss(self):
        config = ContinualLearningConfig(
            random_seed=42,
            forgetting_method=ForgettingPreventionMethod.EWC,
        )
        controller = ContinualLearningController(config=config)

        old_params = {"layer1": np.zeros((5, 5))}
        controller.register_task_params("task1", old_params)

        new_params = {"layer1": np.ones((5, 5))}
        total, losses = controller.compute_forgetting_loss(new_params)

        assert total > 0
        assert "ewc" in losses

    def test_register_task_params(self):
        controller = ContinualLearningController()

        params = {"layer1": np.random.randn(5, 5)}
        controller.register_task_params("task1", params)

        assert controller._forgetting.get_task_memory("task1") is not None

    def test_measure_capability(self):
        controller = ContinualLearningController()

        results = {"test1": 0.8, "test2": 0.9}
        metric = controller.measure_capability("reasoning", results)

        assert metric.name == "reasoning"
        assert metric.score == pytest.approx(0.85)

    def test_update_performance(self):
        config = ContinualLearningConfig(random_seed=42, auto_detect_tasks=False)
        controller = ContinualLearningController(config=config)

        controller._consolidation.register_task("task1")

        controller.update_performance("task1", 0.8)

        history = controller._consolidation.get_performance_history("task1")
        assert 0.8 in history

    def test_trigger_consolidation(self):
        controller = ContinualLearningController()

        controller._consolidation.register_task("task1", 0.8)
        controller._consolidation.update_performance("task1", 0.4)

        plan = controller.trigger_consolidation()

        assert plan is not None
        assert plan.total_budget > 0

    def test_get_regressing_capabilities(self):
        config = ContinualLearningConfig(random_seed=42)
        controller = ContinualLearningController(config=config)

        for i in range(5):
            controller.measure_capability("reasoning", {"test": 0.9 - i * 0.1})

        regressing = controller.get_regressing_capabilities()

        assert isinstance(regressing, list)

    def test_get_remediation_suggestions(self):
        config = ContinualLearningConfig(random_seed=42)
        controller = ContinualLearningController(config=config)

        for i in range(5):
            controller.measure_capability("reasoning", {"test": 0.9 - i * 0.1})

        suggestions = controller.get_remediation_suggestions()

        assert isinstance(suggestions, dict)

    def test_get_current_task(self):
        config = ContinualLearningConfig(random_seed=42, auto_detect_tasks=True)
        controller = ContinualLearningController(config=config)

        state = np.random.randn(10)
        controller.observe(state, 0, 1.0, state, False)

        task = controller.get_current_task()

        assert task is not None

    def test_get_all_tasks(self):
        config = ContinualLearningConfig(random_seed=42, auto_detect_tasks=True)
        controller = ContinualLearningController(config=config)

        state = np.random.randn(10)
        controller.observe(state, 0, 1.0, state, False)

        tasks = controller.get_all_tasks()

        assert len(tasks) >= 1

    def test_get_task_info(self):
        config = ContinualLearningConfig(random_seed=42, auto_detect_tasks=True)
        controller = ContinualLearningController(config=config)

        state = np.random.randn(10)
        controller.observe(state, 0, 1.0, state, False)

        task_id = controller.get_current_task()
        info = controller.get_task_info(task_id)

        assert info is not None
        assert "task_id" in info

    def test_reset(self):
        config = ContinualLearningConfig(random_seed=42, auto_detect_tasks=False)
        controller = ContinualLearningController(config=config)

        for i in range(10):
            state = np.random.randn(10)
            controller.observe(state, i, 1.0, state, False)

        controller.reset()

        assert controller.get_replay_buffer_size() == 0

    def test_statistics(self):
        config = ContinualLearningConfig(random_seed=42, auto_detect_tasks=False)
        controller = ContinualLearningController(config=config)

        state = np.random.randn(10)
        controller.observe(state, 0, 1.0, state, False)

        stats = controller.statistics()

        assert "timestep" in stats
        assert "task_inference" in stats
        assert "consolidation" in stats
        assert "forgetting" in stats
        assert "replay" in stats
        assert "capabilities" in stats

class TestEndToEndContinualLearning:
    """End-to-end tests for continual learning."""

    def test_full_learning_cycle(self):
        config = ContinualLearningConfig(
            random_seed=42,
            auto_detect_tasks=True,
            auto_consolidate=False,
            consolidation_frequency=50,
        )
        controller = ContinualLearningController(config=config)

        for step in range(100):
            state = np.random.randn(10)
            action = step % 4
            reward = float(step % 10) / 10

            result = controller.observe(
                state=state,
                action=action,
                reward=reward,
                next_state=state,
                done=(step % 20 == 19),
            )

        stats = controller.statistics()
        assert stats["replay"]["buffer_size"] == 100

    def test_task_transition(self):
        config = ContinualLearningConfig(
            random_seed=42,
            auto_detect_tasks=True,
            task_change_threshold=0.3,
        )
        controller = ContinualLearningController(config=config)

        for i in range(20):
            state = np.ones(10) * i
            controller.observe(state, 0, 1.0, state, False)

        task_changes = controller._total_task_changes

        for i in range(20):
            state = -np.ones(10) * i
            controller.observe(state, 0, 1.0, state, False)

        assert controller.get_replay_buffer_size() == 40

    def test_forgetting_prevention_integration(self):
        config = ContinualLearningConfig(
            random_seed=42,
            forgetting_method=ForgettingPreventionMethod.EWC,
            ewc_lambda=100.0,
        )
        controller = ContinualLearningController(config=config)

        params_task1 = {"layer1": np.zeros((5, 5))}
        controller.register_task_params("task1", params_task1)

        params_task2 = {"layer1": np.ones((5, 5))}
        total, losses = controller.compute_forgetting_loss(params_task2)

        assert total > 0

    def test_capability_monitoring_integration(self):
        config = ContinualLearningConfig(random_seed=42)
        controller = ContinualLearningController(config=config)

        for i in range(10):
            score = 0.5 + i * 0.05
            controller.measure_capability("reasoning", {"test": score})

        record = controller._capabilities.get_capability_record("reasoning")

        assert record is not None
        assert record.current_score > 0.5
