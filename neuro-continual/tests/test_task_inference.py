"""Tests for task inference."""

import pytest
import numpy as np

from neuro.modules.continual.task_inference import (
    TaskInference,
    TaskInferenceConfig,
    TaskInfo,
    TaskChangeMethod,
)


class TestTaskInferenceConfig:
    """Tests for TaskInferenceConfig class."""

    def test_default_config(self):
        config = TaskInferenceConfig()
        assert config.change_threshold == 0.5
        assert config.embedding_dim == 64
        assert config.min_samples_for_task == 10

    def test_custom_config(self):
        config = TaskInferenceConfig(change_threshold=0.3, embedding_dim=128)
        assert config.change_threshold == 0.3
        assert config.embedding_dim == 128


class TestTaskInference:
    """Tests for TaskInference class."""

    def test_creation(self):
        ti = TaskInference(random_seed=42)
        assert ti is not None

    def test_detect_first_task(self):
        ti = TaskInference(random_seed=42)
        state = np.random.randn(10)

        changed = ti.detect_task_change(state)

        assert changed
        assert ti.get_current_task() is not None

    def test_no_change_same_distribution(self):
        ti = TaskInference(random_seed=42)

        for i in range(20):
            state = np.random.randn(10) * 0.1
            ti.detect_task_change(state)

        current_task = ti.get_current_task()

        for i in range(10):
            state = np.random.randn(10) * 0.1
            changed = ti.detect_task_change(state)

        assert ti.get_current_task() == current_task

    def test_infer_task_id(self):
        ti = TaskInference(random_seed=42)

        state1 = np.ones(10)
        task_id = ti.infer_task_id(state1)

        assert task_id is not None
        assert task_id.startswith("task_")

    def test_infer_same_task(self):
        config = TaskInferenceConfig(similarity_threshold=0.5)
        ti = TaskInference(config=config, random_seed=42)

        state1 = np.ones(10)
        task1 = ti.infer_task_id(state1)

        state2 = np.ones(10) * 1.1
        task2 = ti.infer_task_id(state2)

        assert task1 == task2

    def test_infer_different_task(self):
        config = TaskInferenceConfig(similarity_threshold=0.9)
        ti = TaskInference(config=config, random_seed=42)

        state1 = np.ones(10)
        task1 = ti.infer_task_id(state1)

        state2 = -np.ones(10)
        task2 = ti.infer_task_id(state2)

        assert task1 != task2

    def test_create_task_embedding(self):
        ti = TaskInference(random_seed=42)

        exemplars = [np.random.randn(10) for _ in range(5)]
        embedding = ti.create_task_embedding(exemplars)

        assert embedding.shape == (ti.config.embedding_dim,)
        assert np.abs(np.linalg.norm(embedding) - 1.0) < 0.01

    def test_create_embedding_empty(self):
        ti = TaskInference(random_seed=42)

        embedding = ti.create_task_embedding([])

        assert embedding.shape == (ti.config.embedding_dim,)
        assert np.allclose(embedding, 0)

    def test_merge_similar_tasks(self):
        config = TaskInferenceConfig(similarity_threshold=0.5)
        ti = TaskInference(config=config, random_seed=42)

        state1 = np.ones(10)
        ti.infer_task_id(state1)

        state2 = np.ones(10) * 0.9
        ti.infer_task_id(state2)

        initial_tasks = len(ti.get_all_tasks())

        merged = ti.merge_similar_tasks(threshold=0.5)

        assert ti.get_all_tasks()

    def test_record_performance(self):
        ti = TaskInference(random_seed=42)

        state = np.random.randn(10)
        task_id = ti.infer_task_id(state)

        ti.record_performance(task_id, 0.8)
        ti.record_performance(task_id, 0.9)

        info = ti.get_task_info(task_id)
        assert len(info.performance_history) == 2

    def test_get_task_info(self):
        ti = TaskInference(random_seed=42)

        state = np.random.randn(10)
        task_id = ti.infer_task_id(state)

        info = ti.get_task_info(task_id)

        assert info is not None
        assert info.task_id == task_id
        assert info.sample_count >= 1

    def test_get_nonexistent_task(self):
        ti = TaskInference(random_seed=42)

        info = ti.get_task_info("nonexistent")
        assert info is None

    def test_set_current_task(self):
        ti = TaskInference(random_seed=42)

        state = np.random.randn(10)
        task_id = ti.infer_task_id(state)

        ti.set_current_task(task_id)
        assert ti.get_current_task() == task_id

    def test_statistics(self):
        ti = TaskInference(random_seed=42)

        state = np.random.randn(10)
        ti.detect_task_change(state)

        stats = ti.statistics()

        assert "total_tasks" in stats
        assert "current_task" in stats
        assert "task_changes" in stats
        assert stats["total_tasks"] >= 1


class TestTaskChangeMethod:
    """Tests for TaskChangeMethod enum."""

    def test_enum_values(self):
        assert TaskChangeMethod.DISTRIBUTION_SHIFT.value == "distribution_shift"
        assert TaskChangeMethod.PERFORMANCE_DROP.value == "performance_drop"
        assert TaskChangeMethod.CONTEXT_CHANGE.value == "context_change"
        assert TaskChangeMethod.REWARD_STRUCTURE.value == "reward_structure"
