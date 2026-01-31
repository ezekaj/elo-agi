"""Tests for importance-weighted experience replay."""

import pytest
import numpy as np

from src.experience_replay import (
    ImportanceWeightedReplay,
    ReplayConfig,
    ReplayStrategy,
    Experience,
)


class TestReplayConfig:
    """Tests for ReplayConfig class."""

    def test_default_config(self):
        config = ReplayConfig()
        assert config.buffer_size == 10000
        assert config.strategy == ReplayStrategy.PRIORITIZED
        assert config.priority_alpha == 0.6

    def test_custom_config(self):
        config = ReplayConfig(buffer_size=1000, strategy=ReplayStrategy.UNIFORM)
        assert config.buffer_size == 1000
        assert config.strategy == ReplayStrategy.UNIFORM


class TestImportanceWeightedReplay:
    """Tests for ImportanceWeightedReplay class."""

    def test_creation(self):
        replay = ImportanceWeightedReplay(random_seed=42)
        assert replay is not None
        assert len(replay) == 0

    def test_add_experience(self):
        replay = ImportanceWeightedReplay(random_seed=42)

        state = np.random.randn(10)
        action = 0
        reward = 1.0
        next_state = np.random.randn(10)

        idx = replay.add(state, action, reward, next_state, False, "task1")

        assert len(replay) == 1
        assert idx == 0

    def test_add_multiple_experiences(self):
        replay = ImportanceWeightedReplay(random_seed=42)

        for i in range(100):
            state = np.random.randn(10)
            replay.add(state, i, 1.0, state, False, "task1")

        assert len(replay) == 100

    def test_buffer_overflow(self):
        config = ReplayConfig(buffer_size=10)
        replay = ImportanceWeightedReplay(config=config, random_seed=42)

        for i in range(20):
            state = np.random.randn(10)
            replay.add(state, i, float(i), state, False, "task1")

        assert len(replay) == 10

    def test_compute_importance(self):
        replay = ImportanceWeightedReplay(random_seed=42)

        state = np.random.randn(10)
        idx = replay.add(state, 0, 1.0, state, False, "task1")

        exp = replay._buffer[idx]
        weight = replay.compute_importance(exp, 0.5)

        assert weight > 0
        assert np.isfinite(weight)

    def test_sample_batch_uniform(self):
        config = ReplayConfig(strategy=ReplayStrategy.UNIFORM)
        replay = ImportanceWeightedReplay(config=config, random_seed=42)

        for i in range(50):
            state = np.random.randn(10)
            replay.add(state, i, 1.0, state, False, "task1")

        batch = replay.sample_batch(10)

        assert len(batch) == 10
        for exp, weight in batch:
            assert isinstance(exp, Experience)
            assert weight > 0

    def test_sample_batch_prioritized(self):
        config = ReplayConfig(strategy=ReplayStrategy.PRIORITIZED)
        replay = ImportanceWeightedReplay(config=config, random_seed=42)

        for i in range(50):
            state = np.random.randn(10)
            replay.add(state, i, float(i + 1), state, False, "task1", td_error=float(i + 1))

        batch = replay.sample_batch(10)

        assert len(batch) == 10

    def test_sample_batch_task_balanced(self):
        config = ReplayConfig(strategy=ReplayStrategy.TASK_BALANCED)
        replay = ImportanceWeightedReplay(config=config, random_seed=42)

        for i in range(25):
            state = np.random.randn(10)
            replay.add(state, i, 1.0, state, False, "task1")

        for i in range(25):
            state = np.random.randn(10)
            replay.add(state, i, 1.0, state, False, "task2")

        batch = replay.sample_batch(20)

        assert len(batch) == 20

    def test_sample_empty_buffer(self):
        replay = ImportanceWeightedReplay(random_seed=42)

        batch = replay.sample_batch(10)

        assert len(batch) == 0

    def test_update_priorities(self):
        replay = ImportanceWeightedReplay(random_seed=42)

        for i in range(10):
            state = np.random.randn(10)
            replay.add(state, i, 1.0, state, False, "task1")

        old_priorities = replay._priorities.copy()

        replay.update_priorities([0, 1, 2], [10.0, 5.0, 1.0])

        assert not np.allclose(replay._priorities[:3], old_priorities[:3])

    def test_get_task_experiences(self):
        replay = ImportanceWeightedReplay(random_seed=42)

        for i in range(10):
            state = np.random.randn(10)
            replay.add(state, i, 1.0, state, False, "task1")

        for i in range(5):
            state = np.random.randn(10)
            replay.add(state, i, 1.0, state, False, "task2")

        task1_exps = replay.get_task_experiences("task1")
        task2_exps = replay.get_task_experiences("task2")

        assert len(task1_exps) == 10
        assert len(task2_exps) == 5

    def test_get_task_count(self):
        replay = ImportanceWeightedReplay(random_seed=42)

        for i in range(10):
            state = np.random.randn(10)
            replay.add(state, i, 1.0, state, False, "task1")

        assert replay.get_task_count("task1") == 10
        assert replay.get_task_count("task2") == 0

    def test_get_all_tasks(self):
        replay = ImportanceWeightedReplay(random_seed=42)

        for task in ["task1", "task2", "task3"]:
            state = np.random.randn(10)
            replay.add(state, 0, 1.0, state, False, task)

        tasks = replay.get_all_tasks()

        assert set(tasks) == {"task1", "task2", "task3"}

    def test_clear_task(self):
        replay = ImportanceWeightedReplay(random_seed=42)

        for i in range(10):
            state = np.random.randn(10)
            replay.add(state, i, 1.0, state, False, "task1")

        for i in range(5):
            state = np.random.randn(10)
            replay.add(state, i, 1.0, state, False, "task2")

        cleared = replay.clear_task("task1")

        assert cleared == 10
        assert len(replay) == 5
        assert replay.get_task_count("task1") == 0

    def test_statistics(self):
        replay = ImportanceWeightedReplay(random_seed=42)

        for i in range(10):
            state = np.random.randn(10)
            replay.add(state, i, 1.0, state, False, "task1")

        stats = replay.statistics()

        assert "buffer_size" in stats
        assert "max_size" in stats
        assert "num_tasks" in stats
        assert "strategy" in stats
        assert stats["buffer_size"] == 10


class TestReplayStrategy:
    """Tests for ReplayStrategy enum."""

    def test_strategies(self):
        assert ReplayStrategy.UNIFORM.value == "uniform"
        assert ReplayStrategy.PRIORITIZED.value == "prioritized"
        assert ReplayStrategy.TASK_BALANCED.value == "task_balanced"
        assert ReplayStrategy.RESERVOIR.value == "reservoir"


class TestExperience:
    """Tests for Experience dataclass."""

    def test_experience_creation(self):
        exp = Experience(
            state=np.array([1, 2, 3]),
            action=0,
            reward=1.0,
            next_state=np.array([4, 5, 6]),
            done=False,
            task_id="task1",
        )

        assert exp.reward == 1.0
        assert exp.task_id == "task1"
        assert exp.priority == 1.0
