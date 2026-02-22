"""Tests for meta-learning controller."""

import pytest
from neuro.modules.m05_sleep_consolidation.meta_learning import (
    MetaLearningController,
    LearningCurve,
    ReplayWeights,
    MemoryType,
)


class TestLearningCurve:
    """Tests for LearningCurve."""

    def test_initialization(self):
        """Test learning curve initialization."""
        curve = LearningCurve(
            memory_id="test_mem",
            memory_type=MemoryType.EPISODIC,
        )
        assert curve.memory_id == "test_mem"
        assert curve.memory_type == MemoryType.EPISODIC
        assert curve.learning_rate == 0.1
        assert not curve.is_consolidated

    def test_add_consolidation_step(self):
        """Test adding consolidation steps."""
        curve = LearningCurve(
            memory_id="test_mem",
            memory_type=MemoryType.EPISODIC,
        )
        curve.add_consolidation_step(0.3, 2)
        curve.add_consolidation_step(0.5, 3)

        assert len(curve.consolidation_history) == 2
        assert curve.consolidation_history == [0.3, 0.5]
        assert curve.total_replays == 5

    def test_learning_velocity(self):
        """Test learning velocity computation."""
        curve = LearningCurve(
            memory_id="test_mem",
            memory_type=MemoryType.EPISODIC,
        )
        curve.add_consolidation_step(0.1, 1)
        curve.add_consolidation_step(0.3, 1)
        curve.add_consolidation_step(0.5, 1)

        velocity = curve.get_learning_velocity()
        assert velocity > 0  # Positive velocity means learning

    def test_learning_velocity_no_history(self):
        """Test learning velocity with no history."""
        curve = LearningCurve(
            memory_id="test_mem",
            memory_type=MemoryType.EPISODIC,
        )
        assert curve.get_learning_velocity() == 0.0

    def test_efficiency(self):
        """Test efficiency computation."""
        curve = LearningCurve(
            memory_id="test_mem",
            memory_type=MemoryType.EPISODIC,
        )
        curve.add_consolidation_step(0.0, 0)
        curve.add_consolidation_step(0.5, 5)

        efficiency = curve.get_efficiency()
        assert efficiency == pytest.approx(0.1, abs=0.01)


class TestReplayWeights:
    """Tests for ReplayWeights."""

    def test_default_weights(self):
        """Test default weights sum to 1."""
        weights = ReplayWeights()
        total = weights.recency + weights.emotional_salience + weights.incompleteness
        assert total == pytest.approx(1.0, abs=0.01)

    def test_normalize(self):
        """Test weight normalization."""
        weights = ReplayWeights(recency=2.0, emotional_salience=2.0, incompleteness=2.0)
        normalized = weights.normalize()
        total = normalized.recency + normalized.emotional_salience + normalized.incompleteness
        assert total == pytest.approx(1.0, abs=0.01)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        weights = ReplayWeights()
        d = weights.to_dict()
        assert "recency" in d
        assert "emotional_salience" in d
        assert "incompleteness" in d


class TestMetaLearningController:
    """Tests for MetaLearningController."""

    @pytest.fixture
    def controller(self):
        return MetaLearningController(random_seed=42)

    def test_initialization(self, controller):
        """Test controller initialization."""
        assert controller.weights is not None
        stats = controller.statistics()
        assert stats["total_memories"] == 0

    def test_register_memory(self, controller):
        """Test memory registration."""
        curve = controller.register_memory(
            memory_id="mem1",
            memory_type=MemoryType.EPISODIC,
            initial_strength=0.2,
        )
        assert curve.memory_id == "mem1"
        assert len(curve.consolidation_history) == 1

    def test_get_learning_curve(self, controller):
        """Test getting learning curve."""
        controller.register_memory("mem1", MemoryType.EPISODIC)
        curve = controller.get_learning_curve("mem1")
        assert curve is not None
        assert curve.memory_id == "mem1"

    def test_track_consolidation_success(self, controller):
        """Test tracking consolidation."""
        controller.register_memory("mem1", MemoryType.EPISODIC)
        outcome = controller.track_consolidation_success(
            memory_id="mem1",
            before_strength=0.3,
            after_strength=0.5,
            replays_used=3,
        )
        assert outcome.success
        assert outcome.improvement == pytest.approx(0.2, abs=0.01)

    def test_track_failed_consolidation(self, controller):
        """Test tracking failed consolidation."""
        controller.register_memory("mem1", MemoryType.EPISODIC)
        outcome = controller.track_consolidation_success(
            memory_id="mem1",
            before_strength=0.5,
            after_strength=0.5,
            replays_used=5,
        )
        assert not outcome.success

    def test_predict_optimal_replays(self, controller):
        """Test predicting optimal replays."""
        controller.register_memory("mem1", MemoryType.EPISODIC, initial_strength=0.1)
        replays = controller.predict_optimal_replays("mem1")
        assert replays > 0

    def test_predict_optimal_replays_consolidated(self, controller):
        """Test prediction for consolidated memory."""
        controller.register_memory("mem1", MemoryType.EPISODIC, initial_strength=0.95)
        curve = controller.get_learning_curve("mem1")
        curve.is_consolidated = True
        replays = controller.predict_optimal_replays("mem1")
        assert replays == 0

    def test_update_learning_rates(self, controller):
        """Test learning rate updates."""
        new_rate = controller.update_learning_rates(MemoryType.EPISODIC, 0.9)
        assert new_rate > 0
        assert new_rate == controller.get_learning_rate(MemoryType.EPISODIC)

    def test_compute_priority_score(self, controller):
        """Test priority score computation."""
        controller.register_memory("mem1", MemoryType.EPISODIC)
        score = controller.compute_priority_score(
            memory_id="mem1",
            recency_score=0.8,
            emotional_score=0.5,
            incompleteness_score=0.7,
        )
        assert 0 <= score <= 1

    def test_get_memories_needing_replay(self, controller):
        """Test getting memories needing replay."""
        controller.register_memory("mem1", MemoryType.EPISODIC, initial_strength=0.2)
        controller.register_memory("mem2", MemoryType.SEMANTIC, initial_strength=0.4)
        controller.register_memory("mem3", MemoryType.EMOTIONAL, initial_strength=0.95)

        curve3 = controller.get_learning_curve("mem3")
        curve3.is_consolidated = True

        needs_replay = controller.get_memories_needing_replay()
        memory_ids = [m_id for m_id, _ in needs_replay]
        assert "mem1" in memory_ids or "mem2" in memory_ids
        assert "mem3" not in memory_ids

    def test_adapt_replay_weights_insufficient_samples(self, controller):
        """Test adaptation with insufficient samples."""
        weights_before = controller.weights.to_dict()
        controller.adapt_replay_weights()
        weights_after = controller.weights.to_dict()
        assert weights_before == weights_after

    def test_get_type_success_rate(self, controller):
        """Test getting type success rate."""
        rate = controller.get_type_success_rate(MemoryType.EPISODIC)
        assert 0 <= rate <= 1

    def test_statistics(self, controller):
        """Test statistics generation."""
        controller.register_memory("mem1", MemoryType.EPISODIC)
        controller.track_consolidation_success("mem1", 0.3, 0.5, 2)

        stats = controller.statistics()
        assert stats["total_memories"] == 1
        assert stats["n_outcomes"] == 1
        assert "current_weights" in stats
