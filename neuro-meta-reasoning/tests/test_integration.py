"""Tests for meta-reasoning controller integration."""

import pytest
import numpy as np

from src.integration import (
    MetaReasoningController,
    MetaReasoningConfig,
    ReasoningSession,
)
from src.problem_classifier import ProblemType
from src.style_selector import ReasoningStyle
from src.fallacy_detector import ReasoningStep


class TestMetaReasoningConfig:
    """Tests for MetaReasoningConfig class."""

    def test_default_config(self):
        config = MetaReasoningConfig()
        assert config.embedding_dim == 128
        assert config.exploration_rate == 0.1
        assert config.enable_fallacy_detection

    def test_custom_config(self):
        config = MetaReasoningConfig(
            embedding_dim=64,
            time_limit_seconds=30.0,
            random_seed=42,
        )
        assert config.embedding_dim == 64
        assert config.time_limit_seconds == 30.0


class TestMetaReasoningController:
    """Tests for MetaReasoningController class."""

    def test_creation(self):
        controller = MetaReasoningController()
        assert controller is not None

    def test_creation_with_config(self):
        config = MetaReasoningConfig(random_seed=42)
        controller = MetaReasoningController(config=config)
        assert controller.config.random_seed == 42

    def test_analyze_problem(self):
        controller = MetaReasoningController()

        embedding = np.random.randn(128)
        analysis = controller.analyze_problem(embedding)

        assert analysis is not None
        assert analysis.problem_type in ProblemType

    def test_analyze_with_context(self):
        controller = MetaReasoningController()

        embedding = np.random.randn(128)
        context = {"constraints": ["time_limit"]}

        analysis = controller.analyze_problem(embedding, context)

        assert analysis.features.get("has_constraints", 0) == 1.0

    def test_select_style(self):
        controller = MetaReasoningController()

        embedding = np.random.randn(128)
        analysis = controller.analyze_problem(embedding)

        selection = controller.select_style(analysis)

        assert selection is not None
        assert selection.primary_style in ReasoningStyle

    def test_create_session(self):
        config = MetaReasoningConfig(random_seed=42)
        controller = MetaReasoningController(config=config)

        embedding = np.random.randn(128)
        session = controller.create_session(embedding)

        assert isinstance(session, ReasoningSession)
        assert session.problem_analysis is not None
        assert session.style_selection is not None
        assert session.plan is not None

    @pytest.mark.skip(reason="Requires module registration - tested in integration tests")
    def test_execute_session(self):
        config = MetaReasoningConfig(random_seed=42)
        controller = MetaReasoningController(config=config)

        embedding = np.random.randn(128)
        session = controller.create_session(embedding)

        result = controller.execute_session(session.session_id, {"data": "test"})

        assert result is not None
        assert result.steps_executed > 0

    def test_execute_nonexistent_session(self):
        controller = MetaReasoningController()

        with pytest.raises(ValueError):
            controller.execute_session("nonexistent", {})

    def test_add_reasoning_step(self):
        config = MetaReasoningConfig(random_seed=42)
        controller = MetaReasoningController(config=config)

        embedding = np.random.randn(128)
        session = controller.create_session(embedding)

        step = ReasoningStep(
            step_id="step1",
            content="Test step",
            premises=["A"],
            conclusion="B",
            evidence_used=[],
            confidence=0.8,
        )
        controller.add_reasoning_step(session.session_id, step)

        retrieved = controller.get_session(session.session_id)
        assert len(retrieved.reasoning_trace) == 1

    def test_check_for_fallacies(self):
        config = MetaReasoningConfig(random_seed=42, enable_fallacy_detection=True)
        controller = MetaReasoningController(config=config)

        embedding = np.random.randn(128)
        session = controller.create_session(embedding)

        step = ReasoningStep(
            step_id="step1",
            content="A implies A",
            premises=["A"],
            conclusion="A",
            evidence_used=[],
            confidence=0.8,
        )
        controller.add_reasoning_step(session.session_id, step)

        fallacies = controller.check_for_fallacies(session.session_id)

        assert len(fallacies) >= 1

    def test_get_corrections(self):
        config = MetaReasoningConfig(random_seed=42)
        controller = MetaReasoningController(config=config)

        embedding = np.random.randn(128)
        session = controller.create_session(embedding)

        step = ReasoningStep(
            step_id="step1",
            content="A",
            premises=["A"],
            conclusion="A",
            evidence_used=[],
            confidence=0.8,
        )
        controller.add_reasoning_step(session.session_id, step)
        controller.check_for_fallacies(session.session_id)

        corrections = controller.get_corrections(session.session_id)

        assert isinstance(corrections, list)

    def test_record_feedback(self):
        config = MetaReasoningConfig(random_seed=42)
        controller = MetaReasoningController(config=config)

        embedding = np.random.randn(128)
        session = controller.create_session(embedding)

        controller.record_feedback(
            session.session_id,
            success=True,
            efficiency=0.8,
            quality=0.9,
        )

        stats = controller._style_selector.statistics()
        assert stats["total_feedbacks"] >= 1

    def test_register_module(self):
        controller = MetaReasoningController()

        def test_module(problem, step):
            return {"progress": 0.5, "quality": 0.5}

        controller.register_module("test", test_module)

        stats = controller._orchestrator.statistics()
        assert "test" in stats["registered_modules"]

    def test_get_session(self):
        controller = MetaReasoningController()

        embedding = np.random.randn(128)
        session = controller.create_session(embedding)

        retrieved = controller.get_session(session.session_id)

        assert retrieved == session

    def test_get_nonexistent_session(self):
        controller = MetaReasoningController()

        retrieved = controller.get_session("nonexistent")

        assert retrieved is None

    def test_get_all_sessions(self):
        controller = MetaReasoningController()

        for _ in range(3):
            embedding = np.random.randn(128)
            controller.create_session(embedding)

        sessions = controller.get_all_sessions()

        assert len(sessions) == 3

    def test_get_style_rankings(self):
        controller = MetaReasoningController()

        rankings = controller.get_style_rankings(ProblemType.LOGICAL)

        assert len(rankings) == len(ReasoningStyle)

    def test_statistics(self):
        config = MetaReasoningConfig(random_seed=42)
        controller = MetaReasoningController(config=config)

        # Create session without executing (execution requires registered modules)
        embedding = np.random.randn(128)
        controller.create_session(embedding)

        stats = controller.statistics()

        assert "total_sessions" in stats
        assert "classifier" in stats
        assert "style_selector" in stats
        assert "orchestrator" in stats
        assert "fallacy_detector" in stats


class TestEndToEndMetaReasoning:
    """End-to-end tests for meta-reasoning."""

    def test_full_reasoning_session(self):
        config = MetaReasoningConfig(random_seed=42)
        controller = MetaReasoningController(config=config)

        embedding = np.random.randn(128)
        context = {"constraints": ["time_limit"], "examples": [1, 2, 3]}

        session = controller.create_session(embedding, context)

        assert session.problem_analysis.problem_type in ProblemType
        assert session.style_selection.primary_style in ReasoningStyle

        result = controller.execute_session(session.session_id, {"data": "test"})

        assert result.steps_executed > 0

        controller.record_feedback(
            session.session_id,
            success=result.success,
            efficiency=0.8,
            quality=result.final_quality,
        )

    def test_adaptive_reasoning(self):
        config = MetaReasoningConfig(random_seed=42, exploration_rate=0.0)
        controller = MetaReasoningController(config=config)

        for i in range(5):
            embedding = np.random.randn(128)
            session = controller.create_session(embedding)
            result = controller.execute_session(session.session_id, {})

            controller.record_feedback(
                session.session_id,
                success=True,
                efficiency=0.9,
                quality=0.9,
            )

        stats = controller.statistics()
        assert stats["total_sessions"] == 5

    def test_fallacy_detection_in_session(self):
        config = MetaReasoningConfig(random_seed=42, enable_fallacy_detection=True)
        controller = MetaReasoningController(config=config)

        embedding = np.random.randn(128)
        session = controller.create_session(embedding)

        steps = [
            ReasoningStep("s1", "A", ["A"], "A", [], 0.8),
        ]
        for step in steps:
            controller.add_reasoning_step(session.session_id, step)

        fallacies = controller.check_for_fallacies(session.session_id)
        corrections = controller.get_corrections(session.session_id)

        assert len(fallacies) >= 1
        assert len(corrections) >= 1
