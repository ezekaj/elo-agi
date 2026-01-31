"""Tests for problem classifier."""

import pytest
import numpy as np

from src.problem_classifier import (
    ProblemClassifier,
    ProblemClassifierConfig,
    ProblemAnalysis,
    ProblemType,
    ProblemDifficulty,
)


class TestProblemClassifierConfig:
    """Tests for ProblemClassifierConfig class."""

    def test_default_config(self):
        config = ProblemClassifierConfig()
        assert config.embedding_dim == 128
        assert config.min_confidence == 0.5

    def test_custom_config(self):
        config = ProblemClassifierConfig(embedding_dim=64, min_confidence=0.7)
        assert config.embedding_dim == 64
        assert config.min_confidence == 0.7


class TestProblemClassifier:
    """Tests for ProblemClassifier class."""

    def test_creation(self):
        classifier = ProblemClassifier(random_seed=42)
        assert classifier is not None

    def test_classify_basic(self):
        classifier = ProblemClassifier(random_seed=42)

        embedding = np.random.randn(128)
        analysis = classifier.classify(embedding)

        assert isinstance(analysis, ProblemAnalysis)
        assert analysis.problem_type in ProblemType
        assert 0 <= analysis.type_confidence <= 1
        assert 0 <= analysis.complexity <= 1

    def test_classify_with_context(self):
        classifier = ProblemClassifier(random_seed=42)

        embedding = np.random.randn(128)
        context = {
            "constraints": ["time_limit"],
            "examples": [1, 2, 3],
        }
        analysis = classifier.classify(embedding, context)

        assert analysis.features.get("has_constraints", 0) == 1.0
        assert analysis.features.get("has_examples", 0) == 1.0

    def test_classify_different_sizes(self):
        classifier = ProblemClassifier(random_seed=42)

        small_embedding = np.random.randn(32)
        analysis1 = classifier.classify(small_embedding)
        assert isinstance(analysis1, ProblemAnalysis)

        large_embedding = np.random.randn(256)
        analysis2 = classifier.classify(large_embedding)
        assert isinstance(analysis2, ProblemAnalysis)

    def test_estimate_complexity(self):
        classifier = ProblemClassifier(random_seed=42)

        embedding = np.random.randn(128)
        analysis = classifier.classify(embedding)

        complexity = classifier.estimate_complexity(analysis)
        assert 0 <= complexity <= 1

    def test_identify_subproblems(self):
        classifier = ProblemClassifier(random_seed=42)

        problem = {
            "embedding": np.random.randn(128),
            "context": {},
        }

        subproblems = classifier.identify_subproblems(problem)

        assert isinstance(subproblems, list)

    def test_difficulty_levels(self):
        config = ProblemClassifierConfig(
            complexity_threshold_easy=0.2,
            complexity_threshold_medium=0.5,
            complexity_threshold_hard=0.8,
        )
        classifier = ProblemClassifier(config=config, random_seed=42)

        embeddings = [np.random.randn(128) for _ in range(10)]

        difficulties = set()
        for emb in embeddings:
            analysis = classifier.classify(emb)
            difficulties.add(analysis.difficulty)

        assert len(difficulties) >= 1

    def test_classification_history(self):
        classifier = ProblemClassifier(random_seed=42)

        for _ in range(5):
            embedding = np.random.randn(128)
            classifier.classify(embedding)

        history = classifier.get_classification_history()
        assert len(history) == 5

        recent = classifier.get_classification_history(n=3)
        assert len(recent) == 3

    def test_statistics(self):
        classifier = ProblemClassifier(random_seed=42)

        for _ in range(10):
            embedding = np.random.randn(128)
            classifier.classify(embedding)

        stats = classifier.statistics()

        assert "total_classifications" in stats
        assert "type_distribution" in stats
        assert "avg_complexity" in stats
        assert stats["total_classifications"] == 10


class TestProblemType:
    """Tests for ProblemType enum."""

    def test_types(self):
        assert ProblemType.LOGICAL.value == "logical"
        assert ProblemType.MATHEMATICAL.value == "mathematical"
        assert ProblemType.CAUSAL.value == "causal"
        assert ProblemType.UNKNOWN.value == "unknown"


class TestProblemDifficulty:
    """Tests for ProblemDifficulty enum."""

    def test_difficulties(self):
        assert ProblemDifficulty.TRIVIAL.value == "trivial"
        assert ProblemDifficulty.EASY.value == "easy"
        assert ProblemDifficulty.MEDIUM.value == "medium"
        assert ProblemDifficulty.HARD.value == "hard"
        assert ProblemDifficulty.EXPERT.value == "expert"


class TestProblemAnalysis:
    """Tests for ProblemAnalysis dataclass."""

    def test_analysis_creation(self):
        analysis = ProblemAnalysis(
            problem_type=ProblemType.LOGICAL,
            type_confidence=0.8,
            complexity=0.5,
            difficulty=ProblemDifficulty.MEDIUM,
            features={"test": 1.0},
            subproblems=[],
            estimated_steps=10,
            requires_domain_knowledge=False,
        )

        assert analysis.problem_type == ProblemType.LOGICAL
        assert analysis.complexity == 0.5
        assert analysis.estimated_steps == 10
