"""
Problem Classifier for Meta-Reasoning

Classifies problems to guide reasoning strategy selection:
- Problem type identification
- Complexity estimation
- Subproblem decomposition
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np


class ProblemType(Enum):
    """Types of problems."""

    LOGICAL = "logical"
    MATHEMATICAL = "mathematical"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    LINGUISTIC = "linguistic"
    PLANNING = "planning"
    CREATIVE = "creative"
    UNKNOWN = "unknown"


class ProblemDifficulty(Enum):
    """Problem difficulty levels."""

    TRIVIAL = "trivial"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


@dataclass
class ProblemClassifierConfig:
    """Configuration for problem classification."""

    embedding_dim: int = 128
    num_problem_types: int = 10
    complexity_threshold_easy: float = 0.3
    complexity_threshold_medium: float = 0.6
    complexity_threshold_hard: float = 0.8
    min_confidence: float = 0.5


@dataclass
class ProblemAnalysis:
    """Analysis of a problem."""

    problem_type: ProblemType
    type_confidence: float
    complexity: float
    difficulty: ProblemDifficulty
    features: Dict[str, float]
    subproblems: List[Dict[str, Any]]
    estimated_steps: int
    requires_domain_knowledge: bool


class ProblemClassifier:
    """
    Classifies problems to guide reasoning strategy.

    Capabilities:
    - Identify problem type (logical, mathematical, etc.)
    - Estimate complexity and difficulty
    - Decompose into subproblems
    - Identify required resources
    """

    def __init__(
        self,
        config: Optional[ProblemClassifierConfig] = None,
        random_seed: Optional[int] = None,
    ):
        self.config = config or ProblemClassifierConfig()
        self._rng = np.random.default_rng(random_seed)

        self._type_embeddings: Dict[ProblemType, np.ndarray] = {}
        self._initialize_embeddings()

        self._classification_history: List[ProblemAnalysis] = []
        self._total_classifications = 0

    def _initialize_embeddings(self) -> None:
        """
        Initialize type embeddings with semantic features.

        Each problem type has a characteristic semantic signature based on
        its cognitive requirements. The embedding dimensions encode:
        - Logical/symbolic processing
        - Numerical/mathematical reasoning
        - Spatial/visual processing
        - Temporal/sequential reasoning
        - Causal understanding
        - Linguistic/semantic processing
        - Creative/divergent thinking
        - Memory/retrieval requirements
        """
        dim = self.config.embedding_dim

        # Semantic feature templates for each problem type
        # These encode characteristic features rather than random noise
        type_features = {
            ProblemType.LOGICAL: {
                "symbolic": 0.9,
                "formal": 0.8,
                "deductive": 0.9,
                "structured": 0.8,
                "rule_based": 0.9,
            },
            ProblemType.MATHEMATICAL: {
                "numerical": 0.9,
                "quantitative": 0.9,
                "symbolic": 0.7,
                "precise": 0.9,
                "computational": 0.8,
            },
            ProblemType.ANALOGICAL: {
                "relational": 0.9,
                "structural": 0.8,
                "transfer": 0.9,
                "similarity": 0.8,
                "mapping": 0.9,
            },
            ProblemType.CAUSAL: {
                "causal": 0.9,
                "interventional": 0.8,
                "counterfactual": 0.8,
                "mechanism": 0.9,
                "dependency": 0.8,
            },
            ProblemType.SPATIAL: {
                "spatial": 0.9,
                "visual": 0.8,
                "geometric": 0.9,
                "topological": 0.7,
                "transformation": 0.8,
            },
            ProblemType.TEMPORAL: {
                "temporal": 0.9,
                "sequential": 0.9,
                "ordering": 0.8,
                "duration": 0.7,
                "causality": 0.6,
            },
            ProblemType.LINGUISTIC: {
                "semantic": 0.9,
                "syntactic": 0.8,
                "linguistic": 0.9,
                "contextual": 0.8,
                "pragmatic": 0.7,
            },
            ProblemType.PLANNING: {
                "goal_oriented": 0.9,
                "sequential": 0.8,
                "hierarchical": 0.8,
                "constraint": 0.7,
                "resource": 0.7,
            },
            ProblemType.CREATIVE: {
                "divergent": 0.9,
                "generative": 0.9,
                "novel": 0.9,
                "associative": 0.8,
                "flexible": 0.8,
            },
            ProblemType.UNKNOWN: {
                "uncertain": 0.5,
                "ambiguous": 0.5,
                "mixed": 0.5,
                "exploratory": 0.5,
                "general": 0.5,
            },
        }

        # Create embeddings from semantic features
        for ptype in ProblemType:
            embedding = np.zeros(dim)
            features = type_features.get(ptype, {})

            # Map feature names to embedding dimensions deterministically
            list(features.keys())
            for i, (name, value) in enumerate(features.items()):
                # Spread features across embedding space
                base_idx = (hash(name) % (dim - 10)) + 5
                # Create a local pattern around the base index
                for offset in range(-2, 3):
                    idx = (base_idx + offset) % dim
                    embedding[idx] += value * (1.0 - 0.2 * abs(offset))

            # Add deterministic type-specific pattern
            type_hash = hash(ptype.value)
            pattern_indices = [(type_hash + i * 17) % dim for i in range(10)]
            for idx in pattern_indices:
                embedding[idx] += 0.5

            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 1e-8:
                embedding = embedding / norm

            self._type_embeddings[ptype] = embedding

    def classify(
        self,
        problem_embedding: np.ndarray,
        context: Optional[Dict[str, Any]] = None,
    ) -> ProblemAnalysis:
        """
        Classify a problem.

        Args:
            problem_embedding: Embedding of the problem
            context: Optional context information

        Returns:
            ProblemAnalysis with classification results
        """
        problem_embedding = np.asarray(problem_embedding)

        if problem_embedding.shape[0] != self.config.embedding_dim:
            problem_embedding = self._resize_embedding(problem_embedding)

        problem_type, type_confidence = self._identify_type(problem_embedding)

        features = self._extract_features(problem_embedding, context)

        complexity = self._compute_complexity(features)

        difficulty = self._estimate_difficulty(complexity)

        subproblems = self._decompose_problem(problem_embedding, context)

        estimated_steps = self._estimate_steps(complexity, len(subproblems))

        requires_domain = self._check_domain_knowledge(features)

        analysis = ProblemAnalysis(
            problem_type=problem_type,
            type_confidence=type_confidence,
            complexity=complexity,
            difficulty=difficulty,
            features=features,
            subproblems=subproblems,
            estimated_steps=estimated_steps,
            requires_domain_knowledge=requires_domain,
        )

        self._classification_history.append(analysis)
        self._total_classifications += 1

        return analysis

    def _resize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Resize embedding to expected dimension."""
        embedding = embedding.flatten()
        if len(embedding) >= self.config.embedding_dim:
            indices = np.linspace(0, len(embedding) - 1, self.config.embedding_dim).astype(int)
            return embedding[indices]
        else:
            result = np.zeros(self.config.embedding_dim)
            result[: len(embedding)] = embedding
            return result

    def _identify_type(
        self,
        embedding: np.ndarray,
    ) -> Tuple[ProblemType, float]:
        """Identify problem type from embedding."""
        best_type = ProblemType.UNKNOWN
        best_similarity = -1.0

        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        for ptype, type_emb in self._type_embeddings.items():
            similarity = np.dot(embedding, type_emb)
            if similarity > best_similarity:
                best_similarity = similarity
                best_type = ptype

        confidence = (best_similarity + 1) / 2

        if confidence < self.config.min_confidence:
            return ProblemType.UNKNOWN, confidence

        return best_type, confidence

    def _extract_features(
        self,
        embedding: np.ndarray,
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Extract problem features."""
        features = {}

        features["embedding_norm"] = float(np.linalg.norm(embedding))
        features["embedding_variance"] = float(np.var(embedding))
        features["embedding_entropy"] = self._compute_entropy(embedding)

        if context:
            features["has_constraints"] = 1.0 if context.get("constraints") else 0.0
            features["has_examples"] = 1.0 if context.get("examples") else 0.0
            features["has_prior_knowledge"] = 1.0 if context.get("prior_knowledge") else 0.0

        return features

    def _compute_entropy(self, embedding: np.ndarray) -> float:
        """Compute entropy of embedding."""
        probs = np.abs(embedding) / (np.sum(np.abs(embedding)) + 1e-8)
        probs = probs + 1e-10
        entropy = -np.sum(probs * np.log(probs))
        max_entropy = np.log(len(embedding))
        return float(entropy / (max_entropy + 1e-8))

    def estimate_complexity(self, analysis: ProblemAnalysis) -> float:
        """
        Estimate problem complexity.

        Args:
            analysis: Problem analysis

        Returns:
            Complexity score [0, 1]
        """
        return analysis.complexity

    def _compute_complexity(self, features: Dict[str, float]) -> float:
        """Compute complexity from features."""
        complexity = 0.0

        complexity += features.get("embedding_entropy", 0.5) * 0.3
        complexity += min(features.get("embedding_variance", 0.5), 1.0) * 0.3

        if features.get("has_constraints", 0):
            complexity += 0.15
        if not features.get("has_examples", 0):
            complexity += 0.1
        if features.get("has_prior_knowledge", 0):
            complexity -= 0.05

        return float(np.clip(complexity, 0, 1))

    def _estimate_difficulty(self, complexity: float) -> ProblemDifficulty:
        """Estimate difficulty from complexity."""
        if complexity < self.config.complexity_threshold_easy:
            return ProblemDifficulty.EASY
        elif complexity < self.config.complexity_threshold_medium:
            return ProblemDifficulty.MEDIUM
        elif complexity < self.config.complexity_threshold_hard:
            return ProblemDifficulty.HARD
        else:
            return ProblemDifficulty.EXPERT

    def identify_subproblems(
        self,
        problem: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Identify subproblems in a problem.

        Args:
            problem: Problem description

        Returns:
            List of subproblem dictionaries
        """
        embedding = problem.get("embedding", np.zeros(self.config.embedding_dim))
        context = problem.get("context", {})
        return self._decompose_problem(np.asarray(embedding), context)

    def _decompose_problem(
        self,
        embedding: np.ndarray,
        context: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Decompose problem into subproblems."""
        subproblems = []

        complexity = self._compute_complexity(self._extract_features(embedding, context))

        if complexity < 0.3:
            return subproblems

        num_subproblems = int(complexity * 5) + 1
        num_subproblems = min(num_subproblems, 5)

        chunk_size = len(embedding) // num_subproblems

        for i in range(num_subproblems):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < num_subproblems - 1 else len(embedding)

            sub_embedding = np.zeros_like(embedding)
            sub_embedding[start_idx:end_idx] = embedding[start_idx:end_idx]

            subproblem = {
                "index": i,
                "embedding": sub_embedding,
                "estimated_complexity": complexity / num_subproblems,
                "dependencies": list(range(i)) if i > 0 else [],
            }
            subproblems.append(subproblem)

        return subproblems

    def _estimate_steps(self, complexity: float, num_subproblems: int) -> int:
        """Estimate number of reasoning steps."""
        base_steps = int(complexity * 10) + 1

        if num_subproblems > 1:
            base_steps *= num_subproblems

        return min(base_steps, 50)

    def _check_domain_knowledge(self, features: Dict[str, float]) -> bool:
        """Check if domain knowledge is required."""
        if features.get("has_prior_knowledge", 0) > 0:
            return True

        if features.get("embedding_entropy", 0.5) > 0.7:
            return True

        return False

    def get_classification_history(
        self,
        n: Optional[int] = None,
    ) -> List[ProblemAnalysis]:
        """Get classification history."""
        if n is not None:
            return self._classification_history[-n:]
        return list(self._classification_history)

    def statistics(self) -> Dict[str, Any]:
        """Get classifier statistics."""
        if not self._classification_history:
            return {
                "total_classifications": 0,
                "type_distribution": {},
                "avg_complexity": 0.0,
                "avg_confidence": 0.0,
            }

        type_counts: Dict[str, int] = {}
        complexities = []
        confidences = []

        for analysis in self._classification_history:
            ptype = analysis.problem_type.value
            type_counts[ptype] = type_counts.get(ptype, 0) + 1
            complexities.append(analysis.complexity)
            confidences.append(analysis.type_confidence)

        return {
            "total_classifications": self._total_classifications,
            "type_distribution": type_counts,
            "avg_complexity": float(np.mean(complexities)),
            "avg_confidence": float(np.mean(confidences)),
            "difficulty_distribution": self._get_difficulty_distribution(),
        }

    def _get_difficulty_distribution(self) -> Dict[str, int]:
        """Get distribution of difficulties."""
        dist: Dict[str, int] = {}
        for analysis in self._classification_history:
            diff = analysis.difficulty.value
            dist[diff] = dist.get(diff, 0) + 1
        return dist
