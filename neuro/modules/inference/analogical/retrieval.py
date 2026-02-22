"""
Analogical Retrieval: Finding relevant analogies from memory.

Implements case-based reasoning with analogical retrieval.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
import numpy as np

from .mapping import (
    RelationalStructure,
    StructureMapper,
    StructuralAlignment,
    Analogy,
)


@dataclass
class Case:
    """
    A case in the case library.

    Contains a problem situation, solution, and outcome.
    """

    name: str
    problem: RelationalStructure
    solution: Optional[Any] = None
    outcome: Optional[Any] = None
    features: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None

    def __hash__(self):
        return hash(self.name)


@dataclass
class RetrievalResult:
    """Result of case retrieval."""

    case: Case
    similarity: float
    alignment: Optional[StructuralAlignment] = None
    feature_match: float = 0.0
    structural_match: float = 0.0


class CaseLibrary:
    """
    Library of cases for analogical reasoning.

    Stores cases and supports efficient retrieval.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
    ):
        self.embedding_dim = embedding_dim
        self._cases: Dict[str, Case] = {}
        self._feature_index: Dict[str, Set[str]] = {}  # feature -> case names

    def add_case(self, case: Case) -> None:
        """Add a case to the library."""
        self._cases[case.name] = case

        # Index by features
        for feature, value in case.features.items():
            key = f"{feature}:{value}"
            if key not in self._feature_index:
                self._feature_index[key] = set()
            self._feature_index[key].add(case.name)

        # Generate embedding if not provided
        if case.embedding is None:
            case.embedding = self._generate_embedding(case)

    def get_case(self, name: str) -> Optional[Case]:
        """Get a case by name."""
        return self._cases.get(name)

    def remove_case(self, name: str) -> bool:
        """Remove a case from the library."""
        if name in self._cases:
            case = self._cases[name]

            # Remove from feature index
            for feature, value in case.features.items():
                key = f"{feature}:{value}"
                if key in self._feature_index:
                    self._feature_index[key].discard(name)

            del self._cases[name]
            return True
        return False

    def _generate_embedding(self, case: Case) -> np.ndarray:
        """Generate a simple embedding for a case."""
        # Hash-based embedding (simple approximation)
        embedding = np.zeros(self.embedding_dim)

        # Encode features
        for i, (feature, value) in enumerate(case.features.items()):
            idx = hash(f"{feature}:{value}") % self.embedding_dim
            embedding[idx] += 1.0

        # Encode structure
        for pred in case.problem.predicates:
            idx = hash(pred.name) % self.embedding_dim
            embedding[idx] += 0.5 * pred.order.value

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm

        return embedding

    def find_by_features(
        self,
        features: Dict[str, Any],
        min_match: int = 1,
    ) -> List[str]:
        """Find cases matching features."""
        candidates = set()

        for feature, value in features.items():
            key = f"{feature}:{value}"
            if key in self._feature_index:
                if not candidates:
                    candidates = self._feature_index[key].copy()
                else:
                    candidates &= self._feature_index[key]

        return list(candidates)

    def all_cases(self) -> List[Case]:
        """Get all cases."""
        return list(self._cases.values())

    def size(self) -> int:
        """Get number of cases."""
        return len(self._cases)


class AnalogyRetriever:
    """
    Retrieves relevant analogies from a case library.

    Supports:
    - Feature-based retrieval (fast, surface similarity)
    - Structure-based retrieval (slow, deep similarity)
    - Hybrid retrieval (best of both)
    """

    def __init__(
        self,
        library: CaseLibrary,
        mapper: Optional[StructureMapper] = None,
    ):
        self.library = library
        self.mapper = mapper or StructureMapper()

    def retrieve(
        self,
        query: RelationalStructure,
        query_features: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        method: str = "hybrid",
    ) -> List[RetrievalResult]:
        """
        Retrieve most relevant cases for a query.

        Methods:
        - "feature": Fast retrieval based on surface features
        - "structural": Deep retrieval based on relational structure
        - "hybrid": Combine both approaches
        """
        query_features = query_features or {}

        if method == "feature":
            return self._feature_retrieval(query, query_features, top_k)
        elif method == "structural":
            return self._structural_retrieval(query, top_k)
        else:
            return self._hybrid_retrieval(query, query_features, top_k)

    def _feature_retrieval(
        self,
        query: RelationalStructure,
        query_features: Dict[str, Any],
        top_k: int,
    ) -> List[RetrievalResult]:
        """Feature-based retrieval using surface similarity."""
        results = []

        for case in self.library.all_cases():
            # Compute feature similarity
            feature_sim = self._feature_similarity(query_features, case.features)

            # Basic structural similarity (predicate overlap)
            query_preds = {p.name for p in query.predicates}
            case_preds = {p.name for p in case.problem.predicates}
            pred_overlap = len(query_preds & case_preds) / max(len(query_preds | case_preds), 1)

            similarity = 0.7 * feature_sim + 0.3 * pred_overlap

            results.append(
                RetrievalResult(
                    case=case,
                    similarity=similarity,
                    feature_match=feature_sim,
                )
            )

        # Sort by similarity
        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:top_k]

    def _structural_retrieval(
        self,
        query: RelationalStructure,
        top_k: int,
    ) -> List[RetrievalResult]:
        """Structure-based retrieval using SME."""
        results = []

        for case in self.library.all_cases():
            # Compute structural alignment
            alignment = self.mapper.map(case.problem, query)

            # Use alignment score as similarity
            # Normalize by max possible score
            max_score = len(case.problem.predicates) * 3 * self.mapper.systematicity_weight
            similarity = alignment.score / max(max_score, 1)

            results.append(
                RetrievalResult(
                    case=case,
                    similarity=similarity,
                    alignment=alignment,
                    structural_match=similarity,
                )
            )

        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:top_k]

    def _hybrid_retrieval(
        self,
        query: RelationalStructure,
        query_features: Dict[str, Any],
        top_k: int,
    ) -> List[RetrievalResult]:
        """
        Hybrid retrieval: feature pre-filtering + structural re-ranking.

        Two-stage approach for efficiency.
        """
        # Stage 1: Feature-based candidate selection
        candidates = self._feature_retrieval(query, query_features, top_k * 3)

        if not candidates:
            return []

        # Stage 2: Structural re-ranking of top candidates
        results = []
        for result in candidates:
            alignment = self.mapper.map(result.case.problem, query)

            # Normalize structural score
            max_score = len(result.case.problem.predicates) * 3 * self.mapper.systematicity_weight
            structural_sim = alignment.score / max(max_score, 1)

            # Combine feature and structural similarity
            combined = 0.4 * result.feature_match + 0.6 * structural_sim

            results.append(
                RetrievalResult(
                    case=result.case,
                    similarity=combined,
                    alignment=alignment,
                    feature_match=result.feature_match,
                    structural_match=structural_sim,
                )
            )

        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:top_k]

    def _feature_similarity(
        self,
        query_features: Dict[str, Any],
        case_features: Dict[str, Any],
    ) -> float:
        """Compute feature similarity between query and case."""
        if not query_features and not case_features:
            return 1.0
        if not query_features or not case_features:
            return 0.0

        shared_keys = set(query_features.keys()) & set(case_features.keys())
        all_keys = set(query_features.keys()) | set(case_features.keys())

        if not all_keys:
            return 0.0

        # Count matches
        matches = sum(1 for k in shared_keys if query_features[k] == case_features[k])

        return matches / len(all_keys)

    def retrieve_and_adapt(
        self,
        query: RelationalStructure,
        query_features: Optional[Dict[str, Any]] = None,
    ) -> Optional[Analogy]:
        """
        Retrieve best matching case and create analogy with adaptations.

        Returns analogy with inferences (adaptations) for the query.
        """
        results = self.retrieve(query, query_features, top_k=1)

        if not results:
            return None

        best = results[0]

        # Create full analogy with inferences
        analogy = self.mapper.make_analogy(best.case.problem, query)

        return analogy

    def explain_retrieval(
        self,
        result: RetrievalResult,
    ) -> Dict[str, Any]:
        """Generate explanation for why a case was retrieved."""
        explanation = {
            "case_name": result.case.name,
            "similarity": result.similarity,
            "feature_match": result.feature_match,
            "structural_match": result.structural_match,
        }

        if result.alignment:
            explanation["object_mappings"] = result.alignment.object_mappings
            explanation["matched_predicates"] = [
                (s.name, t.name) for s, t in result.alignment.matched_predicates
            ]

        return explanation
