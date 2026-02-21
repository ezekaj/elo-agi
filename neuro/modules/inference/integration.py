"""
Integration: Unified probabilistic reasoning interface.

Provides a unified interface to Bayesian, causal, and analogical reasoning.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
import numpy as np


class ReasoningType(Enum):
    """Types of reasoning queries."""
    PROBABILISTIC = "probabilistic"  # P(Y | X)
    CAUSAL = "causal"                # P(Y | do(X))
    COUNTERFACTUAL = "counterfactual"  # P(Y_x | X', Y')
    ANALOGICAL = "analogical"        # Inference by analogy


@dataclass
class InferenceResult:
    """Result of a reasoning query."""
    query_type: ReasoningType
    query: str
    result: Any
    confidence: float = 1.0
    explanation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate a summary of the result."""
        lines = [
            f"Query type: {self.query_type.value}",
            f"Query: {self.query}",
            f"Result: {self.result}",
            f"Confidence: {self.confidence:.2f}",
        ]
        if self.explanation:
            lines.append(f"Explanation: {self.explanation}")
        return "\n".join(lines)


class ProbabilisticReasoner:
    """
    Unified probabilistic reasoning system.

    Integrates:
    - Bayesian networks for probabilistic inference
    - Structural causal models for causal reasoning
    - Structure mapping for analogical reasoning

    Provides a unified query interface.
    """

    def __init__(self):
        # Models
        self._bayesian_network = None
        self._causal_model = None
        self._case_library = None
        self._structure_mapper = None

        # Components (lazy initialized)
        self._belief_propagation = None
        self._intervention_engine = None
        self._counterfactual_reasoner = None
        self._analogy_retriever = None

    def set_bayesian_network(self, bn) -> None:
        """Set the Bayesian network for probabilistic queries."""
        self._bayesian_network = bn

    def set_causal_model(self, scm) -> None:
        """Set the structural causal model for causal queries."""
        self._causal_model = scm

        # Initialize causal reasoning components
        from .causal.intervention import InterventionEngine
        from .causal.counterfactual import CounterfactualReasoner

        self._intervention_engine = InterventionEngine(scm)
        self._counterfactual_reasoner = CounterfactualReasoner(scm)

    def set_case_library(self, library, mapper=None) -> None:
        """Set the case library for analogical reasoning."""
        self._case_library = library

        from .analogical.mapping import StructureMapper
        from .analogical.retrieval import AnalogyRetriever

        self._structure_mapper = mapper or StructureMapper()
        self._analogy_retriever = AnalogyRetriever(library, self._structure_mapper)

    def query(
        self,
        query_type: ReasoningType,
        query_vars: Optional[List[str]] = None,
        evidence: Optional[Dict[str, Any]] = None,
        intervention: Optional[Dict[str, Any]] = None,
        counterfactual: Optional[Dict[str, Any]] = None,
        analogy_query: Optional[Any] = None,
        **kwargs
    ) -> InferenceResult:
        """
        Execute a reasoning query.

        Args:
            query_type: Type of reasoning to perform
            query_vars: Variables to query (for probabilistic/causal)
            evidence: Observed evidence
            intervention: Intervention for causal queries (do operator)
            counterfactual: Counterfactual specification
            analogy_query: Relational structure for analogical queries
        """
        if query_type == ReasoningType.PROBABILISTIC:
            return self._probabilistic_query(query_vars, evidence, **kwargs)

        elif query_type == ReasoningType.CAUSAL:
            return self._causal_query(query_vars, intervention, evidence, **kwargs)

        elif query_type == ReasoningType.COUNTERFACTUAL:
            return self._counterfactual_query(counterfactual, evidence, **kwargs)

        elif query_type == ReasoningType.ANALOGICAL:
            return self._analogical_query(analogy_query, **kwargs)

        else:
            return InferenceResult(
                query_type=query_type,
                query="unknown",
                result=None,
                confidence=0.0,
                explanation="Unknown query type",
            )

    def _probabilistic_query(
        self,
        query_vars: Optional[List[str]],
        evidence: Optional[Dict[str, Any]],
        **kwargs
    ) -> InferenceResult:
        """Execute a probabilistic query P(Y | X)."""
        if self._bayesian_network is None:
            return InferenceResult(
                query_type=ReasoningType.PROBABILISTIC,
                query=f"P({query_vars} | {evidence})",
                result=None,
                confidence=0.0,
                explanation="No Bayesian network set",
            )

        query_vars = query_vars or []
        evidence = evidence or {}

        # Convert evidence values to strings if needed
        str_evidence = {k: str(v) for k, v in evidence.items()}

        result = self._bayesian_network.query(query_vars, str_evidence)

        # Compute confidence from entropy
        confidence = 1.0
        for var, dist in result.items():
            if dist:
                probs = list(dist.values())
                entropy = -sum(p * np.log(p + 1e-10) for p in probs if p > 0)
                max_entropy = np.log(len(probs))
                if max_entropy > 0:
                    confidence = min(confidence, 1 - entropy / max_entropy)

        query_str = f"P({', '.join(query_vars)} | {evidence})"

        return InferenceResult(
            query_type=ReasoningType.PROBABILISTIC,
            query=query_str,
            result=result,
            confidence=confidence,
            metadata={"evidence": evidence},
        )

    def _causal_query(
        self,
        query_vars: Optional[List[str]],
        intervention: Optional[Dict[str, Any]],
        evidence: Optional[Dict[str, Any]],
        **kwargs
    ) -> InferenceResult:
        """Execute a causal query P(Y | do(X))."""
        if self._causal_model is None:
            return InferenceResult(
                query_type=ReasoningType.CAUSAL,
                query=f"P({query_vars} | do({intervention}))",
                result=None,
                confidence=0.0,
                explanation="No causal model set",
            )

        intervention = intervention or {}
        query_vars = query_vars or []

        from .causal.intervention import DoOperator

        do_op = DoOperator(intervention)
        n_samples = kwargs.get("n_samples", 1000)

        results = {}
        for var in query_vars:
            dist = self._intervention_engine.interventional_distribution(
                do_op, var, n_samples
            )
            results[var] = dist

        query_str = f"P({', '.join(query_vars)} | {do_op})"

        return InferenceResult(
            query_type=ReasoningType.CAUSAL,
            query=query_str,
            result=results,
            confidence=0.9,  # Lower confidence for interventional
            metadata={"intervention": intervention, "n_samples": n_samples},
        )

    def _counterfactual_query(
        self,
        counterfactual: Optional[Dict[str, Any]],
        evidence: Optional[Dict[str, Any]],
        **kwargs
    ) -> InferenceResult:
        """Execute a counterfactual query."""
        if self._counterfactual_reasoner is None:
            return InferenceResult(
                query_type=ReasoningType.COUNTERFACTUAL,
                query=f"Counterfactual: {counterfactual}",
                result=None,
                confidence=0.0,
                explanation="No causal model set for counterfactuals",
            )

        if counterfactual is None:
            return InferenceResult(
                query_type=ReasoningType.COUNTERFACTUAL,
                query="None",
                result=None,
                confidence=0.0,
                explanation="No counterfactual specification provided",
            )

        evidence = evidence or {}

        from .causal.counterfactual import CounterfactualQuery

        cf_query = CounterfactualQuery(
            outcome=counterfactual.get("outcome", ""),
            intervention_var=counterfactual.get("intervention_var", ""),
            intervention_value=counterfactual.get("intervention_value"),
            evidence=evidence,
        )

        n_samples = kwargs.get("n_samples", 1000)
        result = self._counterfactual_reasoner.compute_counterfactual(cf_query, n_samples)

        query_str = str(cf_query)

        # Extract confidence from result
        confidence = 0.8  # Base confidence for counterfactuals
        if "mean" in result:
            # Lower variance = higher confidence
            if result.get("std", 1) < 0.5:
                confidence = 0.9

        return InferenceResult(
            query_type=ReasoningType.COUNTERFACTUAL,
            query=query_str,
            result=result,
            confidence=confidence,
            metadata={"evidence": evidence, "counterfactual": counterfactual},
        )

    def _analogical_query(
        self,
        analogy_query: Optional[Any],
        **kwargs
    ) -> InferenceResult:
        """Execute an analogical reasoning query."""
        if self._analogy_retriever is None:
            return InferenceResult(
                query_type=ReasoningType.ANALOGICAL,
                query="Analogical query",
                result=None,
                confidence=0.0,
                explanation="No case library set for analogical reasoning",
            )

        if analogy_query is None:
            return InferenceResult(
                query_type=ReasoningType.ANALOGICAL,
                query="None",
                result=None,
                confidence=0.0,
                explanation="No analogy query provided",
            )

        features = kwargs.get("features", {})
        top_k = kwargs.get("top_k", 5)

        # Retrieve similar cases
        results = self._analogy_retriever.retrieve(
            analogy_query, features, top_k
        )

        if results:
            # Create analogy with best match
            best = results[0]
            analogy = self._structure_mapper.make_analogy(
                best.case.problem, analogy_query
            )

            return InferenceResult(
                query_type=ReasoningType.ANALOGICAL,
                query=f"Analogy for {analogy_query.name}",
                result={
                    "best_match": best.case.name,
                    "similarity": best.similarity,
                    "inferences": [
                        f"{p.name}({', '.join(p.arguments)})"
                        for p in analogy.inferences
                    ],
                    "object_mappings": analogy.alignment.object_mappings,
                },
                confidence=best.similarity,
                explanation=analogy.explain(),
            )

        return InferenceResult(
            query_type=ReasoningType.ANALOGICAL,
            query=f"Analogy for {analogy_query.name if analogy_query else 'unknown'}",
            result=None,
            confidence=0.0,
            explanation="No similar cases found",
        )

    def causal_effect(
        self,
        treatment: str,
        outcome: str,
        control_value: Any = 0,
        treatment_value: Any = 1,
        n_samples: int = 1000,
    ) -> InferenceResult:
        """Compute average causal effect."""
        if self._intervention_engine is None:
            return InferenceResult(
                query_type=ReasoningType.CAUSAL,
                query=f"ACE({treatment} -> {outcome})",
                result=None,
                confidence=0.0,
                explanation="No causal model set",
            )

        ate, se = self._intervention_engine.average_treatment_effect(
            treatment, outcome, control_value, treatment_value, n_samples
        )

        return InferenceResult(
            query_type=ReasoningType.CAUSAL,
            query=f"ACE({treatment} -> {outcome})",
            result={"ate": ate, "standard_error": se},
            confidence=max(0.5, 1 - 2 * se),  # Higher SE = lower confidence
            explanation=f"Average Treatment Effect: {ate:.3f} (SE: {se:.3f})",
        )

    def explain(
        self,
        outcome_var: str,
        outcome_value: Any,
        evidence: Dict[str, Any],
    ) -> InferenceResult:
        """Generate explanation for an outcome."""
        if self._counterfactual_reasoner is None:
            return InferenceResult(
                query_type=ReasoningType.COUNTERFACTUAL,
                query=f"Explain {outcome_var}={outcome_value}",
                result=None,
                confidence=0.0,
                explanation="No causal model set for explanations",
            )

        evidence_with_outcome = {**evidence, outcome_var: outcome_value}
        explanation = self._counterfactual_reasoner.explain_outcome(
            evidence_with_outcome, outcome_var
        )

        # Format explanation
        if "explanations" in explanation:
            causes = [
                f"{e['cause']}={e['value']} (necessity: {e['probability_necessary']:.2f})"
                for e in explanation["explanations"]
            ]
            explanation_str = "Likely causes:\n" + "\n".join(f"  - {c}" for c in causes)
        else:
            explanation_str = str(explanation)

        return InferenceResult(
            query_type=ReasoningType.COUNTERFACTUAL,
            query=f"Explain {outcome_var}={outcome_value}",
            result=explanation,
            confidence=0.8,
            explanation=explanation_str,
        )

    def statistics(self) -> Dict[str, Any]:
        """Get reasoner statistics."""
        stats = {
            "has_bayesian_network": self._bayesian_network is not None,
            "has_causal_model": self._causal_model is not None,
            "has_case_library": self._case_library is not None,
        }

        if self._bayesian_network:
            stats["bn_stats"] = self._bayesian_network.statistics()

        if self._causal_model:
            stats["scm_stats"] = self._causal_model.statistics()

        if self._case_library:
            stats["case_library_size"] = self._case_library.size()

        return stats
