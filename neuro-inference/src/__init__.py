"""
neuro-inference: Probabilistic and causal reasoning.

This module provides:
- Bayesian networks and belief propagation
- Causal models and intervention reasoning
- Analogical reasoning and structure mapping
"""

from .bayesian.networks import (
    BayesianNetwork,
    DiscreteNode,
    ContinuousNode,
    CPT,
    NetworkQuery,
)
from .bayesian.belief_prop import (
    BeliefPropagation,
    Message,
    FactorGraph,
    Factor,
)
from .bayesian.learning import (
    StructureLearner,
    ParameterLearner,
    BayesianScore,
)
from .causal.scm import (
    StructuralCausalModel,
    CausalVariable,
    StructuralEquation,
)
from .causal.intervention import (
    Intervention,
    InterventionEngine,
    DoOperator,
)
from .causal.counterfactual import (
    CounterfactualReasoner,
    CounterfactualQuery,
    PotentialOutcome,
)
from .analogical.mapping import (
    StructureMapper,
    Analogy,
    StructuralAlignment,
)
from .analogical.retrieval import (
    AnalogyRetriever,
    CaseLibrary,
    Case,
)
from .integration import (
    ProbabilisticReasoner,
    InferenceResult,
)

__all__ = [
    # Bayesian
    "BayesianNetwork",
    "DiscreteNode",
    "ContinuousNode",
    "CPT",
    "NetworkQuery",
    "BeliefPropagation",
    "Message",
    "FactorGraph",
    "Factor",
    "StructureLearner",
    "ParameterLearner",
    "BayesianScore",
    # Causal
    "StructuralCausalModel",
    "CausalVariable",
    "StructuralEquation",
    "Intervention",
    "InterventionEngine",
    "DoOperator",
    "CounterfactualReasoner",
    "CounterfactualQuery",
    "PotentialOutcome",
    # Analogical
    "StructureMapper",
    "Analogy",
    "StructuralAlignment",
    "AnalogyRetriever",
    "CaseLibrary",
    "Case",
    # Integration
    "ProbabilisticReasoner",
    "InferenceResult",
]
