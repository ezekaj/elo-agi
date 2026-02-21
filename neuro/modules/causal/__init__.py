"""
Neuro-Causal: Advanced causal reasoning for AGI.

Extends causal inference with:
- Differentiable Structural Causal Models (PyTorch-based)
- Nested and contrastive counterfactuals
- Causal structure learning (PC algorithm)
- Causal representation learning
- Integration with active inference (FEP)
"""

from .differentiable_scm import (
    CausalMechanism,
    DifferentiableSCM,
)
from .counterfactual import (
    NestedCounterfactual,
    ContrastiveExplanation,
)
from .causal_discovery import (
    CausalDiscovery,
    ConditionalIndependenceTest,
)
from .causal_representation import (
    CausalRepresentationLearner,
    CausalEncoder,
    CausalDecoder,
)
from .active_inference import (
    CausalActiveInference,
    CausalBelief,
)
from .inference_adapter import (
    InferenceSCMAdapter,
    CausalInferenceEnhanced,
    AdapterConfig,
)

__all__ = [
    # Differentiable SCM
    "CausalMechanism",
    "DifferentiableSCM",
    # Counterfactuals
    "NestedCounterfactual",
    "ContrastiveExplanation",
    # Discovery
    "CausalDiscovery",
    "ConditionalIndependenceTest",
    # Representation
    "CausalRepresentationLearner",
    "CausalEncoder",
    "CausalDecoder",
    # Active Inference
    "CausalActiveInference",
    "CausalBelief",
    # Inference Adapter
    "InferenceSCMAdapter",
    "CausalInferenceEnhanced",
    "AdapterConfig",
]
