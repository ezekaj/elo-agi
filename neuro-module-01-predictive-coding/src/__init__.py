"""
Neuro Module 01: Mathematical Foundations of Thought

Implements the brain's core prediction machinery:
- Hierarchical predictive coding (top-down predictions, bottom-up errors)
- Precision-weighted learning (confidence-based error weighting)
- Geometric cognitive manifold (states as points, thinking as gradient flow)
"""

from .predictive_hierarchy import PredictiveLayer, PredictiveHierarchy
from .precision_weighting import PrecisionWeightedError, AdaptivePrecision
from .cognitive_manifold import CognitiveState, CognitiveManifold, DualProcess
from .temporal_dynamics import TemporalLayer, TemporalHierarchy
from .omission_detector import OmissionDetector

__version__ = "0.1.0"
__all__ = [
    "PredictiveLayer",
    "PredictiveHierarchy",
    "PrecisionWeightedError",
    "AdaptivePrecision",
    "CognitiveState",
    "CognitiveManifold",
    "DualProcess",
    "TemporalLayer",
    "TemporalHierarchy",
    "OmissionDetector",
]
