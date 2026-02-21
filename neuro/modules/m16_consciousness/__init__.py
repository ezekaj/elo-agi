"""Consciousness and Self-Awareness Module

Based on Part XVI of neuroscience research:
- Self-Awareness Hierarchy: Minimal (agency/ownership), Narrative (autobiographical), Metacognitive (monitoring)
- Neural Correlates: Premotor/parietal (minimal), mPFC/hippocampus (narrative), Anterior PFC (metacognitive)
- Key Insight: Metacognition may play constitutive role in conscious awareness
"""

from .minimal_self import MinimalSelf, AgencyDetector, OwnershipProcessor, MinimalSelfParams
from .narrative_self import NarrativeSelf, AutobiographicalMemory, SelfConcept, NarrativeParams
from .metacognition import MetacognitiveSystem, ConfidenceEstimator, PerformanceMonitor, MetaParams
from .introspection import IntrospectionSystem, MentalStateAccessor, SelfReflection
from .consciousness_network import ConsciousnessNetwork, GlobalWorkspace

__version__ = "0.1.0"
__all__ = [
    "MinimalSelf", "AgencyDetector", "OwnershipProcessor", "MinimalSelfParams",
    "NarrativeSelf", "AutobiographicalMemory", "SelfConcept", "NarrativeParams",
    "MetacognitiveSystem", "ConfidenceEstimator", "PerformanceMonitor", "MetaParams",
    "IntrospectionSystem", "MentalStateAccessor", "SelfReflection",
    "ConsciousnessNetwork", "GlobalWorkspace"
]
