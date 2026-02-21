"""
Neuro-Integrate: Cross-module integration and shared representations.

Provides:
- Unified semantic embedding space
- Cross-module learning
- Conflict resolution
- Evidence accumulation
- Coherence checking
"""

from .shared_space import (
    SharedSpace,
    SemanticEmbedding,
    ProjectionLayer,
    SharedSpaceConfig,
)
from .cross_module_learning import (
    CrossModuleLearner,
    LearningSignal,
    GradientRouter,
    ModuleSynapse,
)
from .conflict_resolution import (
    ConflictResolver,
    Conflict,
    Resolution,
    ResolutionStrategy,
)
from .evidence_accumulation import (
    EvidenceAccumulator,
    Evidence,
    EvidenceSource,
    AccumulatorConfig,
)
from .coherence_checker import (
    CoherenceChecker,
    Inconsistency,
    CoherenceReport,
    BeliefNetwork,
)

__all__ = [
    # Shared Space
    'SharedSpace',
    'SemanticEmbedding',
    'ProjectionLayer',
    'SharedSpaceConfig',
    # Cross-Module Learning
    'CrossModuleLearner',
    'LearningSignal',
    'GradientRouter',
    'ModuleSynapse',
    # Conflict Resolution
    'ConflictResolver',
    'Conflict',
    'Resolution',
    'ResolutionStrategy',
    # Evidence Accumulation
    'EvidenceAccumulator',
    'Evidence',
    'EvidenceSource',
    'AccumulatorConfig',
    # Coherence Checking
    'CoherenceChecker',
    'Inconsistency',
    'CoherenceReport',
    'BeliefNetwork',
]
