"""
Neuro-Continual: Continual Learning Controller

Implements lifelong learning capabilities:
- Task inference and boundary detection
- Selective consolidation based on performance
- Catastrophic forgetting prevention (EWC, PackNet, SI)
- Importance-weighted experience replay
- Capability tracking and regression detection
"""

from .task_inference import (
    TaskInference,
    TaskInferenceConfig,
    TaskInfo,
    TaskChangeMethod,
)
from .selective_consolidation import (
    SelectiveConsolidation,
    ConsolidationConfig,
    ConsolidationStrategy,
    ConsolidationPlan,
    PerformanceRecord,
)
from .forgetting_prevention import (
    CatastrophicForgettingPrevention,
    ForgettingPreventionConfig,
    ForgettingPreventionMethod,
    TaskMemory,
)
from .experience_replay import (
    ImportanceWeightedReplay,
    ReplayConfig,
    ReplayStrategy,
    Experience,
)
from .capability_tracking import (
    CapabilityTracker,
    CapabilityConfig,
    CapabilityStatus,
    CapabilityMetric,
    CapabilityRecord,
    InterferenceReport,
)
from .integration import (
    ContinualLearningController,
    ContinualLearningConfig,
)

__all__ = [
    # Task Inference
    "TaskInference",
    "TaskInferenceConfig",
    "TaskInfo",
    "TaskChangeMethod",
    # Selective Consolidation
    "SelectiveConsolidation",
    "ConsolidationConfig",
    "ConsolidationStrategy",
    "ConsolidationPlan",
    "PerformanceRecord",
    # Forgetting Prevention
    "CatastrophicForgettingPrevention",
    "ForgettingPreventionConfig",
    "ForgettingPreventionMethod",
    "TaskMemory",
    # Experience Replay
    "ImportanceWeightedReplay",
    "ReplayConfig",
    "ReplayStrategy",
    "Experience",
    # Capability Tracking
    "CapabilityTracker",
    "CapabilityConfig",
    "CapabilityStatus",
    "CapabilityMetric",
    "CapabilityRecord",
    "InterferenceReport",
    # Integration
    "ContinualLearningController",
    "ContinualLearningConfig",
]
