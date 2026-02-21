"""
Neuro-Meta-Reasoning: Meta-Reasoning Orchestration

Implements meta-level reasoning control:
- Problem classification and analysis
- Reasoning style selection
- Efficiency monitoring and early termination
- Dynamic orchestration across modules
- Fallacy detection and correction
"""

try:
    from .problem_classifier import (
        ProblemClassifier,
        ProblemClassifierConfig,
        ProblemAnalysis,
        ProblemType,
        ProblemDifficulty,
    )
    from .style_selector import (
        StyleSelector,
        StyleSelectorConfig,
        StyleSelection,
        ReasoningStyle,
        StyleFeedback,
    )
    from .efficiency_monitor import (
        EfficiencyMonitor,
        EfficiencyConfig,
        ReasoningMetrics,
        EfficiencyReport,
        TerminationReason,
    )
    from .orchestrator import (
        DynamicOrchestrator,
        OrchestratorConfig,
        OrchestrationPlan,
        OrchestrationStep,
        ExecutionResult,
        PlanStatus,
        CheckpointAction,
    )
    from .fallacy_detector import (
        FallacyDetector,
        FallacyDetectorConfig,
        FallacyDetection,
        FallacyType,
        ReasoningStep,
    )
    from .integration import (
        MetaReasoningController,
        MetaReasoningConfig,
        ReasoningSession,
    )
except ImportError:
    from problem_classifier import (
        ProblemClassifier,
        ProblemClassifierConfig,
        ProblemAnalysis,
        ProblemType,
        ProblemDifficulty,
    )
    from style_selector import (
        StyleSelector,
        StyleSelectorConfig,
        StyleSelection,
        ReasoningStyle,
        StyleFeedback,
    )
    from efficiency_monitor import (
        EfficiencyMonitor,
        EfficiencyConfig,
        ReasoningMetrics,
        EfficiencyReport,
        TerminationReason,
    )
    from orchestrator import (
        DynamicOrchestrator,
        OrchestratorConfig,
        OrchestrationPlan,
        OrchestrationStep,
        ExecutionResult,
        PlanStatus,
        CheckpointAction,
    )
    from fallacy_detector import (
        FallacyDetector,
        FallacyDetectorConfig,
        FallacyDetection,
        FallacyType,
        ReasoningStep,
    )
    from integration import (
        MetaReasoningController,
        MetaReasoningConfig,
        ReasoningSession,
    )

__all__ = [
    # Problem Classifier
    "ProblemClassifier",
    "ProblemClassifierConfig",
    "ProblemAnalysis",
    "ProblemType",
    "ProblemDifficulty",
    # Style Selector
    "StyleSelector",
    "StyleSelectorConfig",
    "StyleSelection",
    "ReasoningStyle",
    "StyleFeedback",
    # Efficiency Monitor
    "EfficiencyMonitor",
    "EfficiencyConfig",
    "ReasoningMetrics",
    "EfficiencyReport",
    "TerminationReason",
    # Orchestrator
    "DynamicOrchestrator",
    "OrchestratorConfig",
    "OrchestrationPlan",
    "OrchestrationStep",
    "ExecutionResult",
    "PlanStatus",
    "CheckpointAction",
    # Fallacy Detector
    "FallacyDetector",
    "FallacyDetectorConfig",
    "FallacyDetection",
    "FallacyType",
    "ReasoningStep",
    # Integration
    "MetaReasoningController",
    "MetaReasoningConfig",
    "ReasoningSession",
]
