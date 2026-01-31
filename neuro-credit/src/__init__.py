"""
neuro-credit: Unified Credit Assignment Module

Provides temporal credit assignment, blame attribution, and
learning rate modulation for AGI systems.
"""

from .eligibility_traces import (
    EligibilityTrace,
    TraceConfig,
    EligibilityTraceManager,
    TraceType,
)
from .policy_gradient import (
    Advantage,
    GAEConfig,
    CrossModulePolicyGradient,
    PolicyGradientResult,
)
from .blame_assignment import (
    Failure,
    BlameResult,
    BlameAssignment,
    CounterfactualBlame,
)
from .surprise_modulation import (
    SurpriseMetrics,
    SurpriseConfig,
    SurpriseModulatedLearning,
)
from .contribution_accounting import (
    Contribution,
    ShapleyConfig,
    ContributionAccountant,
)
from .integration import (
    CreditConfig,
    CreditAssignmentSystem,
)

__all__ = [
    # Eligibility traces
    "EligibilityTrace",
    "TraceConfig",
    "EligibilityTraceManager",
    "TraceType",
    # Policy gradient
    "Advantage",
    "GAEConfig",
    "CrossModulePolicyGradient",
    "PolicyGradientResult",
    # Blame assignment
    "Failure",
    "BlameResult",
    "BlameAssignment",
    "CounterfactualBlame",
    # Surprise modulation
    "SurpriseMetrics",
    "SurpriseConfig",
    "SurpriseModulatedLearning",
    # Contribution accounting
    "Contribution",
    "ShapleyConfig",
    "ContributionAccountant",
    # Integration
    "CreditConfig",
    "CreditAssignmentSystem",
]
