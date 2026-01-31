"""
Neuro Module 06: Motivation - Why Humans Act

Implements the motivation systems that drive behavior:
- Intrinsic motivation (action-state path entropy maximization)
- Dopamine system (prediction error, incentive salience, benefit/cost)
- Curiosity and exploration (information-seeking drives)
- Homeostatic regulation (internal state-dependent valuation)

Key insight: Humans are NOT primarily reward-maximizers. They maximize
action-state path entropy - pursuing diverse experiences and possibilities.
"""

from .intrinsic_motivation import (
    PathEntropyMaximizer,
    PossibilitySpace,
    IntrinsicDrive,
    ActionDiversityTracker
)
from .dopamine_system import (
    DopamineSignal,
    PredictionErrorComputer,
    IncentiveSalience,
    BenefitCostEvaluator,
    DopamineSystem
)
from .curiosity_drive import (
    CuriosityModule,
    NoveltyDetector,
    InformationValue,
    ExplorationController
)
from .homeostatic_regulation import (
    HomeostaticState,
    NeedBasedValuation,
    InternalStateTracker
)
from .effort_valuation import (
    EffortCostModel,
    ParadoxicalEffort,
    MotivationalTransform
)

__version__ = "0.1.0"
__all__ = [
    "PathEntropyMaximizer",
    "PossibilitySpace",
    "IntrinsicDrive",
    "ActionDiversityTracker",
    "DopamineSignal",
    "PredictionErrorComputer",
    "IncentiveSalience",
    "BenefitCostEvaluator",
    "DopamineSystem",
    "CuriosityModule",
    "NoveltyDetector",
    "InformationValue",
    "ExplorationController",
    "HomeostaticState",
    "NeedBasedValuation",
    "InternalStateTracker",
    "EffortCostModel",
    "ParadoxicalEffort",
    "MotivationalTransform",
]
