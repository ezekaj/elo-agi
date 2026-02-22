"""
Module 7: Emotions and Decision-Making

Working implementation of emotional processing and moral decision-making based on neuroscience research.

Components:
- EmotionCircuit: Brain emotion network (VMPFC, Amygdala, ACC, Insula)
- DualRouteProcessor: Fast (~12ms) vs Slow (~100ms) emotion pathways
- MotivationalSystem: Goal-directed drives (approach/avoid)
- OutcomeEvaluator: Emotional reactions to outcomes
- MoralDilemmaProcessor: Deontological vs Utilitarian reasoning
- EmotionDecisionSystem: Full integrated system
"""

from .emotion_circuit import (
    VMPFC,
    Amygdala,
    ACC,
    Insula,
    EmotionCircuit,
    EmotionalEvaluation,
)
from .dual_emotion_routes import (
    ThalamusRelay,
    FastEmotionRoute,
    SlowEmotionRoute,
    DualRouteProcessor,
    EmotionRouteResponse,
)
from .motivational_states import (
    Drive,
    MotivationalSystem,
    IncentiveSalience,
)
from .emotional_states import (
    EmotionalResponse,
    OutcomeEvaluator,
    EmotionalDynamics,
    Outcome,
)
from .moral_reasoning import (
    DeontologicalSystem,
    UtilitarianSystem,
    MoralDilemmaProcessor,
    VMPFCLesionModel,
    MoralScenario,
    MoralDecision,
)
from .value_computation import (
    ValueSignal,
    OFCValueComputer,
    VMPFCIntegrator,
)
from .emotion_decision_integrator import (
    EmotionDecisionSystem,
    Situation,
    Decision,
)

__all__ = [
    # Emotion Circuit
    "VMPFC",
    "Amygdala",
    "ACC",
    "Insula",
    "EmotionCircuit",
    "EmotionalEvaluation",
    # Dual Routes
    "ThalamusRelay",
    "FastEmotionRoute",
    "SlowEmotionRoute",
    "DualRouteProcessor",
    "EmotionRouteResponse",
    # Motivational States
    "Drive",
    "MotivationalSystem",
    "IncentiveSalience",
    # Emotional States
    "EmotionalResponse",
    "OutcomeEvaluator",
    "EmotionalDynamics",
    "Outcome",
    # Moral Reasoning
    "DeontologicalSystem",
    "UtilitarianSystem",
    "MoralDilemmaProcessor",
    "VMPFCLesionModel",
    "MoralScenario",
    "MoralDecision",
    # Value Computation
    "ValueSignal",
    "OFCValueComputer",
    "VMPFCIntegrator",
    # Integration
    "EmotionDecisionSystem",
    "Situation",
    "Decision",
]
