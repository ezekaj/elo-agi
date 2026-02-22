"""Embodied and Grounded Cognition Module

Based on Part XIV of neuroscience research:
- Core Claims: Cognition is situated, time-pressured, action-oriented, environment is part of cognition
- Three Enactivism types: Autopoietic, Sensorimotor, Radical
- Key Principles: Closed-loop sensorimotor integration, Predictive processing, Error-driven learning
"""

from .sensorimotor import (
    SensorimotorLoop,
    MotorSensoryCoupling,
    PredictiveProcessor,
    SensorimotorParams,
)
from .grounded_concepts import GroundedConcept, ConceptGrounding, ModalityBindings, GroundingParams
from .action_simulation import MotorSimulator, ActionUnderstanding, MirrorSystem, SimulationParams
from .situated_cognition import SituatedContext, ExternalMemory, ContextualReasoner, SituatedParams
from .enactive_system import EnactiveCognitiveSystem, AutopoieticSystem, SensoriMotorEnaction

__version__ = "0.1.0"
__all__ = [
    "SensorimotorLoop",
    "MotorSensoryCoupling",
    "PredictiveProcessor",
    "SensorimotorParams",
    "GroundedConcept",
    "ConceptGrounding",
    "ModalityBindings",
    "GroundingParams",
    "MotorSimulator",
    "ActionUnderstanding",
    "MirrorSystem",
    "SimulationParams",
    "SituatedContext",
    "ExternalMemory",
    "ContextualReasoner",
    "SituatedParams",
    "EnactiveCognitiveSystem",
    "AutopoieticSystem",
    "SensoriMotorEnaction",
]
