"""
Neuro System: Unified Cognitive Architecture

Wires all 20 neuro-modules into a single cognitive system with
active inference control for perception-action loops.
"""

from .config import SystemConfig, ModuleConfig
from .module_loader import ModuleLoader, LoadedModule
from .sensory_interface import SensoryInterface, SensoryInput, InputType
from .motor_interface import MotorInterface, MotorOutput, OutputType
from .active_inference import ActiveInferenceController, Policy, EFEResult
from .cognitive_core import CognitiveCore, CognitiveState, CycleResult

__all__ = [
    "SystemConfig",
    "ModuleConfig",
    "ModuleLoader",
    "LoadedModule",
    "SensoryInterface",
    "SensoryInput",
    "InputType",
    "MotorInterface",
    "MotorOutput",
    "OutputType",
    "ActiveInferenceController",
    "Policy",
    "EFEResult",
    "CognitiveCore",
    "CognitiveState",
    "CycleResult",
]
