"""Executive Function and Cognitive Control Module

Based on Part XIII of neuroscience research:
- Components: Inhibition (right inferior frontal), Working memory (DLPFC), Cognitive flexibility (lateral PFC)
- Hot vs Cold EFs: Cold = lateral PFC (abstract, logical), Hot = orbital/medial PFC (emotionally-laden)
- Key finding: Working memory and cognitive flexibility share cortical substrate
"""

from .inhibition import InhibitionSystem, ResponseInhibitor, ImpulseController, InhibitionParams
from .working_memory import WorkingMemory, DLPFCNetwork, CapacityLimitedStore, WMParams
from .cognitive_flexibility import CognitiveFlexibility, TaskSwitcher, SetShifter, FlexibilityParams
from .executive_network import ExecutiveNetwork, PFCController, ConflictMonitor
from .hot_cold_ef import HotExecutiveFunction, ColdExecutiveFunction, EmotionalRegulator

__version__ = "0.1.0"
__all__ = [
    "InhibitionSystem",
    "ResponseInhibitor",
    "ImpulseController",
    "InhibitionParams",
    "WorkingMemory",
    "DLPFCNetwork",
    "CapacityLimitedStore",
    "WMParams",
    "CognitiveFlexibility",
    "TaskSwitcher",
    "SetShifter",
    "FlexibilityParams",
    "ExecutiveNetwork",
    "PFCController",
    "ConflictMonitor",
    "HotExecutiveFunction",
    "ColdExecutiveFunction",
    "EmotionalRegulator",
]
