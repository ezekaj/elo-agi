"""
Module 02: Dual-Process Architecture

Implementation of the brain's dual-process system based on neuroscience research:
- System 1: Fast, parallel, intuitive processing
- System 2: Slow, serial, deliberate reasoning
- HPC-PFC Complex: Compositional thinking
- Logic Network: Abstract formal reasoning
"""

from .system1 import PatternRecognition, HabitExecutor, EmotionalValuation
from .system2 import WorkingMemory, CognitiveControl, RelationalReasoning
from .hpc_pfc_complex import Hippocampus, PrefrontalCortex, HPCPFCComplex
from .logic_network import LogicNetwork
from .dual_process_controller import DualProcessController

__all__ = [
    "PatternRecognition",
    "HabitExecutor",
    "EmotionalValuation",
    "WorkingMemory",
    "CognitiveControl",
    "RelationalReasoning",
    "Hippocampus",
    "PrefrontalCortex",
    "HPCPFCComplex",
    "LogicNetwork",
    "DualProcessController",
]
