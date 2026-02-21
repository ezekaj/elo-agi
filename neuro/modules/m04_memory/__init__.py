"""
Module 4: Memory Systems

Complete memory architecture implementing the multi-store model:
- Sensory Memory (iconic/echoic buffers)
- Working Memory (limited capacity, active manipulation)
- Long-Term Memory (episodic, semantic, procedural)
- Engram Formation (physical memory traces with Hebbian learning)
- Memory Processes (encoding, consolidation, retrieval, reconsolidation, forgetting)
"""

from .sensory_memory import IconicBuffer, EchoicBuffer
from .working_memory import WorkingMemory, MemorySlot
from .long_term_memory import EpisodicMemory, SemanticMemory, ProceduralMemory
from .engram import Engram, Neuron
from .memory_processes import (
    MemoryEncoder,
    MemoryConsolidator,
    MemoryRetriever,
    MemoryReconsolidator,
    Forgetter
)
from .memory_controller import MemoryController

__all__ = [
    'IconicBuffer',
    'EchoicBuffer',
    'WorkingMemory',
    'MemorySlot',
    'EpisodicMemory',
    'SemanticMemory',
    'ProceduralMemory',
    'Engram',
    'Neuron',
    'MemoryEncoder',
    'MemoryConsolidator',
    'MemoryRetriever',
    'MemoryReconsolidator',
    'Forgetter',
    'MemoryController',
]
