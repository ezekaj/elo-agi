"""
Neuro Module 00: Integration Layer (Global Workspace)

This module implements a Global Workspace Theory-based integration system
that connects all 16 cognitive modules through a shared workspace with
attention-based competition and broadcast mechanisms.

Based on:
- Baars' Global Workspace Theory (1988)
- Dehaene's Global Neuronal Workspace (2001)
- arXiv:2103.01197 - Coordination Among Neural Modules
- arXiv:2012.10390 - Deep Learning and Global Workspace Theory
"""

from .module_interface import (
    CognitiveModule,
    ModuleProposal,
    ModuleState,
    ModuleParams,
)
from .global_workspace import (
    GlobalWorkspace,
    WorkspaceParams,
    WorkspaceState,
)
from .attention_competition import (
    AttentionCompetition,
    CompetitionParams,
    CompetitionResult,
)
from .broadcast_system import (
    BroadcastSystem,
    BroadcastParams,
    BroadcastEvent,
)
from .ignition import (
    IgnitionDetector,
    IgnitionParams,
    IgnitionEvent,
)

__all__ = [
    # Module Interface
    'CognitiveModule',
    'ModuleProposal',
    'ModuleState',
    'ModuleParams',
    # Global Workspace
    'GlobalWorkspace',
    'WorkspaceParams',
    'WorkspaceState',
    # Competition
    'AttentionCompetition',
    'CompetitionParams',
    'CompetitionResult',
    # Broadcast
    'BroadcastSystem',
    'BroadcastParams',
    'BroadcastEvent',
    # Ignition
    'IgnitionDetector',
    'IgnitionParams',
    'IgnitionEvent',
]
