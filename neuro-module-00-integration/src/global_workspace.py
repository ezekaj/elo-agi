"""
Global Workspace: Central integration mechanism for cognitive modules.

Implements Global Workspace Theory (Baars, 1988) as the integration layer
that connects all cognitive modules. The workspace provides:

1. A shared communication channel for all modules
2. Competition mechanism for workspace access
3. Broadcast of winning content to all modules
4. Ignition detection for conscious-like processing

Based on:
- Baars (1988) - A Cognitive Theory of Consciousness
- Dehaene et al. (2001) - Towards a cognitive neuroscience of consciousness
- arXiv:2103.01197 - Coordination Among Neural Modules
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np
import time

from .module_interface import (
    CognitiveModule,
    ModuleProposal,
    ModuleState,
    ModuleType,
    ContentType,
)
from .attention_competition import AttentionCompetition, CompetitionParams
from .broadcast_system import BroadcastSystem, BroadcastParams
from .ignition import IgnitionDetector, IgnitionParams


class WorkspaceMode(Enum):
    """Operating modes for the global workspace."""
    NORMAL = "normal"           # Standard competition and broadcast
    FOCUSED = "focused"         # Enhanced competition, fewer broadcasts
    DIFFUSE = "diffuse"         # Relaxed competition, more broadcasts
    MAINTENANCE = "maintenance"  # Holding current content, minimal updates


@dataclass
class WorkspaceParams:
    """Parameters for the global workspace."""
    n_features: int = 64
    buffer_capacity: int = 7  # Miller's magical number (7 ± 2)
    ignition_threshold: float = 0.7
    decay_rate: float = 0.95
    broadcast_strength: float = 1.0
    min_competition_interval: float = 0.05  # seconds
    mode: WorkspaceMode = WorkspaceMode.NORMAL


@dataclass
class WorkspaceState:
    """Current state of the global workspace."""
    buffer: List[ModuleProposal]
    current_broadcast: Optional[ModuleProposal]
    ignition_active: bool
    activation_level: float
    mode: WorkspaceMode
    registered_modules: List[ModuleType]
    step_count: int
    last_broadcast_time: float

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the workspace state."""
        return {
            'buffer_size': len(self.buffer),
            'buffer_capacity': 7,
            'ignition_active': self.ignition_active,
            'activation': self.activation_level,
            'mode': self.mode.value,
            'module_count': len(self.registered_modules),
            'step_count': self.step_count,
        }


class GlobalWorkspace:
    """
    Central integration mechanism implementing Global Workspace Theory.

    The global workspace serves as a shared "blackboard" where cognitive
    modules can post information and receive broadcasts. Key features:

    1. **Limited Capacity**: Only ~7 items can be held in the workspace
       at once (Miller's Law), forcing competition.

    2. **Competition**: Modules compete for workspace access based on
       activation strength, confidence, and relevance.

    3. **Broadcast**: The winning proposal is broadcast to all modules,
       enabling global information sharing.

    4. **Ignition**: When activation exceeds threshold, content becomes
       globally available (conscious-like processing).

    Usage:
        workspace = GlobalWorkspace(params)
        workspace.register_module(module1)
        workspace.register_module(module2)

        # Main loop
        while running:
            proposals = workspace.collect_proposals(input_state)
            winners = workspace.compete(proposals)
            workspace.broadcast(winners)
            workspace.step(dt)
    """

    def __init__(self, params: Optional[WorkspaceParams] = None):
        self.params = params or WorkspaceParams()

        # Core components
        self.competition = AttentionCompetition(CompetitionParams(
            n_features=self.params.n_features,
            capacity=self.params.buffer_capacity,
        ))
        self.broadcast_system = BroadcastSystem(BroadcastParams(
            broadcast_strength=self.params.broadcast_strength,
        ))
        self.ignition_detector = IgnitionDetector(IgnitionParams(
            threshold=self.params.ignition_threshold,
        ))

        # Registered modules
        self._modules: Dict[ModuleType, CognitiveModule] = {}

        # Workspace buffer (limited capacity)
        self._buffer: List[ModuleProposal] = []

        # Current broadcast content
        self._current_broadcast: Optional[ModuleProposal] = None

        # State tracking
        self._activation_level = 0.0
        self._ignition_active = False
        self._step_count = 0
        self._last_broadcast_time = 0.0
        self._mode = self.params.mode

        # History for analysis
        self._broadcast_history: List[ModuleProposal] = []
        self._ignition_history: List[Tuple[float, bool]] = []

    def register_module(self, module: CognitiveModule) -> None:
        """Register a cognitive module with the workspace."""
        self._modules[module.module_type] = module

    def unregister_module(self, module_type: ModuleType) -> None:
        """Remove a module from the workspace."""
        if module_type in self._modules:
            del self._modules[module_type]

    def get_module(self, module_type: ModuleType) -> Optional[CognitiveModule]:
        """Get a registered module by type."""
        return self._modules.get(module_type)

    def collect_proposals(self, input_state: np.ndarray) -> List[ModuleProposal]:
        """
        Collect proposals from all registered modules.

        Each active module generates proposals based on the current input
        and its internal state.

        Args:
            input_state: Current input to the system

        Returns:
            All proposals from all modules
        """
        all_proposals = []

        for module in self._modules.values():
            if module._is_active:
                try:
                    proposals = module.propose(input_state)
                    all_proposals.extend(proposals)
                except Exception as e:
                    # Module failed to generate proposals
                    pass

        return all_proposals

    def compete(self, proposals: List[ModuleProposal]) -> List[ModuleProposal]:
        """
        Run competition among proposals for workspace access.

        Uses attention-based competition to select winning proposals.
        The number of winners is limited by buffer capacity.

        Args:
            proposals: All proposals from modules

        Returns:
            Winning proposals that enter the workspace
        """
        if not proposals:
            return []

        # Adjust competition based on mode
        if self._mode == WorkspaceMode.FOCUSED:
            # Stricter competition - fewer winners
            self.competition.params.capacity = max(3, self.params.buffer_capacity - 2)
        elif self._mode == WorkspaceMode.DIFFUSE:
            # Relaxed competition - more winners
            self.competition.params.capacity = self.params.buffer_capacity + 2
        else:
            self.competition.params.capacity = self.params.buffer_capacity

        # Run competition
        result = self.competition.compete(proposals)

        # Update buffer with winners
        self._update_buffer(result.winners)

        # Update activation level
        if result.winners:
            self._activation_level = max(p.activation for p in result.winners)

        return result.winners

    def _update_buffer(self, winners: List[ModuleProposal]) -> None:
        """Update the workspace buffer with winning proposals."""
        # Add new winners
        for winner in winners:
            # Check if similar content already in buffer
            is_duplicate = False
            for existing in self._buffer:
                if self._is_similar(winner.content, existing.content):
                    # Update existing entry if new one is stronger
                    if winner.activation > existing.activation:
                        self._buffer.remove(existing)
                        self._buffer.append(winner)
                    is_duplicate = True
                    break

            if not is_duplicate:
                self._buffer.append(winner)

        # Enforce capacity limit (remove oldest/weakest)
        while len(self._buffer) > self.params.buffer_capacity:
            # Find weakest proposal
            weakest = min(self._buffer, key=lambda p: p.priority)
            self._buffer.remove(weakest)

        # Decay existing buffer items
        for proposal in self._buffer:
            proposal.decay(self.params.decay_rate)

        # Remove items below activation threshold
        self._buffer = [p for p in self._buffer if p.activation > 0.1]

    def _is_similar(self, content1: np.ndarray, content2: np.ndarray) -> bool:
        """Check if two content vectors are similar."""
        if len(content1) != len(content2):
            return False
        norm1 = np.linalg.norm(content1)
        norm2 = np.linalg.norm(content2)
        if norm1 < 1e-8 or norm2 < 1e-8:
            return False
        similarity = np.dot(content1, content2) / (norm1 * norm2)
        return similarity > 0.9

    def check_ignition(self) -> bool:
        """
        Check if ignition threshold is reached.

        Ignition occurs when workspace activation exceeds threshold,
        triggering global broadcast of content.

        Returns:
            True if ignition is active
        """
        result = self.ignition_detector.detect(
            activation=self._activation_level,
            buffer=self._buffer,
        )

        self._ignition_active = result.ignited
        self._ignition_history.append((time.time(), result.ignited))

        return result.ignited

    def broadcast(self, winners: Optional[List[ModuleProposal]] = None) -> Optional[ModuleProposal]:
        """
        Broadcast winning content to all modules.

        The strongest winning proposal is broadcast to all registered
        modules, allowing global information sharing.

        Args:
            winners: Proposals to broadcast (uses buffer if None)

        Returns:
            The broadcast proposal, or None if no broadcast
        """
        current_time = time.time()

        # Rate limiting
        if current_time - self._last_broadcast_time < self.params.min_competition_interval:
            return None

        # Determine what to broadcast
        if winners is None:
            winners = self._buffer

        if not winners:
            return None

        # Check ignition
        if not self.check_ignition():
            return None

        # Select strongest proposal
        broadcast_proposal = max(winners, key=lambda p: p.priority)

        # Broadcast to all modules
        for module in self._modules.values():
            if module._is_active:
                try:
                    module.receive_broadcast(broadcast_proposal)
                except Exception as e:
                    # Module failed to receive broadcast
                    pass

        # Update state
        self._current_broadcast = broadcast_proposal
        self._last_broadcast_time = current_time
        self._broadcast_history.append(broadcast_proposal)

        # Record broadcast event
        self.broadcast_system.record_broadcast(broadcast_proposal)

        return broadcast_proposal

    def step(self, dt: float = 0.1) -> WorkspaceState:
        """
        Perform one integration step.

        This should be called each simulation timestep. It:
        1. Decays buffer activations
        2. Updates all registered modules
        3. Checks ignition state
        4. Returns current workspace state

        Args:
            dt: Time step size in seconds

        Returns:
            Current workspace state
        """
        self._step_count += 1

        # Decay activation
        self._activation_level *= self.params.decay_rate

        # Update ignition state
        self.check_ignition()

        # Process all modules
        for module in self._modules.values():
            if module._is_active:
                try:
                    module.process(dt)
                except Exception as e:
                    pass

        return self.get_state()

    def run_cycle(self, input_state: np.ndarray, dt: float = 0.1) -> Tuple[List[ModuleProposal], Optional[ModuleProposal]]:
        """
        Run a complete workspace cycle.

        This is a convenience method that runs:
        1. Collect proposals from all modules
        2. Run competition
        3. Broadcast winners
        4. Step all modules

        Args:
            input_state: Current input to the system
            dt: Time step size

        Returns:
            Tuple of (winners, broadcast proposal)
        """
        # Collect proposals
        proposals = self.collect_proposals(input_state)

        # Competition
        winners = self.compete(proposals)

        # Broadcast
        broadcast = self.broadcast(winners)

        # Step
        self.step(dt)

        return winners, broadcast

    def get_state(self) -> WorkspaceState:
        """Get current workspace state."""
        return WorkspaceState(
            buffer=self._buffer.copy(),
            current_broadcast=self._current_broadcast,
            ignition_active=self._ignition_active,
            activation_level=self._activation_level,
            mode=self._mode,
            registered_modules=list(self._modules.keys()),
            step_count=self._step_count,
            last_broadcast_time=self._last_broadcast_time,
        )

    def set_mode(self, mode: WorkspaceMode) -> None:
        """Set the workspace operating mode."""
        self._mode = mode

    def set_attention_bias(self, bias: np.ndarray) -> None:
        """Set top-down attention bias for competition."""
        self.competition.set_attention_bias(bias)

    def reset(self) -> None:
        """Reset the workspace to initial state."""
        self._buffer = []
        self._current_broadcast = None
        self._activation_level = 0.0
        self._ignition_active = False
        self._step_count = 0
        self._last_broadcast_time = 0.0
        self._broadcast_history = []
        self._ignition_history = []

        # Reset all modules
        for module in self._modules.values():
            module.reset()

    def get_buffer_contents(self) -> List[Dict[str, Any]]:
        """Get human-readable buffer contents."""
        return [
            {
                'source': p.source_module.name,
                'type': p.content_type.value,
                'activation': p.activation,
                'confidence': p.confidence,
                'priority': p.priority,
            }
            for p in self._buffer
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """Get workspace statistics."""
        return {
            'step_count': self._step_count,
            'total_broadcasts': len(self._broadcast_history),
            'ignition_rate': sum(1 for _, i in self._ignition_history if i) / max(1, len(self._ignition_history)),
            'module_count': len(self._modules),
            'buffer_utilization': len(self._buffer) / self.params.buffer_capacity,
            'current_activation': self._activation_level,
            'mode': self._mode.value,
        }
