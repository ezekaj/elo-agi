"""
Broadcast System: Distributes winning content to all cognitive modules.

When a proposal wins the competition and triggers ignition, its content
is broadcast to all registered modules. This enables global information
sharing, which is the key mechanism of consciousness in Global Workspace Theory.

Based on:
- Baars' "theater" metaphor (spotlight broadcasts to audience)
- Dehaene's "ignition and broadcast" mechanism
- arXiv:2012.10390 - bidirectional broadcast for System-2 reasoning
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
import numpy as np
import time

try:
    from .module_interface import ModuleProposal, ModuleType, ContentType
except ImportError:
    from module_interface import ModuleProposal, ModuleType, ContentType


class BroadcastPriority(Enum):
    """Priority levels for broadcasts."""
    CRITICAL = 4    # Emergency, immediate attention required
    HIGH = 3        # Important, should be processed soon
    NORMAL = 2      # Standard broadcast
    LOW = 1         # Background information
    MINIMAL = 0     # Can be ignored if busy


@dataclass
class BroadcastParams:
    """Parameters for the broadcast system."""
    broadcast_strength: float = 1.0
    max_history_size: int = 100
    enable_filtering: bool = True
    min_broadcast_interval: float = 0.01  # seconds
    amplification_factor: float = 1.2  # Boost for broadcast content


@dataclass
class BroadcastEvent:
    """Record of a broadcast event."""
    proposal: ModuleProposal
    timestamp: float
    priority: BroadcastPriority
    recipients: List[ModuleType]
    success_rate: float  # Fraction of modules that received successfully
    metadata: Dict[str, Any] = field(default_factory=dict)


class BroadcastSystem:
    """
    System for broadcasting winning content to all modules.

    The broadcast system implements the "global availability" aspect of
    consciousness in Global Workspace Theory. When content wins the
    competition and triggers ignition, it is broadcast to all modules,
    making it globally available for processing.

    Features:
    1. **Amplification**: Broadcast content is amplified for salience
    2. **Filtering**: Modules can filter broadcasts they don't need
    3. **History**: Maintains history of recent broadcasts
    4. **Priority**: Different priority levels for different content
    5. **Callbacks**: Supports custom broadcast handlers
    """

    def __init__(self, params: Optional[BroadcastParams] = None):
        self.params = params or BroadcastParams()

        # Broadcast history
        self._history: List[BroadcastEvent] = []

        # Registered broadcast handlers (callbacks)
        self._handlers: List[Callable[[BroadcastEvent], None]] = []

        # Filtering rules (module_type -> list of content types to ignore)
        self._filters: Dict[ModuleType, List[ContentType]] = {}

        # Statistics
        self._total_broadcasts = 0
        self._successful_deliveries = 0
        self._last_broadcast_time = 0.0

    def broadcast(
        self,
        proposal: ModuleProposal,
        recipients: List[ModuleType],
        priority: BroadcastPriority = BroadcastPriority.NORMAL,
    ) -> BroadcastEvent:
        """
        Broadcast a proposal to specified recipients.

        Args:
            proposal: The winning proposal to broadcast
            recipients: List of module types to receive broadcast
            priority: Priority level of the broadcast

        Returns:
            BroadcastEvent record
        """
        current_time = time.time()

        # Rate limiting
        if current_time - self._last_broadcast_time < self.params.min_broadcast_interval:
            # Skip this broadcast
            return BroadcastEvent(
                proposal=proposal,
                timestamp=current_time,
                priority=priority,
                recipients=[],
                success_rate=0.0,
                metadata={'skipped': True, 'reason': 'rate_limited'},
            )

        # Amplify broadcast content
        amplified_proposal = self._amplify(proposal)

        # Filter recipients based on content type
        filtered_recipients = self._filter_recipients(recipients, proposal.content_type)

        # Record statistics
        self._total_broadcasts += 1
        self._last_broadcast_time = current_time

        # Create event record
        event = BroadcastEvent(
            proposal=amplified_proposal,
            timestamp=current_time,
            priority=priority,
            recipients=filtered_recipients,
            success_rate=len(filtered_recipients) / max(1, len(recipients)),
        )

        # Add to history
        self._history.append(event)
        if len(self._history) > self.params.max_history_size:
            self._history.pop(0)

        # Notify handlers
        for handler in self._handlers:
            try:
                handler(event)
            except Exception:
                pass

        return event

    def _amplify(self, proposal: ModuleProposal) -> ModuleProposal:
        """Amplify proposal content for broadcast."""
        # Boost activation
        amplified_activation = min(1.0, proposal.activation * self.params.amplification_factor)

        # Create amplified copy
        return ModuleProposal(
            source_module=proposal.source_module,
            content_type=proposal.content_type,
            content=proposal.content * self.params.broadcast_strength,
            activation=amplified_activation,
            confidence=proposal.confidence,
            relevance=proposal.relevance,
            timestamp=proposal.timestamp,
            metadata={**proposal.metadata, 'amplified': True},
        )

    def _filter_recipients(
        self,
        recipients: List[ModuleType],
        content_type: ContentType,
    ) -> List[ModuleType]:
        """Filter recipients based on their filter rules."""
        if not self.params.enable_filtering:
            return recipients

        filtered = []
        for recipient in recipients:
            if recipient in self._filters:
                if content_type in self._filters[recipient]:
                    continue  # This recipient filters this content type
            filtered.append(recipient)

        return filtered

    def record_broadcast(self, proposal: ModuleProposal) -> None:
        """Record a broadcast event (for external broadcasts)."""
        event = BroadcastEvent(
            proposal=proposal,
            timestamp=time.time(),
            priority=BroadcastPriority.NORMAL,
            recipients=[],
            success_rate=1.0,
        )
        self._history.append(event)
        if len(self._history) > self.params.max_history_size:
            self._history.pop(0)

    def add_handler(self, handler: Callable[[BroadcastEvent], None]) -> None:
        """Add a broadcast event handler."""
        self._handlers.append(handler)

    def remove_handler(self, handler: Callable[[BroadcastEvent], None]) -> None:
        """Remove a broadcast event handler."""
        if handler in self._handlers:
            self._handlers.remove(handler)

    def set_filter(
        self,
        module_type: ModuleType,
        content_types: List[ContentType],
    ) -> None:
        """Set content type filter for a module."""
        self._filters[module_type] = content_types

    def clear_filter(self, module_type: ModuleType) -> None:
        """Clear content type filter for a module."""
        if module_type in self._filters:
            del self._filters[module_type]

    def get_recent_broadcasts(self, n: int = 10) -> List[BroadcastEvent]:
        """Get the n most recent broadcasts."""
        return self._history[-n:]

    def get_broadcasts_by_type(self, content_type: ContentType) -> List[BroadcastEvent]:
        """Get broadcasts of a specific content type."""
        return [e for e in self._history if e.proposal.content_type == content_type]

    def get_broadcasts_by_source(self, source: ModuleType) -> List[BroadcastEvent]:
        """Get broadcasts from a specific source module."""
        return [e for e in self._history if e.proposal.source_module == source]

    def get_statistics(self) -> Dict[str, Any]:
        """Get broadcast statistics."""
        if not self._history:
            return {
                'total_broadcasts': 0,
                'avg_success_rate': 0.0,
                'broadcasts_by_type': {},
                'broadcasts_by_source': {},
            }

        # Count by type
        by_type: Dict[str, int] = {}
        for event in self._history:
            type_name = event.proposal.content_type.value
            by_type[type_name] = by_type.get(type_name, 0) + 1

        # Count by source
        by_source: Dict[str, int] = {}
        for event in self._history:
            source_name = event.proposal.source_module.name
            by_source[source_name] = by_source.get(source_name, 0) + 1

        return {
            'total_broadcasts': len(self._history),
            'avg_success_rate': np.mean([e.success_rate for e in self._history]),
            'broadcasts_by_type': by_type,
            'broadcasts_by_source': by_source,
            'last_broadcast_time': self._last_broadcast_time,
        }

    def reset(self) -> None:
        """Reset broadcast system state."""
        self._history = []
        self._total_broadcasts = 0
        self._successful_deliveries = 0
        self._last_broadcast_time = 0.0
