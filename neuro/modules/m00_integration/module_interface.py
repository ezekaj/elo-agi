"""
Module Interface: Standard API for all cognitive modules.

Every cognitive module (01-16) must implement the CognitiveModule interface
to participate in the Global Workspace. This ensures consistent communication
and enables dynamic module discovery and coordination.

Based on:
- Global Workspace Theory requirements for specialist modules
- arXiv:2103.01197 - shared workspace communication
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import numpy as np
import time


class ModuleType(Enum):
    """Types of cognitive modules in the system."""

    PREDICTIVE_CODING = 1  # Module 01
    DUAL_PROCESS = 2  # Module 02
    REASONING = 3  # Module 03
    MEMORY = 4  # Module 04
    SLEEP_CONSOLIDATION = 5  # Module 05
    MOTIVATION = 6  # Module 06
    EMOTION = 7  # Module 07
    LANGUAGE = 8  # Module 08
    CREATIVITY = 9  # Module 09
    SPATIAL = 10  # Module 10
    TEMPORAL = 11  # Module 11
    LEARNING = 12  # Module 12
    EXECUTIVE = 13  # Module 13
    EMBODIED = 14  # Module 14
    SOCIAL = 15  # Module 15
    CONSCIOUSNESS = 16  # Module 16
    INTEGRATION = 0  # Module 00 (this module)
    WORLD_MODEL = 17  # Module 17
    SELF_IMPROVEMENT = 18  # Module 18


class ContentType(Enum):
    """Types of content that can be broadcast in the workspace."""

    PERCEPT = "percept"  # Sensory input
    BELIEF = "belief"  # Belief state
    INTENTION = "intention"  # Goal/intention
    MEMORY = "memory"  # Retrieved memory
    PREDICTION = "prediction"  # Predicted state
    ERROR = "error"  # Prediction error
    EMOTION = "emotion"  # Emotional state
    ACTION = "action"  # Motor command
    QUERY = "query"  # Information request
    RESPONSE = "response"  # Query response
    METACOGNITIVE = "metacognitive"  # Self-reflective content


@dataclass
class ModuleParams:
    """Parameters for a cognitive module."""

    module_type: ModuleType
    name: str
    n_features: int = 64
    activation_decay: float = 0.95
    learning_rate: float = 0.01
    priority_weight: float = 1.0
    max_proposals_per_step: int = 3


@dataclass
class ModuleProposal:
    """
    A proposal from a module to enter the global workspace.

    Proposals compete for access to the workspace. The winning proposal
    is broadcast to all other modules, enabling global information sharing.
    """

    source_module: ModuleType
    content_type: ContentType
    content: np.ndarray
    activation: float  # Strength of proposal (0-1)
    confidence: float  # Module's confidence (0-1)
    relevance: float  # Relevance to current context (0-1)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __eq__(self, other: object) -> bool:
        """Compare proposals by identity (timestamp + source)."""
        if not isinstance(other, ModuleProposal):
            return False
        return (
            self.timestamp == other.timestamp
            and self.source_module == other.source_module
            and self.content_type == other.content_type
        )

    def __hash__(self) -> int:
        """Hash based on timestamp and source."""
        return hash((self.timestamp, self.source_module, self.content_type))

    @property
    def priority(self) -> float:
        """Compute priority score for competition."""
        return self.activation * self.confidence * self.relevance

    def decay(self, rate: float = 0.95) -> None:
        """Apply activation decay."""
        self.activation *= rate

    def boost(self, amount: float) -> None:
        """Boost activation (e.g., from attention)."""
        self.activation = min(1.0, self.activation + amount)


@dataclass
class ModuleState:
    """Current state of a cognitive module."""

    module_type: ModuleType
    internal_state: np.ndarray
    activation_level: float
    is_active: bool
    pending_proposals: List[ModuleProposal] = field(default_factory=list)
    last_broadcast_received: Optional[ModuleProposal] = None
    processing_load: float = 0.0

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the module state."""
        return {
            "type": self.module_type.name,
            "activation": self.activation_level,
            "active": self.is_active,
            "pending_count": len(self.pending_proposals),
            "load": self.processing_load,
        }


class CognitiveModule(ABC):
    """
    Abstract base class for all cognitive modules.

    Every module in the neuro-system must implement this interface to
    participate in the Global Workspace. The interface defines:

    1. propose() - Generate proposals for the workspace
    2. receive_broadcast() - Process broadcast from winning proposal
    3. process() - Internal processing step
    4. get_state() - Return current module state
    5. reset() - Reset module to initial state

    Based on Global Workspace Theory, modules are specialists that:
    - Operate in parallel on their specialized domain
    - Compete for access to the global workspace
    - Receive broadcasts from winning modules
    - Update their internal state based on broadcasts
    """

    def __init__(self, params: ModuleParams):
        self.params = params
        self.module_type = params.module_type
        self.name = params.name
        self.n_features = params.n_features

        # Internal state
        self._internal_state = np.zeros(params.n_features)
        self._activation_level = 0.0
        self._is_active = True
        self._pending_proposals: List[ModuleProposal] = []
        self._last_broadcast: Optional[ModuleProposal] = None
        self._processing_load = 0.0

        # Callbacks for receiving broadcasts
        self._broadcast_handlers: List[Callable[[ModuleProposal], None]] = []

    @abstractmethod
    def propose(self, input_state: np.ndarray) -> List[ModuleProposal]:
        """
        Generate proposals for the global workspace.

        Each module analyzes its internal state and the input, then
        generates proposals that may compete for global broadcast.

        Args:
            input_state: Current input to the system

        Returns:
            List of proposals (up to max_proposals_per_step)
        """
        pass

    @abstractmethod
    def receive_broadcast(self, proposal: ModuleProposal) -> None:
        """
        Process a broadcast from the global workspace.

        When a proposal wins the competition, it is broadcast to all
        modules. Each module updates its internal state based on the
        broadcast content.

        Args:
            proposal: The winning proposal being broadcast
        """
        pass

    @abstractmethod
    def process(self, dt: float = 0.1) -> None:
        """
        Perform internal processing step.

        This is called each simulation step, allowing the module to
        update its internal state, decay activations, consolidate
        information, etc.

        Args:
            dt: Time step size in seconds
        """
        pass

    def get_state(self) -> ModuleState:
        """Return current module state."""
        return ModuleState(
            module_type=self.module_type,
            internal_state=self._internal_state.copy(),
            activation_level=self._activation_level,
            is_active=self._is_active,
            pending_proposals=self._pending_proposals.copy(),
            last_broadcast_received=self._last_broadcast,
            processing_load=self._processing_load,
        )

    def reset(self) -> None:
        """Reset module to initial state."""
        self._internal_state = np.zeros(self.n_features)
        self._activation_level = 0.0
        self._is_active = True
        self._pending_proposals = []
        self._last_broadcast = None
        self._processing_load = 0.0

    def set_active(self, active: bool) -> None:
        """Enable or disable the module."""
        self._is_active = active

    def add_broadcast_handler(self, handler: Callable[[ModuleProposal], None]) -> None:
        """Add a callback to be called when receiving broadcasts."""
        self._broadcast_handlers.append(handler)

    def _notify_broadcast_handlers(self, proposal: ModuleProposal) -> None:
        """Notify all registered broadcast handlers."""
        for handler in self._broadcast_handlers:
            handler(proposal)

    def _create_proposal(
        self,
        content_type: ContentType,
        content: np.ndarray,
        activation: float,
        confidence: float,
        relevance: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ModuleProposal:
        """Helper to create a proposal from this module."""
        return ModuleProposal(
            source_module=self.module_type,
            content_type=content_type,
            content=content,
            activation=activation,
            confidence=confidence,
            relevance=relevance,
            metadata=metadata or {},
        )

    def compute_relevance(
        self,
        content: np.ndarray,
        context: np.ndarray,
    ) -> float:
        """
        Compute relevance of content to current context.

        Uses cosine similarity between content and context.
        """
        content_norm = np.linalg.norm(content)
        context_norm = np.linalg.norm(context)

        if content_norm < 1e-8 or context_norm < 1e-8:
            return 0.0

        similarity = np.dot(content, context) / (content_norm * context_norm)
        return float(np.clip((similarity + 1) / 2, 0, 1))  # Map [-1, 1] to [0, 1]


class DummyModule(CognitiveModule):
    """
    A simple implementation for testing the interface.

    This module generates random proposals and updates its state
    based on broadcasts. Used for testing the global workspace.
    """

    def __init__(
        self,
        module_type: ModuleType = ModuleType.INTEGRATION,
        name: str = "DummyModule",
        n_features: int = 64,
    ):
        params = ModuleParams(
            module_type=module_type,
            name=name,
            n_features=n_features,
        )
        super().__init__(params)

    def propose(self, input_state: np.ndarray) -> List[ModuleProposal]:
        """Generate a random proposal."""
        if not self._is_active:
            return []

        # Generate random content
        content = np.random.randn(self.n_features) * 0.1
        content = (
            content + input_state[: self.n_features]
            if len(input_state) >= self.n_features
            else content
        )

        # Random activation based on input strength
        activation = float(np.clip(np.mean(np.abs(input_state)) + np.random.rand() * 0.3, 0, 1))
        confidence = float(np.random.rand() * 0.5 + 0.5)
        relevance = self.compute_relevance(
            content,
            input_state[: self.n_features] if len(input_state) >= self.n_features else input_state,
        )

        proposal = self._create_proposal(
            content_type=ContentType.PERCEPT,
            content=content,
            activation=activation,
            confidence=confidence,
            relevance=relevance,
        )

        self._pending_proposals = [proposal]
        return [proposal]

    def receive_broadcast(self, proposal: ModuleProposal) -> None:
        """Update internal state based on broadcast."""
        self._last_broadcast = proposal

        # Blend broadcast content into internal state
        if len(proposal.content) == len(self._internal_state):
            blend_rate = 0.3 * proposal.activation
            self._internal_state = (
                1 - blend_rate
            ) * self._internal_state + blend_rate * proposal.content

        # Update activation based on relevance to broadcast
        relevance = self.compute_relevance(self._internal_state, proposal.content)
        self._activation_level = 0.7 * self._activation_level + 0.3 * relevance

        # Notify handlers
        self._notify_broadcast_handlers(proposal)

    def process(self, dt: float = 0.1) -> None:
        """Apply decay and update state."""
        # Activation decay
        self._activation_level *= self.params.activation_decay

        # State decay toward zero
        self._internal_state *= 0.99

        # Update processing load
        self._processing_load = len(self._pending_proposals) / self.params.max_proposals_per_step
