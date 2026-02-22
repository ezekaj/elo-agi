"""
Cognitive Agent: Individual agent in a multi-agent system.

Each agent can:
- Generate proposals for the collective
- Receive and integrate broadcasts from others
- Maintain beliefs about self and others (theory of mind)
- Communicate directly with other agents

Based on:
- SwarmSys agent roles (arXiv:2510.10047)
- Theory of Mind from Module 15
- CognitiveModule interface from Module 00
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
from queue import Queue
import numpy as np
import time
import uuid


class AgentRole(Enum):
    """Roles an agent can take in the swarm."""

    EXPLORER = "explorer"  # Discover new information
    WORKER = "worker"  # Execute tasks
    VALIDATOR = "validator"  # Verify results
    GENERALIST = "generalist"  # Flexible role


class ContentType(Enum):
    """Types of content agents can propose."""

    OBSERVATION = "observation"
    HYPOTHESIS = "hypothesis"
    SOLUTION = "solution"
    QUERY = "query"
    VALIDATION = "validation"


@dataclass
class BeliefState:
    """Represents an agent's beliefs."""

    beliefs: Dict[str, Any] = field(default_factory=dict)
    confidence: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def update(self, key: str, value: Any, conf: float = 0.5) -> None:
        """Update a belief with confidence."""
        self.beliefs[key] = value
        self.confidence[key] = conf
        self.timestamp = time.time()

    def get(self, key: str, default: Any = None) -> Tuple[Any, float]:
        """Get belief and its confidence."""
        return self.beliefs.get(key, default), self.confidence.get(key, 0.0)

    def merge(self, other: "BeliefState", weight: float = 0.5) -> None:
        """Merge another belief state into this one."""
        for key, value in other.beliefs.items():
            if key in self.beliefs:
                # Weighted average for confidence
                old_conf = self.confidence.get(key, 0.5)
                new_conf = other.confidence.get(key, 0.5)
                self.confidence[key] = (1 - weight) * old_conf + weight * new_conf
            else:
                self.beliefs[key] = value
                self.confidence[key] = other.confidence.get(key, 0.5) * weight


@dataclass
class Message:
    """Message passed between agents."""

    sender_id: str
    recipient_id: str
    content: Any
    content_type: ContentType
    timestamp: float = field(default_factory=time.time)
    priority: float = 0.5


@dataclass
class AgentParams:
    """Parameters for a cognitive agent."""

    agent_id: str = field(default_factory=lambda: f"agent_{uuid.uuid4().hex[:8]}")
    role: AgentRole = AgentRole.GENERALIST
    specialization: np.ndarray = field(default_factory=lambda: np.random.randn(32))
    communication_range: float = 1.0  # 0-1, how far broadcasts reach
    learning_rate: float = 0.1
    curiosity: float = 0.5  # Drive to explore
    confidence_threshold: float = 0.3  # Min confidence to broadcast


@dataclass
class ModuleProposal:
    """Proposal from an agent to the collective."""

    source_agent: str
    content: np.ndarray
    content_type: ContentType
    activation: float
    confidence: float
    relevance: float
    timestamp: float = field(default_factory=time.time)

    @property
    def priority(self) -> float:
        """Compute proposal priority."""
        return self.activation * self.confidence * self.relevance


class CognitiveAgent:
    """
    Individual cognitive agent that can coordinate with others.

    Each agent maintains:
    - Own belief state (what it knows)
    - Models of other agents (theory of mind)
    - Message queues for communication

    The agent participates in collective intelligence by:
    - Proposing content to the shared workspace
    - Receiving broadcasts from winning proposals
    - Learning from collective experience
    """

    def __init__(self, params: Optional[AgentParams] = None):
        self.params = params or AgentParams()

        # Core state
        self.belief_state = BeliefState()
        self.other_models: Dict[str, BeliefState] = {}

        # Communication
        self.messages_in: Queue = Queue()
        self.messages_out: Queue = Queue()

        # Internal processing state
        self._activation = 0.5
        self._last_broadcast: Optional[ModuleProposal] = None
        self._proposal_history: List[ModuleProposal] = []

        # Statistics
        self._n_proposals = 0
        self._n_accepted = 0
        self._total_contribution = 0.0

    @property
    def agent_id(self) -> str:
        return self.params.agent_id

    @property
    def role(self) -> AgentRole:
        return self.params.role

    def propose(self, input_state: np.ndarray) -> List[ModuleProposal]:
        """
        Generate proposals for the global workspace.

        Proposals are based on:
        - Current role (explorer, worker, validator)
        - Specialization (what the agent is good at)
        - Current activation level
        """
        proposals = []

        # Compute relevance based on specialization match
        if len(input_state) > 0:
            spec = self.params.specialization
            if len(spec) != len(input_state):
                # Resize specialization to match input
                spec = np.resize(spec, len(input_state))
            relevance = float(
                np.tanh(
                    np.dot(spec, input_state)
                    / (np.linalg.norm(spec) * np.linalg.norm(input_state) + 1e-8)
                )
            )
            relevance = (relevance + 1) / 2  # Normalize to 0-1
        else:
            relevance = 0.5

        # Generate content based on role
        content_type, content = self._generate_content(input_state)

        # Only propose if confidence is above threshold
        confidence = self._compute_confidence(content)
        if confidence >= self.params.confidence_threshold:
            proposal = ModuleProposal(
                source_agent=self.agent_id,
                content=content,
                content_type=content_type,
                activation=self._activation,
                confidence=confidence,
                relevance=relevance,
            )
            proposals.append(proposal)
            self._n_proposals += 1
            self._proposal_history.append(proposal)

        return proposals

    def _generate_content(self, input_state: np.ndarray) -> Tuple[ContentType, np.ndarray]:
        """Generate content based on role."""
        if self.role == AgentRole.EXPLORER:
            # Explorers generate hypotheses - add noise to explore
            content = input_state + self.params.curiosity * np.random.randn(len(input_state))
            return ContentType.HYPOTHESIS, content

        elif self.role == AgentRole.WORKER:
            # Workers generate solutions - transform input
            content = np.tanh(input_state * self.params.specialization[: len(input_state)])
            return ContentType.SOLUTION, content

        elif self.role == AgentRole.VALIDATOR:
            # Validators generate validations - assess input
            content = np.array([np.mean(input_state), np.std(input_state), np.max(input_state)])
            return ContentType.VALIDATION, content

        else:  # GENERALIST
            # Generalists observe - pass through with small transform
            content = input_state * 0.9 + 0.1 * np.random.randn(len(input_state))
            return ContentType.OBSERVATION, content

    def _compute_confidence(self, content: np.ndarray) -> float:
        """Compute confidence in generated content."""
        # Base confidence from activation
        base = self._activation

        # Modifier based on content stability
        if len(content) > 0:
            stability = 1.0 / (1.0 + np.std(content))
        else:
            stability = 0.5

        # Modifier based on experience
        experience = min(1.0, self._n_proposals / 100)

        return float(np.clip(base * stability * (0.5 + 0.5 * experience), 0, 1))

    def receive_broadcast(self, proposal: ModuleProposal) -> None:
        """
        Process a broadcast from the global workspace.

        Updates beliefs based on winning proposals from other agents.
        """
        self._last_broadcast = proposal

        # Don't learn from own proposals
        if proposal.source_agent == self.agent_id:
            return

        # Update belief state based on broadcast
        key = f"broadcast_{proposal.content_type.value}"
        self.belief_state.update(key, proposal.content, proposal.confidence)

        # Update model of sender
        if proposal.source_agent not in self.other_models:
            self.other_models[proposal.source_agent] = BeliefState()

        self.other_models[proposal.source_agent].update(
            proposal.content_type.value, proposal.content, proposal.confidence
        )

        # Update activation based on broadcast relevance
        self._activation = (
            1 - self.params.learning_rate
        ) * self._activation + self.params.learning_rate * proposal.relevance

    def send_message(
        self, recipient: str, content: Any, content_type: ContentType = ContentType.OBSERVATION
    ) -> None:
        """Send a direct message to another agent."""
        msg = Message(
            sender_id=self.agent_id,
            recipient_id=recipient,
            content=content,
            content_type=content_type,
        )
        self.messages_out.put(msg)

    def receive_message(self, message: Message) -> None:
        """Receive and process a direct message."""
        self.messages_in.put(message)

        # Update model of sender
        if message.sender_id not in self.other_models:
            self.other_models[message.sender_id] = BeliefState()

        self.other_models[message.sender_id].update(
            f"message_{message.content_type.value}", message.content, message.priority
        )

    def process_messages(self) -> int:
        """Process all pending messages."""
        count = 0
        while not self.messages_in.empty():
            msg = self.messages_in.get()
            self._process_single_message(msg)
            count += 1
        return count

    def _process_single_message(self, message: Message) -> None:
        """Process a single message."""
        # Integrate message content into beliefs
        if isinstance(message.content, np.ndarray):
            key = f"received_{message.content_type.value}"
            self.belief_state.update(key, message.content, message.priority)

    def update_other_model(self, agent_id: str, observation: Any) -> None:
        """Update theory of mind for another agent."""
        if agent_id not in self.other_models:
            self.other_models[agent_id] = BeliefState()

        if isinstance(observation, dict):
            for key, value in observation.items():
                self.other_models[agent_id].update(key, value)
        else:
            self.other_models[agent_id].update("observation", observation)

    def process(self, dt: float = 0.1) -> None:
        """Internal processing step."""
        # Decay activation over time
        self._activation *= 1 - 0.01 * dt

        # Process any pending messages
        self.process_messages()

        # Curiosity-driven activation boost
        if self.role == AgentRole.EXPLORER:
            self._activation += self.params.curiosity * 0.01 * dt

    def evaluate(self, proposal: Any) -> float:
        """Evaluate a proposal (for voting)."""
        if isinstance(proposal, ModuleProposal):
            # Score based on relevance to specialization
            if isinstance(proposal.content, np.ndarray) and len(proposal.content) > 0:
                spec = np.resize(self.params.specialization, len(proposal.content))
                score = float(
                    np.dot(spec, proposal.content)
                    / (np.linalg.norm(spec) * np.linalg.norm(proposal.content) + 1e-8)
                )
                return (score + 1) / 2
        return 0.5

    def expertise_on(self, topic: str) -> float:
        """Return expertise level on a topic."""
        # Simple hash-based expertise
        topic_hash = hash(topic) % 1000 / 1000
        spec_mean = np.mean(np.abs(self.params.specialization))
        return float(np.clip(spec_mean * topic_hash, 0, 1))

    def record_acceptance(self, improvement: float = 0.0) -> None:
        """Record that a proposal was accepted."""
        self._n_accepted += 1
        self._total_contribution += improvement
        # Boost activation on acceptance
        self._activation = min(1.0, self._activation + 0.1)

    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics."""
        acceptance_rate = self._n_accepted / max(1, self._n_proposals)
        return {
            "agent_id": self.agent_id,
            "role": self.role.value,
            "activation": self._activation,
            "n_proposals": self._n_proposals,
            "n_accepted": self._n_accepted,
            "acceptance_rate": acceptance_rate,
            "total_contribution": self._total_contribution,
            "n_beliefs": len(self.belief_state.beliefs),
            "n_other_models": len(self.other_models),
        }

    def reset(self) -> None:
        """Reset agent state."""
        self.belief_state = BeliefState()
        self.other_models = {}
        self.messages_in = Queue()
        self.messages_out = Queue()
        self._activation = 0.5
        self._last_broadcast = None
        self._proposal_history = []
        self._n_proposals = 0
        self._n_accepted = 0
        self._total_contribution = 0.0
