"""
Agent Population: Manages a population of cognitive agents.

The population handles:
- Agent lifecycle (spawn, remove)
- Communication topology
- Collective proposal/broadcast cycle
- Consensus and diversity metrics

Based on:
- SwarmSys population dynamics (arXiv:2510.10047)
- Multi-agent coordination patterns
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np
import time

from .agent import (
    CognitiveAgent,
    AgentParams,
    AgentRole,
    ModuleProposal,
    ContentType,
    Message,
    BeliefState,
)


class TopologyType(Enum):
    """Types of communication topology."""

    FULLY_CONNECTED = "fully_connected"  # All can communicate with all
    RING = "ring"  # Each connects to neighbors
    STAR = "star"  # Central hub
    RANDOM = "random"  # Random connections
    SMALL_WORLD = "small_world"  # Clustered with shortcuts


@dataclass
class PopulationParams:
    """Parameters for the agent population."""

    n_agents: int = 10
    topology: TopologyType = TopologyType.FULLY_CONNECTED
    role_distribution: Dict[AgentRole, float] = field(
        default_factory=lambda: {
            AgentRole.EXPLORER: 0.2,
            AgentRole.WORKER: 0.5,
            AgentRole.VALIDATOR: 0.2,
            AgentRole.GENERALIST: 0.1,
        }
    )
    workspace_capacity: int = 7  # Miller's Law
    ignition_threshold: float = 0.7
    broadcast_decay: float = 0.1


@dataclass
class PopulationState:
    """State of the population at a given time."""

    n_agents: int
    n_proposals: int
    n_broadcasts: int
    diversity: float
    consensus: float
    mean_activation: float
    winning_proposal: Optional[ModuleProposal]
    timestamp: float = field(default_factory=time.time)


class AgentPopulation:
    """
    Manages a population of cognitive agents.

    The population coordinates agents through:
    1. Collecting proposals from all agents
    2. Running competition (attention-based)
    3. Broadcasting winners to all agents
    4. Tracking consensus and diversity
    """

    def __init__(self, params: Optional[PopulationParams] = None):
        self.params = params or PopulationParams()

        # Agents
        self.agents: Dict[str, CognitiveAgent] = {}

        # Communication topology (adjacency matrix)
        self._topology: Optional[np.ndarray] = None

        # Workspace buffer
        self._workspace: List[ModuleProposal] = []

        # History
        self._broadcast_history: List[ModuleProposal] = []
        self._state_history: List[PopulationState] = []

        # Statistics
        self._step_count = 0
        self._total_proposals = 0
        self._total_broadcasts = 0

        # Initialize population
        self._initialize_population()

    def _initialize_population(self) -> None:
        """Initialize the population with agents."""
        for i in range(self.params.n_agents):
            role = self._sample_role()
            params = AgentParams(
                agent_id=f"agent_{i}",
                role=role,
                specialization=np.random.randn(32),
            )
            self.spawn_agent(params)

        # Build topology
        self._build_topology()

    def _sample_role(self) -> AgentRole:
        """Sample a role based on distribution."""
        roles = list(self.params.role_distribution.keys())
        probs = list(self.params.role_distribution.values())
        probs = np.array(probs) / sum(probs)  # Normalize
        return np.random.choice(roles, p=probs)

    def _build_topology(self) -> None:
        """Build communication topology."""
        n = len(self.agents)
        if n == 0:
            self._topology = np.array([])
            return

        if self.params.topology == TopologyType.FULLY_CONNECTED:
            self._topology = np.ones((n, n)) - np.eye(n)

        elif self.params.topology == TopologyType.RING:
            self._topology = np.zeros((n, n))
            for i in range(n):
                self._topology[i, (i + 1) % n] = 1
                self._topology[i, (i - 1) % n] = 1

        elif self.params.topology == TopologyType.STAR:
            self._topology = np.zeros((n, n))
            for i in range(1, n):
                self._topology[0, i] = 1
                self._topology[i, 0] = 1

        elif self.params.topology == TopologyType.RANDOM:
            self._topology = (np.random.rand(n, n) > 0.5).astype(float)
            self._topology = np.triu(self._topology, 1) + np.triu(self._topology, 1).T

        elif self.params.topology == TopologyType.SMALL_WORLD:
            # Ring + random shortcuts
            self._topology = np.zeros((n, n))
            for i in range(n):
                self._topology[i, (i + 1) % n] = 1
                self._topology[i, (i - 1) % n] = 1
            # Add shortcuts
            n_shortcuts = n // 4
            for _ in range(n_shortcuts):
                i, j = np.random.randint(0, n, 2)
                if i != j:
                    self._topology[i, j] = 1
                    self._topology[j, i] = 1

    def spawn_agent(self, params: AgentParams) -> CognitiveAgent:
        """Create and register a new agent."""
        agent = CognitiveAgent(params)
        self.agents[agent.agent_id] = agent
        return agent

    def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent from the population."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            self._build_topology()  # Rebuild topology
            return True
        return False

    def step(self, input_state: np.ndarray) -> PopulationState:
        """
        Run one step of the population.

        1. Each agent proposes
        2. Competition selects winners
        3. Winners broadcast to all
        4. Agents update their beliefs
        """
        self._step_count += 1

        # 1. Collect proposals from all agents
        all_proposals = []
        for agent in self.agents.values():
            proposals = agent.propose(input_state)
            all_proposals.extend(proposals)

        self._total_proposals += len(all_proposals)

        # 2. Competition - select top proposals
        winners = self._compete(all_proposals)

        # 3. Broadcast winners
        broadcast_count = 0
        winning_proposal = None
        for proposal in winners:
            if proposal.activation >= self.params.ignition_threshold:
                self._broadcast(proposal)
                broadcast_count += 1
                winning_proposal = proposal
                self._broadcast_history.append(proposal)
                self._total_broadcasts += 1

        # 4. Update workspace
        self._workspace = winners[: self.params.workspace_capacity]

        # 5. Process agent internal steps
        for agent in self.agents.values():
            agent.process(dt=0.1)

        # Compute state
        state = PopulationState(
            n_agents=len(self.agents),
            n_proposals=len(all_proposals),
            n_broadcasts=broadcast_count,
            diversity=self.get_diversity(),
            consensus=self.get_consensus(),
            mean_activation=self._get_mean_activation(),
            winning_proposal=winning_proposal,
        )
        self._state_history.append(state)

        return state

    def _compete(self, proposals: List[ModuleProposal]) -> List[ModuleProposal]:
        """Run competition among proposals."""
        if not proposals:
            return []

        # Score proposals
        scored = []
        for p in proposals:
            # Weight by content type
            type_weight = {
                ContentType.VALIDATION: 1.5,
                ContentType.SOLUTION: 1.3,
                ContentType.HYPOTHESIS: 1.1,
                ContentType.OBSERVATION: 1.0,
                ContentType.QUERY: 0.9,
            }.get(p.content_type, 1.0)

            score = p.priority * type_weight
            scored.append((score, p))

        # Sort by score
        scored.sort(key=lambda x: x[0], reverse=True)

        # Apply lateral inhibition - winner suppresses others
        winners = []
        suppression = 1.0
        for score, proposal in scored:
            if score * suppression > 0.1:  # Threshold
                winners.append(proposal)
                suppression *= 0.7  # Each winner suppresses next

        return winners

    def _broadcast(self, proposal: ModuleProposal) -> None:
        """Broadcast a proposal to all agents."""
        for agent in self.agents.values():
            agent.receive_broadcast(proposal)

        # Record acceptance for source agent
        if proposal.source_agent in self.agents:
            self.agents[proposal.source_agent].record_acceptance()

    def get_consensus(self, topic: str = "general") -> float:
        """
        Measure belief consensus across agents.

        Returns value 0-1 where 1 = perfect agreement.
        """
        if len(self.agents) < 2:
            return 1.0

        # Collect belief vectors
        belief_vectors = []
        for agent in self.agents.values():
            if agent.belief_state.beliefs:
                # Use recent beliefs
                values = list(agent.belief_state.beliefs.values())
                # Flatten to vector
                flat = []
                for v in values:
                    if isinstance(v, np.ndarray):
                        flat.extend(v.flatten()[:10])  # Limit size
                    elif isinstance(v, (int, float)):
                        flat.append(v)
                if flat:
                    belief_vectors.append(np.array(flat[:32]))  # Limit size

        if len(belief_vectors) < 2:
            return 0.5

        # Compute mean pairwise similarity
        similarities = []
        for i in range(len(belief_vectors)):
            for j in range(i + 1, len(belief_vectors)):
                v1, v2 = belief_vectors[i], belief_vectors[j]
                # Pad to same length
                max_len = max(len(v1), len(v2))
                v1 = np.pad(v1, (0, max_len - len(v1)))
                v2 = np.pad(v2, (0, max_len - len(v2)))
                # Cosine similarity
                sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                similarities.append((sim + 1) / 2)  # Normalize to 0-1

        return float(np.mean(similarities)) if similarities else 0.5

    def get_diversity(self) -> float:
        """
        Measure belief diversity across agents.

        Returns value 0-1 where 1 = maximum diversity.
        """
        if len(self.agents) < 2:
            return 0.0

        # Use activation levels as proxy for diversity
        activations = [a._activation for a in self.agents.values()]

        # Compute entropy-like measure
        activations = np.array(activations)
        activations = activations / (activations.sum() + 1e-8)  # Normalize
        entropy = -np.sum(activations * np.log(activations + 1e-8))
        max_entropy = np.log(len(self.agents))

        return float(entropy / max_entropy) if max_entropy > 0 else 0.0

    def _get_mean_activation(self) -> float:
        """Get mean activation across agents."""
        if not self.agents:
            return 0.0
        return float(np.mean([a._activation for a in self.agents.values()]))

    def get_agent(self, agent_id: str) -> Optional[CognitiveAgent]:
        """Get an agent by ID."""
        return self.agents.get(agent_id)

    def get_agents_by_role(self, role: AgentRole) -> List[CognitiveAgent]:
        """Get all agents with a specific role."""
        return [a for a in self.agents.values() if a.role == role]

    def deliver_messages(self) -> int:
        """Deliver messages between agents based on topology."""
        if self._topology is None or len(self._topology) == 0:
            return 0

        agent_list = list(self.agents.values())
        delivered = 0

        for i, agent in enumerate(agent_list):
            while not agent.messages_out.empty():
                msg = agent.messages_out.get()
                # Find recipient
                recipient = self.agents.get(msg.recipient_id)
                if recipient:
                    # Check topology
                    j = agent_list.index(recipient) if recipient in agent_list else -1
                    if j >= 0 and i < len(self._topology) and j < len(self._topology):
                        if self._topology[i, j] > 0:
                            recipient.receive_message(msg)
                            delivered += 1

        return delivered

    def get_statistics(self) -> Dict[str, Any]:
        """Get population statistics."""
        role_counts = {}
        for role in AgentRole:
            role_counts[role.value] = len(self.get_agents_by_role(role))

        return {
            "n_agents": len(self.agents),
            "step_count": self._step_count,
            "total_proposals": self._total_proposals,
            "total_broadcasts": self._total_broadcasts,
            "broadcast_rate": self._total_broadcasts / max(1, self._total_proposals),
            "mean_activation": self._get_mean_activation(),
            "diversity": self.get_diversity(),
            "consensus": self.get_consensus(),
            "workspace_size": len(self._workspace),
            "role_counts": role_counts,
        }

    def reset(self) -> None:
        """Reset all agents and statistics."""
        for agent in self.agents.values():
            agent.reset()

        self._workspace = []
        self._broadcast_history = []
        self._state_history = []
        self._step_count = 0
        self._total_proposals = 0
        self._total_broadcasts = 0
