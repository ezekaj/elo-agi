"""
Emergent Coordination: Enables collective behavior without central control.

Implements multiple coordination mechanisms:
- Stigmergy: Indirect coordination through environment
- Direct: Explicit message passing
- Broadcast: One-to-all communication
- Consensus: Voting and agreement protocols

Based on:
- SwarmSys coordination patterns (arXiv:2510.10047)
- Emergent coordination in multi-agent systems
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
import numpy as np
import time

from .agent import CognitiveAgent, AgentRole, ModuleProposal, ContentType


class CoordinationMechanism(Enum):
    """Types of coordination mechanisms."""
    STIGMERGY = "stigmergy"      # Indirect via environment
    DIRECT = "direct"            # Explicit messages
    BROADCAST = "broadcast"      # One-to-all
    CONSENSUS = "consensus"      # Voting/agreement


@dataclass
class Action:
    """An action to be taken by an agent."""
    agent_id: str
    action_type: str
    parameters: Dict[str, Any]
    priority: float = 0.5
    timestamp: float = field(default_factory=time.time)


@dataclass
class Task:
    """A task to be solved collectively."""
    task_id: str
    description: str
    requirements: np.ndarray  # Required capabilities
    difficulty: float = 0.5
    deadline: Optional[float] = None


@dataclass
class EmergentPattern:
    """A detected emergent coordination pattern."""
    pattern_type: str
    agents_involved: List[str]
    strength: float
    duration: float
    description: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class CoordinationParams:
    """Parameters for coordination."""
    mechanism: CoordinationMechanism = CoordinationMechanism.BROADCAST
    consensus_threshold: float = 0.6  # Required agreement level
    voting_rounds: int = 3
    stigmergy_decay: float = 0.1
    detection_window: float = 10.0  # Seconds to look back for patterns


class EmergentCoordination:
    """
    Enables emergent collective behavior.

    The coordination system allows agents to:
    - Coordinate without explicit central control
    - Detect and leverage emergent patterns
    - Dynamically assign roles based on tasks
    """

    def __init__(self, params: Optional[CoordinationParams] = None):
        self.params = params or CoordinationParams()

        # Coordination state
        self._environment: Dict[str, Any] = {}  # For stigmergy
        self._vote_history: List[Tuple[str, Dict[str, float]]] = []
        self._detected_patterns: List[EmergentPattern] = []

        # Statistics
        self._coordination_count = 0
        self._pattern_count = 0

    def coordinate(
        self,
        agents: List[CognitiveAgent],
        goal: np.ndarray,
    ) -> List[Action]:
        """
        Produce coordinated actions without central control.

        Args:
            agents: List of agents to coordinate
            goal: Target goal vector

        Returns:
            List of actions for agents to execute
        """
        self._coordination_count += 1

        if self.params.mechanism == CoordinationMechanism.STIGMERGY:
            return self._coordinate_stigmergy(agents, goal)
        elif self.params.mechanism == CoordinationMechanism.DIRECT:
            return self._coordinate_direct(agents, goal)
        elif self.params.mechanism == CoordinationMechanism.BROADCAST:
            return self._coordinate_broadcast(agents, goal)
        elif self.params.mechanism == CoordinationMechanism.CONSENSUS:
            return self._coordinate_consensus(agents, goal)
        else:
            return []

    def _coordinate_stigmergy(
        self,
        agents: List[CognitiveAgent],
        goal: np.ndarray,
    ) -> List[Action]:
        """Coordinate through environmental traces."""
        actions = []

        for agent in agents:
            # Read from environment
            env_state = self._read_environment(agent.agent_id)

            # Compute action based on goal and environment
            if agent.role == AgentRole.EXPLORER:
                # Explore unexplored areas
                action = self._explore_action(agent, goal, env_state)
            elif agent.role == AgentRole.WORKER:
                # Follow strong traces
                action = self._follow_action(agent, goal, env_state)
            else:
                # Default action
                action = self._default_action(agent, goal)

            actions.append(action)

            # Leave trace in environment
            self._leave_trace(agent.agent_id, action)

        # Decay environment
        self._decay_environment()

        return actions

    def _coordinate_direct(
        self,
        agents: List[CognitiveAgent],
        goal: np.ndarray,
    ) -> List[Action]:
        """Coordinate through direct messages."""
        actions = []

        # Agents negotiate in rounds
        for _ in range(2):  # Negotiation rounds
            for agent in agents:
                # Broadcast intention
                intention = self._compute_intention(agent, goal)
                for other in agents:
                    if other.agent_id != agent.agent_id:
                        agent.send_message(
                            other.agent_id,
                            intention,
                            ContentType.OBSERVATION
                        )

        # Compute final actions based on negotiation
        for agent in agents:
            agent.process_messages()
            action = self._negotiated_action(agent, goal)
            actions.append(action)

        return actions

    def _coordinate_broadcast(
        self,
        agents: List[CognitiveAgent],
        goal: np.ndarray,
    ) -> List[Action]:
        """Coordinate through broadcast mechanism."""
        actions = []

        # Collect proposals
        proposals = []
        for agent in agents:
            agent_proposals = agent.propose(goal)
            proposals.extend(agent_proposals)

        # Select best proposal
        if proposals:
            best = max(proposals, key=lambda p: p.priority)

            # All agents align with best proposal
            for agent in agents:
                agent.receive_broadcast(best)
                action = self._aligned_action(agent, goal, best)
                actions.append(action)
        else:
            # Default actions
            for agent in agents:
                actions.append(self._default_action(agent, goal))

        return actions

    def _coordinate_consensus(
        self,
        agents: List[CognitiveAgent],
        goal: np.ndarray,
    ) -> List[Action]:
        """Coordinate through voting consensus."""
        actions = []

        # Generate proposals from each agent
        proposals = {}
        for agent in agents:
            agent_proposals = agent.propose(goal)
            if agent_proposals:
                proposals[agent.agent_id] = agent_proposals[0]

        if not proposals:
            return [self._default_action(a, goal) for a in agents]

        # Voting rounds
        votes: Dict[str, float] = {pid: 0.0 for pid in proposals.keys()}

        for _ in range(self.params.voting_rounds):
            for agent in agents:
                # Each agent votes on all proposals
                for pid, proposal in proposals.items():
                    vote = agent.evaluate(proposal)
                    weight = agent.expertise_on("general")
                    votes[pid] += vote * weight

        # Record votes
        self._vote_history.append((time.time(), votes.copy()))

        # Find winner
        if votes:
            winner_id = max(votes.keys(), key=lambda k: votes[k])
            winner = proposals[winner_id]

            # Check consensus threshold
            max_vote = votes[winner_id]
            total_vote = sum(votes.values())
            consensus = max_vote / total_vote if total_vote > 0 else 0

            if consensus >= self.params.consensus_threshold:
                # Strong consensus - all align
                for agent in agents:
                    action = self._aligned_action(agent, goal, winner)
                    actions.append(action)
            else:
                # Weak consensus - agents act independently
                for agent in agents:
                    action = self._independent_action(agent, goal)
                    actions.append(action)
        else:
            actions = [self._default_action(a, goal) for a in agents]

        return actions

    def _read_environment(self, agent_id: str) -> Dict[str, Any]:
        """Read relevant environment state for an agent."""
        return {k: v for k, v in self._environment.items()
                if not k.startswith(agent_id)}

    def _leave_trace(self, agent_id: str, action: Action) -> None:
        """Leave a trace in the environment."""
        key = f"{agent_id}_trace_{time.time()}"
        self._environment[key] = {
            'action': action.action_type,
            'params': action.parameters,
            'strength': 1.0,
        }

    def _decay_environment(self) -> None:
        """Decay environmental traces."""
        keys_to_remove = []
        for key, value in self._environment.items():
            if isinstance(value, dict) and 'strength' in value:
                value['strength'] *= (1 - self.params.stigmergy_decay)
                if value['strength'] < 0.01:
                    keys_to_remove.append(key)

        for key in keys_to_remove:
            del self._environment[key]

    def _explore_action(
        self,
        agent: CognitiveAgent,
        goal: np.ndarray,
        env_state: Dict,
    ) -> Action:
        """Generate exploration action."""
        # Move away from existing traces
        direction = goal.copy()
        for trace in env_state.values():
            if isinstance(trace, dict):
                direction += np.random.randn(len(goal)) * 0.1

        return Action(
            agent_id=agent.agent_id,
            action_type="explore",
            parameters={'direction': direction.tolist()},
            priority=agent.params.curiosity,
        )

    def _follow_action(
        self,
        agent: CognitiveAgent,
        goal: np.ndarray,
        env_state: Dict,
    ) -> Action:
        """Generate action following strong traces."""
        # Find strongest trace
        strongest = None
        max_strength = 0

        for trace in env_state.values():
            if isinstance(trace, dict) and trace.get('strength', 0) > max_strength:
                max_strength = trace['strength']
                strongest = trace

        if strongest and 'params' in strongest:
            params = strongest['params']
        else:
            params = {'direction': goal.tolist()}

        return Action(
            agent_id=agent.agent_id,
            action_type="follow",
            parameters=params,
            priority=max_strength,
        )

    def _compute_intention(
        self,
        agent: CognitiveAgent,
        goal: np.ndarray,
    ) -> np.ndarray:
        """Compute agent's intention vector."""
        spec = np.resize(agent.params.specialization, len(goal))
        return goal * spec / (np.linalg.norm(spec) + 1e-8)

    def _negotiated_action(
        self,
        agent: CognitiveAgent,
        goal: np.ndarray,
    ) -> Action:
        """Generate action based on negotiation."""
        return Action(
            agent_id=agent.agent_id,
            action_type="execute",
            parameters={'goal': goal.tolist()},
            priority=agent._activation,
        )

    def _aligned_action(
        self,
        agent: CognitiveAgent,
        goal: np.ndarray,
        proposal: ModuleProposal,
    ) -> Action:
        """Generate action aligned with winning proposal."""
        return Action(
            agent_id=agent.agent_id,
            action_type="align",
            parameters={
                'goal': goal.tolist(),
                'alignment': proposal.source_agent,
            },
            priority=proposal.priority,
        )

    def _independent_action(
        self,
        agent: CognitiveAgent,
        goal: np.ndarray,
    ) -> Action:
        """Generate independent action."""
        spec = np.resize(agent.params.specialization, len(goal))
        direction = goal * spec

        return Action(
            agent_id=agent.agent_id,
            action_type="independent",
            parameters={'direction': direction.tolist()},
            priority=agent._activation,
        )

    def _default_action(
        self,
        agent: CognitiveAgent,
        goal: np.ndarray,
    ) -> Action:
        """Generate default action."""
        return Action(
            agent_id=agent.agent_id,
            action_type="default",
            parameters={'goal': goal.tolist()},
            priority=0.5,
        )

    def detect_emergence(
        self,
        agents: List[CognitiveAgent],
    ) -> List[EmergentPattern]:
        """
        Detect emergent coordination patterns.

        Looks for:
        - Synchronized behavior
        - Role specialization
        - Division of labor
        - Collective memory formation
        """
        patterns = []

        # Check for synchronization
        sync_pattern = self._detect_synchronization(agents)
        if sync_pattern:
            patterns.append(sync_pattern)

        # Check for role clustering
        role_pattern = self._detect_role_clustering(agents)
        if role_pattern:
            patterns.append(role_pattern)

        # Check for division of labor
        labor_pattern = self._detect_division_of_labor(agents)
        if labor_pattern:
            patterns.append(labor_pattern)

        self._detected_patterns.extend(patterns)
        self._pattern_count += len(patterns)

        return patterns

    def _detect_synchronization(
        self,
        agents: List[CognitiveAgent],
    ) -> Optional[EmergentPattern]:
        """Detect synchronized behavior."""
        if len(agents) < 2:
            return None

        # Check activation synchrony
        activations = [a._activation for a in agents]
        mean_act = np.mean(activations)
        std_act = np.std(activations)

        # Low variance = synchronization
        if std_act < 0.1 and mean_act > 0.3:
            return EmergentPattern(
                pattern_type="synchronization",
                agents_involved=[a.agent_id for a in agents],
                strength=1.0 - std_act,
                duration=self.params.detection_window,
                description=f"Agents synchronized at activation {mean_act:.2f}",
            )
        return None

    def _detect_role_clustering(
        self,
        agents: List[CognitiveAgent],
    ) -> Optional[EmergentPattern]:
        """Detect role-based clustering."""
        role_groups: Dict[AgentRole, List[str]] = {}

        for agent in agents:
            if agent.role not in role_groups:
                role_groups[agent.role] = []
            role_groups[agent.role].append(agent.agent_id)

        # Check if roles are performing differently
        role_activations = {}
        for role, agent_ids in role_groups.items():
            acts = [a._activation for a in agents if a.agent_id in agent_ids]
            if acts:
                role_activations[role] = np.mean(acts)

        if len(role_activations) >= 2:
            variance = np.var(list(role_activations.values()))
            if variance > 0.05:  # Significant role differentiation
                return EmergentPattern(
                    pattern_type="role_clustering",
                    agents_involved=[a.agent_id for a in agents],
                    strength=float(variance),
                    duration=self.params.detection_window,
                    description=f"Roles showing differentiated behavior",
                )
        return None

    def _detect_division_of_labor(
        self,
        agents: List[CognitiveAgent],
    ) -> Optional[EmergentPattern]:
        """Detect division of labor."""
        # Check proposal type diversity
        proposal_types: Dict[str, ContentType] = {}

        for agent in agents:
            if agent._proposal_history:
                recent = agent._proposal_history[-1]
                proposal_types[agent.agent_id] = recent.content_type

        if len(proposal_types) >= 2:
            unique_types = len(set(proposal_types.values()))
            if unique_types >= 2:
                return EmergentPattern(
                    pattern_type="division_of_labor",
                    agents_involved=list(proposal_types.keys()),
                    strength=unique_types / len(ContentType),
                    duration=self.params.detection_window,
                    description=f"Agents specializing in {unique_types} different content types",
                )
        return None

    def role_assignment(
        self,
        agents: List[CognitiveAgent],
        task: Task,
    ) -> Dict[str, AgentRole]:
        """
        Dynamically assign roles based on task requirements.

        Uses fitness scoring:
        - Explorers: high curiosity, low expertise on task
        - Workers: matched specialization
        - Validators: high confidence, diverse perspective
        """
        assignments = {}

        for agent in agents:
            # Compute fitness for each role
            spec = np.resize(agent.params.specialization, len(task.requirements))

            # Explorer fitness: curiosity * (1 - expertise)
            expertise = float(np.dot(spec, task.requirements) / (np.linalg.norm(spec) * np.linalg.norm(task.requirements) + 1e-8))
            expertise = (expertise + 1) / 2  # Normalize
            fit_explorer = agent.params.curiosity * (1 - expertise)

            # Worker fitness: specialization match
            fit_worker = expertise

            # Validator fitness: confidence * diversity
            diversity = len(agent.other_models) / max(1, len(agents) - 1)
            fit_validator = agent._activation * diversity

            # Generalist: average
            fit_generalist = (fit_explorer + fit_worker + fit_validator) / 3

            # Assign best fitting role
            fits = {
                AgentRole.EXPLORER: fit_explorer,
                AgentRole.WORKER: fit_worker,
                AgentRole.VALIDATOR: fit_validator,
                AgentRole.GENERALIST: fit_generalist,
            }
            best_role = max(fits.keys(), key=lambda r: fits[r])
            assignments[agent.agent_id] = best_role

        return assignments

    def get_statistics(self) -> Dict[str, Any]:
        """Get coordination statistics."""
        return {
            'mechanism': self.params.mechanism.value,
            'coordination_count': self._coordination_count,
            'pattern_count': self._pattern_count,
            'environment_size': len(self._environment),
            'vote_history_size': len(self._vote_history),
            'detected_patterns': len(self._detected_patterns),
        }

    def reset(self) -> None:
        """Reset coordination state."""
        self._environment = {}
        self._vote_history = []
        self._detected_patterns = []
        self._coordination_count = 0
        self._pattern_count = 0
