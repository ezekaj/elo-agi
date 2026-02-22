"""
Swarm Intelligence: Orchestrates collective intelligence.

The SwarmIntelligence class coordinates:
- Agent population management
- Emergent coordination
- Collective memory
- Problem solving and learning

Based on:
- SwarmSys (arXiv:2510.10047)
- Emergent Collective Intelligence
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np
import time

from .agent import CognitiveAgent, AgentParams, AgentRole
from .population import AgentPopulation, PopulationParams
from .coordination import (
    EmergentCoordination,
    CoordinationMechanism,
    CoordinationParams,
    Task,
)
from .collective_memory import CollectiveMemory, MemoryParams


class ProblemStatus(Enum):
    """Status of a problem."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SOLVED = "solved"
    FAILED = "failed"


@dataclass
class Problem:
    """A problem to be solved collectively."""

    problem_id: str
    description: str
    input_state: np.ndarray
    target: Optional[np.ndarray] = None
    difficulty: float = 0.5
    max_steps: int = 100
    status: ProblemStatus = ProblemStatus.PENDING


@dataclass
class Solution:
    """A solution produced by the swarm."""

    problem_id: str
    output: np.ndarray
    confidence: float
    steps_taken: int
    contributors: List[str]
    quality: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class SwarmParams:
    """Parameters for swarm intelligence."""

    n_agents: int = 10
    coordination_mechanism: CoordinationMechanism = CoordinationMechanism.BROADCAST
    convergence_threshold: float = 0.9  # Consensus for solution
    exploration_ratio: float = 0.3  # Fraction of explorers
    max_steps_per_problem: int = 100
    learning_rate: float = 0.1


@dataclass
class SwarmState:
    """Current state of the swarm."""

    n_agents: int
    n_problems_solved: int
    mean_solution_quality: float
    collective_memory_size: int
    diversity: float
    consensus: float
    timestamp: float = field(default_factory=time.time)


# Default role distribution based on exploration ratio
DEFAULT_ROLE_DIST = {
    AgentRole.EXPLORER: 0.2,
    AgentRole.WORKER: 0.5,
    AgentRole.VALIDATOR: 0.2,
    AgentRole.GENERALIST: 0.1,
}


class SwarmIntelligence:
    """
    Orchestrates collective intelligence from multiple agents.

    The swarm combines:
    - Agent population with diverse roles
    - Emergent coordination mechanisms
    - Collective memory for knowledge sharing

    This enables collective problem solving where
    the group intelligence exceeds individual capabilities.
    """

    def __init__(self, params: Optional[SwarmParams] = None):
        self.params = params or SwarmParams()

        # Build role distribution based on exploration ratio
        role_dist = self._build_role_distribution()

        # Initialize components
        self.population = AgentPopulation(
            PopulationParams(
                n_agents=self.params.n_agents,
                role_distribution=role_dist,
            )
        )

        self.coordination = EmergentCoordination(
            CoordinationParams(
                mechanism=self.params.coordination_mechanism,
            )
        )

        self.collective_memory = CollectiveMemory(
            MemoryParams(
                capacity=1000,
                decay_rate=0.01,
            )
        )

        # Problem tracking
        self._current_problem: Optional[Problem] = None
        self._solutions: List[Solution] = []
        self._problem_history: List[Problem] = []

        # Statistics
        self._step_count = 0
        self._problems_solved = 0
        self._total_quality = 0.0

    def _build_role_distribution(self) -> Dict[AgentRole, float]:
        """Build role distribution based on exploration ratio."""
        exp_ratio = self.params.exploration_ratio
        work_ratio = 1.0 - exp_ratio

        return {
            AgentRole.EXPLORER: exp_ratio * 0.7,
            AgentRole.WORKER: work_ratio * 0.6,
            AgentRole.VALIDATOR: work_ratio * 0.3,
            AgentRole.GENERALIST: exp_ratio * 0.3,
        }

    def solve(self, problem: Problem) -> Solution:
        """
        Collective problem solving.

        Process:
        1. Distribute problem to agents
        2. Agents explore in parallel
        3. Share discoveries via collective memory
        4. Coordinate on promising solutions
        5. Validate best candidates
        6. Return consensus solution
        """
        self._current_problem = problem
        problem.status = ProblemStatus.IN_PROGRESS

        # Store problem in collective memory
        self.collective_memory.store(
            agent_id="system",
            key=f"problem_{problem.problem_id}",
            value=problem.input_state,
            confidence=1.0,
        )

        # Assign roles based on problem
        task = Task(
            task_id=problem.problem_id,
            description=problem.description,
            requirements=problem.input_state,
            difficulty=problem.difficulty,
        )
        role_assignments = self.coordination.role_assignment(
            list(self.population.agents.values()), task
        )

        # Update agent roles
        for agent_id, role in role_assignments.items():
            if agent_id in self.population.agents:
                self.population.agents[agent_id].params.role = role

        # Iterative solving
        best_output = problem.input_state.copy()
        best_confidence = 0.0
        contributors: List[str] = []

        for step in range(min(problem.max_steps, self.params.max_steps_per_problem)):
            self._step_count += 1

            # Run population step
            state = self.population.step(best_output)

            # Coordinate actions
            actions = self.coordination.coordinate(
                list(self.population.agents.values()),
                problem.target if problem.target is not None else problem.input_state,
            )

            # Process actions and update best output
            for action in actions:
                agent = self.population.agents.get(action.agent_id)
                if agent and agent._proposal_history:
                    proposal = agent._proposal_history[-1]

                    # Store in collective memory
                    self.collective_memory.store(
                        agent_id=action.agent_id,
                        key=f"step_{step}_{action.agent_id}",
                        value=proposal.content,
                        confidence=proposal.confidence,
                    )

                    # Update best if better
                    if proposal.confidence > best_confidence:
                        best_output = proposal.content
                        best_confidence = proposal.confidence
                        if action.agent_id not in contributors:
                            contributors.append(action.agent_id)

            # Check for convergence
            if state.consensus >= self.params.convergence_threshold:
                break

            # Detect emergent patterns
            self.coordination.detect_emergence(list(self.population.agents.values()))

        # Compute solution quality
        quality = self._compute_quality(problem, best_output)

        # Create solution
        solution = Solution(
            problem_id=problem.problem_id,
            output=best_output,
            confidence=best_confidence,
            steps_taken=step + 1,
            contributors=contributors,
            quality=quality,
        )

        # Update tracking
        problem.status = ProblemStatus.SOLVED if quality > 0.5 else ProblemStatus.FAILED
        self._solutions.append(solution)
        self._problem_history.append(problem)

        if problem.status == ProblemStatus.SOLVED:
            self._problems_solved += 1
            self._total_quality += quality

        return solution

    def _compute_quality(self, problem: Problem, output: np.ndarray) -> float:
        """Compute solution quality."""
        if problem.target is not None:
            # Compare to target
            if len(output) != len(problem.target):
                output = np.resize(output, len(problem.target))

            # Cosine similarity
            sim = np.dot(output, problem.target) / (
                np.linalg.norm(output) * np.linalg.norm(problem.target) + 1e-8
            )
            return float((sim + 1) / 2)  # Normalize to 0-1

        else:
            # Heuristic quality
            return float(np.clip(np.mean(np.abs(output)), 0, 1))

    def learn(self, experience: Tuple[Problem, Solution, float]) -> None:
        """
        Collective learning from experience.

        Updates agent parameters based on solution quality.
        """
        problem, solution, reward = experience

        # Store successful solutions in collective memory
        if reward > 0.5:
            self.collective_memory.store(
                agent_id="system",
                key=f"successful_{problem.problem_id}",
                value=solution.output,
                confidence=reward,
            )

        # Update contributing agents
        for agent_id in solution.contributors:
            agent = self.population.agents.get(agent_id)
            if agent:
                # Boost or penalize based on reward
                agent._activation = np.clip(
                    agent._activation + (reward - 0.5) * self.params.learning_rate, 0.1, 1.0
                )

                # Record contribution
                agent.record_acceptance(reward)

        # Consolidate collective memory
        self.collective_memory.consolidate()
        self.collective_memory.forget()

    def get_collective_belief(self, topic: str) -> np.ndarray:
        """Aggregate beliefs across population."""
        # Query collective memory
        query = np.zeros(64)
        query[hash(topic) % 64] = 1.0

        memories = self.collective_memory.retrieve(query, n=10)

        if not memories:
            return np.zeros(64)

        # Weighted average of memory values
        belief = np.zeros(64)
        total_weight = 0.0

        for mem in memories:
            if isinstance(mem.value, np.ndarray):
                value = np.resize(mem.value, 64)
                weight = mem.confidence
                belief += value * weight
                total_weight += weight

        if total_weight > 0:
            belief /= total_weight

        return belief

    def add_agent(self, params: Optional[AgentParams] = None) -> CognitiveAgent:
        """Add a new agent to the swarm."""
        if params is None:
            params = AgentParams(role=AgentRole.GENERALIST)
        return self.population.spawn_agent(params)

    def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent from the swarm."""
        return self.population.remove_agent(agent_id)

    def get_state(self) -> SwarmState:
        """Get current swarm state."""
        mean_quality = (
            self._total_quality / self._problems_solved if self._problems_solved > 0 else 0.0
        )

        return SwarmState(
            n_agents=len(self.population.agents),
            n_problems_solved=self._problems_solved,
            mean_solution_quality=mean_quality,
            collective_memory_size=len(self.collective_memory._memories),
            diversity=self.population.get_diversity(),
            consensus=self.population.get_consensus(),
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive swarm statistics."""
        state = self.get_state()

        return {
            "n_agents": state.n_agents,
            "step_count": self._step_count,
            "problems_solved": self._problems_solved,
            "mean_solution_quality": state.mean_solution_quality,
            "collective_memory_size": state.collective_memory_size,
            "diversity": state.diversity,
            "consensus": state.consensus,
            "population_stats": self.population.get_statistics(),
            "coordination_stats": self.coordination.get_statistics(),
            "memory_stats": self.collective_memory.get_statistics(),
        }

    def get_top_performers(self, n: int = 5) -> List[Tuple[str, Dict[str, Any]]]:
        """Get top performing agents."""
        agents_stats = [(a.agent_id, a.get_statistics()) for a in self.population.agents.values()]

        # Sort by total contribution
        agents_stats.sort(key=lambda x: x[1]["total_contribution"], reverse=True)

        return agents_stats[:n]

    def reset(self) -> None:
        """Reset the entire swarm."""
        self.population.reset()
        self.coordination.reset()
        self.collective_memory.reset()

        self._current_problem = None
        self._solutions = []
        self._problem_history = []
        self._step_count = 0
        self._problems_solved = 0
        self._total_quality = 0.0
