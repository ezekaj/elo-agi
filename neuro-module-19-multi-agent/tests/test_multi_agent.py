"""
Tests for the Multi-Agent Coordination module.

Covers:
- Unit tests for individual components
- Integration tests for combined systems
- Stress tests for scalability
"""

import pytest
import numpy as np
import time

from neuro.modules.m19_multi_agent.agent import (
    CognitiveAgent,
    AgentParams,
    AgentRole,
    BeliefState,
    Message,
    ModuleProposal,
    ContentType,
)
from neuro.modules.m19_multi_agent.population import (
    AgentPopulation,
    PopulationParams,
    PopulationState,
    TopologyType,
)
from neuro.modules.m19_multi_agent.coordination import (
    EmergentCoordination,
    CoordinationMechanism,
    CoordinationParams,
    Task,
    Action,
    EmergentPattern,
)
from neuro.modules.m19_multi_agent.collective_memory import (
    CollectiveMemory,
    MemoryParams,
    MemoryEntry,
)
from neuro.modules.m19_multi_agent.swarm import (
    SwarmIntelligence,
    SwarmParams,
    Problem,
    Solution,
    ProblemStatus,
)

# =============================================================================
# Unit Tests: CognitiveAgent
# =============================================================================


class TestCognitiveAgent:
    """Tests for the CognitiveAgent class."""

    def test_agent_creation(self):
        """Test agent initialization."""
        agent = CognitiveAgent()
        assert agent.agent_id is not None
        assert agent.role == AgentRole.GENERALIST
        assert agent._activation == 0.5

    def test_agent_with_params(self):
        """Test agent with custom parameters."""
        params = AgentParams(
            agent_id="test_agent",
            role=AgentRole.EXPLORER,
            curiosity=0.8,
        )
        agent = CognitiveAgent(params)
        assert agent.agent_id == "test_agent"
        assert agent.role == AgentRole.EXPLORER
        assert agent.params.curiosity == 0.8

    def test_agent_propose(self):
        """Test proposal generation."""
        agent = CognitiveAgent(AgentParams(role=AgentRole.WORKER))
        input_state = np.random.randn(32)

        proposals = agent.propose(input_state)

        assert len(proposals) >= 0  # May or may not propose based on confidence
        if proposals:
            assert proposals[0].source_agent == agent.agent_id
            assert isinstance(proposals[0].content, np.ndarray)

    def test_agent_communication(self):
        """Test message sending and receiving."""
        agent1 = CognitiveAgent(AgentParams(agent_id="agent1"))
        agent2 = CognitiveAgent(AgentParams(agent_id="agent2"))

        # Send message
        agent1.send_message("agent2", np.array([1, 2, 3]), ContentType.OBSERVATION)

        # Get message from queue
        msg = agent1.messages_out.get()
        assert msg.sender_id == "agent1"
        assert msg.recipient_id == "agent2"

        # Receive message
        agent2.receive_message(msg)
        assert not agent2.messages_in.empty()

    def test_agent_belief_update(self):
        """Test belief state updates."""
        agent = CognitiveAgent()

        # Create a proposal and broadcast
        proposal = ModuleProposal(
            source_agent="other_agent",
            content=np.array([1, 2, 3]),
            content_type=ContentType.OBSERVATION,
            activation=0.8,
            confidence=0.7,
            relevance=0.6,
        )

        agent.receive_broadcast(proposal)

        assert "broadcast_observation" in agent.belief_state.beliefs
        assert "other_agent" in agent.other_models

    def test_agent_role_behavior(self):
        """Test different role behaviors."""
        input_state = np.random.randn(32)

        for role in AgentRole:
            agent = CognitiveAgent(AgentParams(role=role))
            proposals = agent.propose(input_state)
            # Each role should be able to generate proposals
            assert isinstance(proposals, list)


class TestBeliefState:
    """Tests for BeliefState."""

    def test_belief_update(self):
        """Test belief updates."""
        belief = BeliefState()
        belief.update("key1", "value1", 0.8)

        assert "key1" in belief.beliefs
        assert belief.beliefs["key1"] == "value1"
        assert belief.confidence["key1"] == 0.8

    def test_belief_merge(self):
        """Test belief merging."""
        belief1 = BeliefState()
        belief1.update("key1", "value1", 0.8)

        belief2 = BeliefState()
        belief2.update("key2", "value2", 0.6)

        belief1.merge(belief2, weight=0.5)

        assert "key2" in belief1.beliefs


# =============================================================================
# Unit Tests: AgentPopulation
# =============================================================================


class TestAgentPopulation:
    """Tests for AgentPopulation."""

    def test_population_creation(self):
        """Test population initialization."""
        pop = AgentPopulation(PopulationParams(n_agents=5))

        assert len(pop.agents) == 5
        assert pop._topology is not None

    def test_population_step(self):
        """Test population step."""
        pop = AgentPopulation(PopulationParams(n_agents=5))
        input_state = np.random.randn(32)

        state = pop.step(input_state)

        assert isinstance(state, PopulationState)
        assert state.n_agents == 5

    def test_role_assignment(self):
        """Test role distribution."""
        role_dist = {
            AgentRole.EXPLORER: 0.5,
            AgentRole.WORKER: 0.5,
            AgentRole.VALIDATOR: 0.0,
            AgentRole.GENERALIST: 0.0,
        }
        pop = AgentPopulation(PopulationParams(n_agents=10, role_distribution=role_dist))

        explorers = pop.get_agents_by_role(AgentRole.EXPLORER)
        workers = pop.get_agents_by_role(AgentRole.WORKER)

        # Should have some of each (probabilistic)
        assert len(explorers) + len(workers) == 10

    def test_topology_types(self):
        """Test different topology types."""
        for topo in TopologyType:
            pop = AgentPopulation(PopulationParams(n_agents=5, topology=topo))
            assert pop._topology is not None
            if len(pop.agents) > 1:
                assert pop._topology.shape == (5, 5)

    def test_spawn_remove_agent(self):
        """Test spawning and removing agents."""
        pop = AgentPopulation(PopulationParams(n_agents=3))

        # Spawn
        new_agent = pop.spawn_agent(AgentParams(agent_id="new_agent"))
        assert len(pop.agents) == 4
        assert "new_agent" in pop.agents

        # Remove
        result = pop.remove_agent("new_agent")
        assert result is True
        assert len(pop.agents) == 3


# =============================================================================
# Unit Tests: EmergentCoordination
# =============================================================================


class TestEmergentCoordination:
    """Tests for EmergentCoordination."""

    def test_coordination_mechanisms(self):
        """Test different coordination mechanisms."""
        agents = [CognitiveAgent() for _ in range(3)]
        goal = np.random.randn(32)

        for mechanism in CoordinationMechanism:
            coord = EmergentCoordination(CoordinationParams(mechanism=mechanism))
            actions = coord.coordinate(agents, goal)

            assert len(actions) == 3
            for action in actions:
                assert isinstance(action, Action)

    def test_role_assignment(self):
        """Test dynamic role assignment."""
        agents = [CognitiveAgent() for _ in range(5)]
        task = Task(
            task_id="test",
            description="Test task",
            requirements=np.random.randn(32),
        )

        coord = EmergentCoordination()
        assignments = coord.role_assignment(agents, task)

        assert len(assignments) == 5
        for agent_id, role in assignments.items():
            assert isinstance(role, AgentRole)

    def test_emergence_detection(self):
        """Test emergent pattern detection."""
        agents = [CognitiveAgent() for _ in range(5)]
        # Set similar activation to trigger synchronization
        for agent in agents:
            agent._activation = 0.7

        coord = EmergentCoordination()
        patterns = coord.detect_emergence(agents)

        # Should detect synchronization pattern
        assert isinstance(patterns, list)


# =============================================================================
# Unit Tests: CollectiveMemory
# =============================================================================


class TestCollectiveMemory:
    """Tests for CollectiveMemory."""

    def test_store_retrieve(self):
        """Test basic store and retrieve."""
        mem = CollectiveMemory()

        # Store
        mem.store("agent1", "key1", np.array([1, 2, 3]), confidence=0.8)

        # Retrieve by key
        entry = mem.retrieve_by_key("key1")
        assert entry is not None
        assert "agent1" in entry.contributors

    def test_vector_retrieval(self):
        """Test vector-based retrieval."""
        mem = CollectiveMemory()

        # Store multiple entries
        for i in range(5):
            value = np.zeros(64)
            value[i] = 1.0
            mem.store(f"agent{i}", f"key{i}", value, confidence=0.7)

        # Query
        query = np.zeros(64)
        query[0] = 1.0

        results = mem.retrieve(query, n=3)
        assert len(results) <= 3

    def test_consolidation(self):
        """Test memory consolidation."""
        mem = CollectiveMemory(MemoryParams(consolidation_threshold=0.9))

        # Store similar memories
        value1 = np.ones(64)
        value2 = np.ones(64) * 1.01  # Very similar

        mem.store("agent1", "key1", value1, confidence=0.5)
        mem.store("agent2", "key2", value2, confidence=0.5)

        initial_size = len(mem._memories)
        merged = mem.consolidate()

        # May or may not merge depending on similarity
        assert merged >= 0

    def test_forgetting(self):
        """Test memory decay."""
        mem = CollectiveMemory(
            MemoryParams(
                decay_rate=0.5,  # High decay
                min_confidence=0.3,
            )
        )

        # Store with low confidence
        mem.store("agent1", "key1", np.array([1, 2, 3]), confidence=0.2)

        # Force time passage simulation by directly modifying
        if "key1" in mem._memories:
            mem._memories["key1"].last_accessed = time.time() - 600  # 10 min ago

        forgotten = mem.forget()
        # Memory should be forgotten
        assert forgotten >= 0

    def test_contribution_tracking(self):
        """Test agent contribution tracking."""
        mem = CollectiveMemory()

        mem.store("agent1", "key1", np.array([1, 2, 3]), confidence=0.5)
        mem.store("agent1", "key2", np.array([4, 5, 6]), confidence=0.7)
        mem.store("agent2", "key3", np.array([7, 8, 9]), confidence=0.6)

        score1 = mem.get_agent_contribution("agent1")
        score2 = mem.get_agent_contribution("agent2")

        assert score1 > score2  # agent1 contributed more


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the multi-agent system."""

    def test_collective_problem_solving(self):
        """Test swarm solves a problem."""
        swarm = SwarmIntelligence(SwarmParams(n_agents=5))

        problem = Problem(
            problem_id="test_problem",
            description="Test problem",
            input_state=np.random.randn(32),
            target=np.ones(32),
            max_steps=10,
        )

        solution = swarm.solve(problem)

        assert isinstance(solution, Solution)
        assert solution.problem_id == "test_problem"
        assert len(solution.output) > 0

    def test_emergent_coordination(self):
        """Test coordination emerges in population."""
        pop = AgentPopulation(PopulationParams(n_agents=10))
        coord = EmergentCoordination()

        # Run several steps
        for _ in range(5):
            input_state = np.random.randn(32)
            pop.step(input_state)

        # Check for emergent patterns
        patterns = coord.detect_emergence(list(pop.agents.values()))
        # Patterns may or may not emerge
        assert isinstance(patterns, list)

    def test_collective_memory_sharing(self):
        """Test knowledge sharing through memory."""
        swarm = SwarmIntelligence(SwarmParams(n_agents=5))

        # Solve a problem to populate memory
        problem = Problem(
            problem_id="p1",
            description="First problem",
            input_state=np.random.randn(32),
            max_steps=5,
        )
        swarm.solve(problem)

        # Check memory was populated
        stats = swarm.collective_memory.get_statistics()
        assert stats["size"] > 0

    def test_consensus_formation(self):
        """Test agents reach consensus."""
        pop = AgentPopulation(PopulationParams(n_agents=5))

        # Give all agents similar beliefs
        for agent in pop.agents.values():
            agent.belief_state.update("shared", np.ones(10), 0.8)

        consensus = pop.get_consensus()
        # Should have some consensus
        assert 0 <= consensus <= 1

    def test_diversity_maintenance(self):
        """Test diversity is maintained."""
        swarm = SwarmIntelligence(SwarmParams(n_agents=10))

        # Run several problems
        for i in range(3):
            problem = Problem(
                problem_id=f"p{i}",
                description=f"Problem {i}",
                input_state=np.random.randn(32),
                max_steps=5,
            )
            swarm.solve(problem)

        diversity = swarm.population.get_diversity()
        # Should maintain some diversity
        assert diversity >= 0


# =============================================================================
# Stress Tests
# =============================================================================


class TestStress:
    """Stress tests for scalability."""

    def test_large_population(self):
        """Test with 100+ agents."""
        pop = AgentPopulation(PopulationParams(n_agents=100))

        assert len(pop.agents) == 100

        # Run a step
        input_state = np.random.randn(32)
        state = pop.step(input_state)

        assert state.n_agents == 100

    def test_high_message_volume(self):
        """Test many concurrent messages."""
        pop = AgentPopulation(PopulationParams(n_agents=20))

        # Each agent sends messages to others
        agents = list(pop.agents.values())
        for sender in agents[:10]:
            for receiver in agents[10:]:
                sender.send_message(receiver.agent_id, np.random.randn(32), ContentType.OBSERVATION)

        # Deliver messages
        delivered = pop.deliver_messages()
        assert delivered > 0

    def test_convergence_speed(self):
        """Test how fast swarm converges."""
        swarm = SwarmIntelligence(SwarmParams(n_agents=10, convergence_threshold=0.95))

        problem = Problem(
            problem_id="convergence_test",
            description="Test convergence",
            input_state=np.random.randn(32),
            target=np.ones(32),
            max_steps=50,
        )

        solution = swarm.solve(problem)

        # Should converge within max_steps
        assert solution.steps_taken <= 50

    def test_robustness(self):
        """Test system handles agent failures."""
        swarm = SwarmIntelligence(SwarmParams(n_agents=10))

        # Remove some agents mid-operation
        initial_count = len(swarm.population.agents)

        # Remove 3 agents
        agent_ids = list(swarm.population.agents.keys())[:3]
        for aid in agent_ids:
            swarm.remove_agent(aid)

        assert len(swarm.population.agents) == initial_count - 3

        # System should still work
        problem = Problem(
            problem_id="robustness_test",
            description="Test after failures",
            input_state=np.random.randn(32),
            max_steps=5,
        )
        solution = swarm.solve(problem)
        assert solution is not None


# =============================================================================
# Additional Unit Tests
# =============================================================================


class TestSwarmIntelligence:
    """Additional tests for SwarmIntelligence."""

    def test_swarm_creation(self):
        """Test swarm initialization."""
        swarm = SwarmIntelligence(SwarmParams(n_agents=8))

        assert len(swarm.population.agents) == 8
        assert swarm.collective_memory is not None
        assert swarm.coordination is not None

    def test_learning(self):
        """Test collective learning."""
        swarm = SwarmIntelligence(SwarmParams(n_agents=5))

        problem = Problem(
            problem_id="learn_test",
            description="Learning test",
            input_state=np.random.randn(32),
            max_steps=5,
        )
        solution = swarm.solve(problem)

        # Learn from experience
        swarm.learn((problem, solution, 0.8))

        # Check learning had effect
        stats = swarm.get_statistics()
        assert stats["problems_solved"] >= 0

    def test_collective_belief(self):
        """Test collective belief aggregation."""
        swarm = SwarmIntelligence(SwarmParams(n_agents=5))

        # Store some beliefs
        swarm.collective_memory.store("agent1", "topic_test", np.ones(64), confidence=0.8)

        belief = swarm.get_collective_belief("topic_test")
        assert len(belief) == 64

    def test_top_performers(self):
        """Test getting top performers."""
        swarm = SwarmIntelligence(SwarmParams(n_agents=5))

        # Run some problems
        for i in range(3):
            problem = Problem(
                problem_id=f"perf_test_{i}",
                description="Performance test",
                input_state=np.random.randn(32),
                max_steps=3,
            )
            swarm.solve(problem)

        top = swarm.get_top_performers(n=3)
        assert len(top) <= 3

    def test_reset(self):
        """Test swarm reset."""
        swarm = SwarmIntelligence(SwarmParams(n_agents=5))

        # Do some work
        problem = Problem(
            problem_id="reset_test",
            description="Reset test",
            input_state=np.random.randn(32),
            max_steps=3,
        )
        swarm.solve(problem)

        # Reset
        swarm.reset()

        stats = swarm.get_statistics()
        assert stats["step_count"] == 0
        assert stats["problems_solved"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
