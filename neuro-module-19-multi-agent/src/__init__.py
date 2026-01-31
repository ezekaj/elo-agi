"""Multi-Agent Coordination module for collective intelligence."""

from .agent import CognitiveAgent, AgentParams, AgentRole, BeliefState, Message
from .population import AgentPopulation, PopulationState, PopulationParams
from .coordination import EmergentCoordination, CoordinationMechanism, EmergentPattern
from .collective_memory import CollectiveMemory, MemoryEntry, MemoryParams
from .swarm import SwarmIntelligence, Problem, Solution, SwarmParams

__all__ = [
    'CognitiveAgent',
    'AgentParams',
    'AgentRole',
    'BeliefState',
    'Message',
    'AgentPopulation',
    'PopulationState',
    'PopulationParams',
    'EmergentCoordination',
    'CoordinationMechanism',
    'EmergentPattern',
    'CollectiveMemory',
    'MemoryEntry',
    'MemoryParams',
    'SwarmIntelligence',
    'Problem',
    'Solution',
    'SwarmParams',
]
