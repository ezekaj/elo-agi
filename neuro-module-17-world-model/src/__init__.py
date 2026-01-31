"""World Model Module: Unified predictive model of reality."""

from .state_encoder import StateEncoder, EncoderParams, EncodedState
from .transition_model import TransitionModel, TransitionParams, Transition
from .imagination import Imagination, ImaginationParams, Trajectory, Rollout
from .counterfactual import CounterfactualEngine, CounterfactualParams, Counterfactual
from .world_memory import WorldMemory, MemoryParams, WorldState

__all__ = [
    'StateEncoder', 'EncoderParams', 'EncodedState',
    'TransitionModel', 'TransitionParams', 'Transition',
    'Imagination', 'ImaginationParams', 'Trajectory', 'Rollout',
    'CounterfactualEngine', 'CounterfactualParams', 'Counterfactual',
    'WorldMemory', 'MemoryParams', 'WorldState',
]
