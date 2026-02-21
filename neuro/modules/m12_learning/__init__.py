"""
Module 12: Learning and Neuroplasticity

Implements synaptic plasticity rules based on 2025 research:
- Hebbian Learning: "Neurons that fire together, wire together"
- Spike-Timing Dependent Plasticity (STDP)
- Reward-modulated learning (dopamine)
- Homeostatic regulation
- Structural plasticity
"""

from .hebbian import HebbianLearning, HebbianNetwork
from .stdp import STDPRule, STDPSynapse, STDPNetwork
from .reward_modulated import RewardModulatedSTDP, DopamineSystem
from .structural_plasticity import StructuralPlasticity, SynapticPruning, DendriticGrowth
from .homeostatic import HomeostaticRegulation, SynapticScaling

__all__ = [
    'HebbianLearning', 'HebbianNetwork',
    'STDPRule', 'STDPSynapse', 'STDPNetwork',
    'RewardModulatedSTDP', 'DopamineSystem',
    'StructuralPlasticity', 'SynapticPruning', 'DendriticGrowth',
    'HomeostaticRegulation', 'SynapticScaling'
]
