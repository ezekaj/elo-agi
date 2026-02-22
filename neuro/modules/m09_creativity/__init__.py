"""
Module 09: Creativity and Imagination

Implementation of the brain's creative systems based on neuroscience research:
- Default Mode Network (DMN): Idea generation, imagination, spontaneous thought
- Executive Control Network: Idea evaluation, refinement, critical assessment
- Salience Network: Switching between networks at transition moments
- Mental Imagery: Visual, auditory, motor, tactile simulation

Key insight: Creative thought requires DYNAMIC COOPERATION between networks,
not just DMN activity. Network reconfiguration is higher during creative ideation.
"""

from .networks import DefaultModeNetwork, ExecutiveControlNetwork, SalienceNetwork
from .imagery import VisualImagery, AuditoryImagery, MotorImagery, TactileImagery, ImagerySystem
from .creative_process import CreativeProcess, Idea, CreativeOutput

__all__ = [
    "DefaultModeNetwork",
    "ExecutiveControlNetwork",
    "SalienceNetwork",
    "VisualImagery",
    "AuditoryImagery",
    "MotorImagery",
    "TactileImagery",
    "ImagerySystem",
    "CreativeProcess",
    "Idea",
    "CreativeOutput",
]
