"""
Module 11: Time Perception

Neural basis of time perception grounded in embodied experience.
Based on 2025 neuroscience research.
"""

from .time_circuits import (
    Insula,
    SMA,
    BasalGanglia,
    Cerebellum,
    TimeCircuit
)

from .interval_timing import (
    PacemakerAccumulator,
    StriatalBeatFrequency,
    IntervalTimer,
    TimingMode
)

from .time_modulation import (
    EmotionalModulator,
    AttentionalModulator,
    DopamineModulator,
    AgeModulator,
    TimeModulationSystem
)

from .embodied_time import (
    InteroceptiveTimer,
    MotorTimer,
    BodyEnvironmentCoupler,
    EmbodiedTimeSystem
)

from .temporal_integration import (
    TemporalEstimate,
    SubjectiveTimeSystem,
    TimePerceptionOrchestrator
)

__all__ = [
    # Time circuits
    'Insula',
    'SMA',
    'BasalGanglia',
    'Cerebellum',
    'TimeCircuit',
    # Interval timing
    'PacemakerAccumulator',
    'StriatalBeatFrequency',
    'IntervalTimer',
    'TimingMode',
    # Modulation
    'EmotionalModulator',
    'AttentionalModulator',
    'DopamineModulator',
    'AgeModulator',
    'TimeModulationSystem',
    # Embodied time
    'InteroceptiveTimer',
    'MotorTimer',
    'BodyEnvironmentCoupler',
    'EmbodiedTimeSystem',
    # Integration
    'TemporalEstimate',
    'SubjectiveTimeSystem',
    'TimePerceptionOrchestrator',
]
