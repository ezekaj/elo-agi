"""
Self-Improvement Module: Recursive self-modification capabilities.

Implements the Darwin Gödel Machine concept for autonomous self-improvement.
"""

from .generator import ModificationGenerator, GeneratorParams, Modification
from .verifier import ChangeVerifier, VerifierParams, VerificationResult
from .updater import SystemUpdater, UpdaterParams, UpdateResult
from .meta_learner import MetaLearner, MetaParams, LearningStrategy
from .darwin_godel import DarwinGodelMachine, DGMParams, ImprovementCycle

__all__ = [
    "ModificationGenerator",
    "GeneratorParams",
    "Modification",
    "ChangeVerifier",
    "VerifierParams",
    "VerificationResult",
    "SystemUpdater",
    "UpdaterParams",
    "UpdateResult",
    "MetaLearner",
    "MetaParams",
    "LearningStrategy",
    "DarwinGodelMachine",
    "DGMParams",
    "ImprovementCycle",
]
