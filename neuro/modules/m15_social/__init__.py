"""Social Cognition Module

Based on Part XV of neuroscience research:
- Theory of Mind: Attributing mental states to self and others
- Mentalizing: Inferring beliefs, intentions, emotions
- Empathy: Experiencing others' affective states
- Neural Substrates: mPFC (self-referential), TPJ (perspective-taking), pSTS (intention), Insula (empathy)
"""

from .theory_of_mind import TheoryOfMind, MentalStateAttribution, BeliefTracker, ToMParams
from .mentalizing import MentalizingNetwork, IntentionInference, DesireModeling, MentalizingParams
from .empathy import EmpathySystem, AffectiveSharing, EmpathicConcern, EmpathyParams
from .perspective_taking import (
    PerspectiveTaking,
    SelfOtherDistinction,
    TPJNetwork,
    PerspectiveParams,
)
from .social_network import SocialCognitionNetwork, SocialBrain

__version__ = "0.1.0"
__all__ = [
    "TheoryOfMind",
    "MentalStateAttribution",
    "BeliefTracker",
    "ToMParams",
    "MentalizingNetwork",
    "IntentionInference",
    "DesireModeling",
    "MentalizingParams",
    "EmpathySystem",
    "AffectiveSharing",
    "EmpathicConcern",
    "EmpathyParams",
    "PerspectiveTaking",
    "SelfOtherDistinction",
    "TPJNetwork",
    "PerspectiveParams",
    "SocialCognitionNetwork",
    "SocialBrain",
]
