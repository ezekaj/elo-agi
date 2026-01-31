"""
neuro-transfer: Cross-domain knowledge transfer.

Implements transfer learning, meta-learning, and skill reuse
for generalization across domains and tasks.
"""

from abstraction import (
    AbstractionEngine,
    AbstractionLevel,
    AbstractConcept,
    StructuralAnalogy,
    DomainPrinciple,
)
from curriculum import (
    CurriculumLearner,
    Task,
    TaskDifficulty,
    LearningPath,
    ProgressTracker,
)
from few_shot import (
    FewShotLearner,
    PrototypeNetwork,
    MatchingNetwork,
    SupportSet,
    QueryResult,
)
from meta_learner import (
    MetaLearner,
    MAML,
    LearningStrategy,
    TaskDistribution,
    AdaptationResult,
)
from domain_adapter import (
    DomainAdapter,
    DomainEmbedding,
    DomainAlignment,
    TransferMapping,
    AdaptedRepresentation,
)
from skill_library import (
    SkillLibrary,
    Skill,
    SkillPrimitive,
    CompositeSkill,
    SkillExecution,
)

__version__ = "0.1.0"

__all__ = [
    # Abstraction
    "AbstractionEngine",
    "AbstractionLevel",
    "AbstractConcept",
    "StructuralAnalogy",
    "DomainPrinciple",
    # Curriculum
    "CurriculumLearner",
    "Task",
    "TaskDifficulty",
    "LearningPath",
    "ProgressTracker",
    # Few-shot
    "FewShotLearner",
    "PrototypeNetwork",
    "MatchingNetwork",
    "SupportSet",
    "QueryResult",
    # Meta-learning
    "MetaLearner",
    "MAML",
    "LearningStrategy",
    "TaskDistribution",
    "AdaptationResult",
    # Domain adaptation
    "DomainAdapter",
    "DomainEmbedding",
    "DomainAlignment",
    "TransferMapping",
    "AdaptedRepresentation",
    # Skill library
    "SkillLibrary",
    "Skill",
    "SkillPrimitive",
    "CompositeSkill",
    "SkillExecution",
]
