"""
Neuro Module 05: Sleep and Memory Consolidation

Implements sleep's role in memory processing:
- Sleep stage dynamics (NREM 1-2, SWS, REM) with distinct oscillatory signatures
- Active Systems Consolidation Model (hippocampal replay → cortical storage)
- Synaptic homeostasis (downscaling while preserving relative differences)
- Dream generation as consolidation by-product
"""

from .sleep_stages import (
    SleepStage,
    StageProperties,
    SleepStageController,
    OscillationGenerator,
)
from .memory_replay import (
    MemoryTrace,
    HippocampalReplay,
    ReplayPrioritizer,
)
from .systems_consolidation import (
    HippocampalStore,
    CorticalStore,
    HippocampalCorticalDialogue,
    ConsolidationWindow,
    MemoryTransformation,
)
from .synaptic_homeostasis import (
    SynapticHomeostasis,
    SelectiveConsolidation,
)
from .dream_generator import (
    DreamGenerator,
    NarrativeAssembler,
    DreamReport,
)
from .sleep_cycle import (
    SleepCycleOrchestrator,
    SleepArchitecture,
)
from .meta_learning import (
    MetaLearningController,
    LearningCurve,
    ReplayWeights,
    MemoryType,
)
from .spaced_repetition import (
    SpacedRepetitionScheduler,
    RepetitionSchedule,
    ReviewQuality,
)
from .interference_resolution import (
    InterferenceResolver,
    InterferenceEvent,
    ResolutionStrategy,
)
from .schema_refinement import (
    SchemaRefiner,
    Schema,
    SchemaUpdateType,
)

__version__ = "0.2.0"
__all__ = [
    # Sleep stages
    "SleepStage",
    "StageProperties",
    "SleepStageController",
    "OscillationGenerator",
    # Memory replay
    "MemoryTrace",
    "HippocampalReplay",
    "ReplayPrioritizer",
    # Systems consolidation
    "HippocampalStore",
    "CorticalStore",
    "HippocampalCorticalDialogue",
    "ConsolidationWindow",
    "MemoryTransformation",
    # Synaptic homeostasis
    "SynapticHomeostasis",
    "SelectiveConsolidation",
    # Dream generation
    "DreamGenerator",
    "NarrativeAssembler",
    "DreamReport",
    # Sleep cycle
    "SleepCycleOrchestrator",
    "SleepArchitecture",
    # Meta-learning (NEW)
    "MetaLearningController",
    "LearningCurve",
    "ReplayWeights",
    "MemoryType",
    # Spaced repetition (NEW)
    "SpacedRepetitionScheduler",
    "RepetitionSchedule",
    "ReviewQuality",
    # Interference resolution (NEW)
    "InterferenceResolver",
    "InterferenceEvent",
    "ResolutionStrategy",
    # Schema refinement (NEW)
    "SchemaRefiner",
    "Schema",
    "SchemaUpdateType",
]
