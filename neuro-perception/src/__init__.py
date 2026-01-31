"""
neuro-perception: Real sensory processing for the cognitive architecture.

Provides visual and auditory processing pipelines with multimodal integration.
"""

# Visual processing
from visual.retina import (
    Retina,
    RetinaOutput,
    PhotoreceptorResponse,
    PhotoreceptorType,
    GanglionCellType,
)
from visual.v1_v2 import (
    V1Processor,
    V2Processor,
    V1Output,
    V2Output,
    OrientationColumn,
    SpatialFrequencyFilter,
)
from visual.v4_it import (
    V4Processor,
    ITProcessor,
    V4Output,
    ITOutput,
    ShapeDescriptor,
    ObjectRepresentation,
)
from visual.dorsal_ventral import (
    DorsalStream,
    VentralStream,
    VisualPathways,
    DorsalOutput,
    VentralOutput,
    MotionVector,
    SpatialInfo,
)

# Auditory processing
from auditory.cochlea import (
    Cochlea,
    CochleaOutput,
    GammatoneFilterbank,
    HairCellResponse,
)
from auditory.a1 import (
    A1Processor,
    A1Output,
    SpectrotemporalRF,
)
from auditory.speech import (
    SpeechProcessor,
    SpeechOutput,
    PhonemeRecognizer,
    PhonemeDetection,
    PhonemeFeatures,
    PhonemeCategory,
    FormantTracker,
    VoicingDetector,
)

# Multimodal integration
from multimodal.binding import (
    CrossModalBinder,
    BindingOutput,
    BoundPercept,
    ModalityInput,
    Modality,
    TemporalBinder,
    SpatialBinder,
    FeatureBinder,
)
from multimodal.attention import (
    SelectiveAttention,
    AttentionOutput,
    AttentionFocus,
    SaliencyMap,
    AttentionType,
    SaliencyComputer,
    TopDownController,
    AttentionGate,
)

# Interface
from interface import (
    PerceptionSystem,
    VisualPipeline,
    AuditoryPipeline,
    VisualPercept,
    AuditoryPercept,
    MultimodalPercept,
    create_perception_system,
)

__version__ = "0.1.0"

__all__ = [
    # Visual
    "Retina",
    "RetinaOutput",
    "PhotoreceptorResponse",
    "PhotoreceptorType",
    "GanglionCellType",
    "V1Processor",
    "V2Processor",
    "V1Output",
    "V2Output",
    "OrientationColumn",
    "SpatialFrequencyFilter",
    "V4Processor",
    "ITProcessor",
    "V4Output",
    "ITOutput",
    "ShapeDescriptor",
    "ObjectRepresentation",
    "DorsalStream",
    "VentralStream",
    "VisualPathways",
    "DorsalOutput",
    "VentralOutput",
    "MotionVector",
    "SpatialInfo",
    # Auditory
    "Cochlea",
    "CochleaOutput",
    "GammatoneFilterbank",
    "HairCellResponse",
    "A1Processor",
    "A1Output",
    "SpectrotemporalRF",
    "SpeechProcessor",
    "SpeechOutput",
    "PhonemeRecognizer",
    "PhonemeDetection",
    "PhonemeFeatures",
    "PhonemeCategory",
    "FormantTracker",
    "VoicingDetector",
    # Multimodal
    "CrossModalBinder",
    "BindingOutput",
    "BoundPercept",
    "ModalityInput",
    "Modality",
    "TemporalBinder",
    "SpatialBinder",
    "FeatureBinder",
    "SelectiveAttention",
    "AttentionOutput",
    "AttentionFocus",
    "SaliencyMap",
    "AttentionType",
    "SaliencyComputer",
    "TopDownController",
    "AttentionGate",
    # Interface
    "PerceptionSystem",
    "VisualPipeline",
    "AuditoryPipeline",
    "VisualPercept",
    "AuditoryPercept",
    "MultimodalPercept",
    "create_perception_system",
]
