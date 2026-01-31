"""
neuro-robust: Robustness and Reliability for AGI Systems.

Provides:
- Adversarial training and defense
- Uncertainty quantification (epistemic vs aleatoric)
- Out-of-distribution (OOD) detection
- Calibrated confidence estimation
- Robust inference under uncertainty
"""

from .adversarial import (
    AdversarialAttack,
    AdversarialDefense,
    AttackType,
    DefenseType,
)
from .uncertainty import (
    UncertaintyQuantifier,
    UncertaintyType,
    EnsembleUncertainty,
)
from .ood_detection import (
    OODDetector,
    OODMethod,
    OODResult,
)
from .calibration import (
    ConfidenceCalibrator,
    CalibrationMethod,
    CalibrationMetrics,
)
from .robust_inference import (
    RobustInference,
    RobustPrediction,
    SelectivePrediction,
)
from .integration import (
    SharedSpaceRobustness,
    RobustEmbedding,
    RobustnessMetadata,
    RobustnessLevel,
)

__all__ = [
    # Adversarial
    "AdversarialAttack",
    "AdversarialDefense",
    "AttackType",
    "DefenseType",
    # Uncertainty
    "UncertaintyQuantifier",
    "UncertaintyType",
    "EnsembleUncertainty",
    # OOD Detection
    "OODDetector",
    "OODMethod",
    "OODResult",
    # Calibration
    "ConfidenceCalibrator",
    "CalibrationMethod",
    "CalibrationMetrics",
    # Robust Inference
    "RobustInference",
    "RobustPrediction",
    "SelectivePrediction",
    # Integration
    "SharedSpaceRobustness",
    "RobustEmbedding",
    "RobustnessMetadata",
    "RobustnessLevel",
]
