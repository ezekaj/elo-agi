"""
neuro-ground: Real-world sensor and actuator interface.

Provides connection to physical sensors and actuators for
grounding the cognitive architecture in the real world.
"""

from .sensors.camera import (
    Camera,
    CameraConfig,
    CameraFrame,
    VisionProcessor,
)
from .sensors.microphone import (
    Microphone,
    MicrophoneConfig,
    AudioBuffer,
    AudioProcessor,
)
from .sensors.proprioception import (
    ProprioceptionSensor,
    JointState,
    BodyState,
    ProprioceptionProcessor,
)

from .actuators.motor_controller import (
    MotorController,
    MotorConfig,
    MotorCommand,
    TrajectoryPlanner,
)
from .actuators.speech_synth import (
    SpeechSynthesizer,
    SpeechConfig,
    Utterance,
    ProsodyController,
)

from .sim2real import (
    DomainRandomization,
    SimToRealTransfer,
    RealityGap,
)
from .calibration import (
    SensorCalibrator,
    CalibrationResult,
    CalibrationConfig,
)

__version__ = "0.1.0"

__all__ = [
    # Sensors
    "Camera",
    "CameraConfig",
    "CameraFrame",
    "VisionProcessor",
    "Microphone",
    "MicrophoneConfig",
    "AudioBuffer",
    "AudioProcessor",
    "ProprioceptionSensor",
    "JointState",
    "BodyState",
    "ProprioceptionProcessor",
    # Actuators
    "MotorController",
    "MotorConfig",
    "MotorCommand",
    "TrajectoryPlanner",
    "SpeechSynthesizer",
    "SpeechConfig",
    "Utterance",
    "ProsodyController",
    # Sim2Real
    "DomainRandomization",
    "SimToRealTransfer",
    "RealityGap",
    # Calibration
    "SensorCalibrator",
    "CalibrationResult",
    "CalibrationConfig",
]
