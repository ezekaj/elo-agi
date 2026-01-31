# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
# Run all tests (84 tests)
python -m pytest tests/test_ground.py -v

# Run specific test class
python -m pytest tests/test_ground.py::TestCamera -v

# Run single test
python -m pytest tests/test_ground.py::TestCamera::test_camera_capture -v
```

**Note:** Tests require numpy and scipy. Create a venv if needed:
```bash
python -m venv .venv && source .venv/bin/activate
pip install pytest numpy scipy
```

## Architecture

neuro-ground provides real-world sensor/actuator interfaces for the Neuro cognitive architecture, operating in simulation mode by default.

### Component Pairs
Each domain has a sensor/actuator class paired with a processor class:

| Domain | Interface | Processor |
|--------|-----------|-----------|
| Vision | `Camera` | `VisionProcessor` |
| Audio | `Microphone` | `AudioProcessor` |
| Body | `ProprioceptionSensor` | `ProprioceptionProcessor` |
| Motor | `MotorController` | `TrajectoryPlanner` |
| Speech | `SpeechSynthesizer` | `ProsodyController` |

### Import Pattern
Tests use path manipulation to import from `src/`:
```python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from sensors.camera import Camera, VisionProcessor
```

### Sim2Real Bridge
- `DomainRandomization` - Visual/dynamics/sensor noise randomization
- `RealityGap` - Measure sim-to-real distribution distance (MMD)
- `SimToRealTransfer` - Complete transfer pipeline

### Calibration
`SensorCalibrator` handles IMU, joint encoder, force sensor, and camera calibration with `apply_calibration()` for runtime correction.

## Conventions

- All sensors have `_simulated = True` by default (no real hardware required)
- Config dataclasses use sensible defaults (e.g., `CameraConfig(width=640, height=480)`)
- All major classes expose `.statistics()` for introspection
- Processors are stateless where possible; sensors manage state
