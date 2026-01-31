# Module 11: Time Perception

Neural basis of time perception grounded in embodied experience, based on 2025 neuroscience research.

## Key Insight (2025)

Time perception relies on brain regions supporting body-environment interaction (SMA) and internal body signals (insula). **Time is not perceived abstractly - it's grounded in embodied experience.**

## Core Components

### 1. Time Circuits (`src/time_circuits.py`)
Brain regions for time perception:

```python
from src import TimeCircuit, Insula, BasalGanglia, Cerebellum

circuit = TimeCircuit()

# Estimate duration using all circuits
signal = circuit.estimate_duration(
    actual_duration=10.0,
    arousal=0.5,
    attention=0.7,
    movement=True
)

print(f"Perceived: {signal.duration_estimate:.2f}s")
print(f"Confidence: {signal.confidence:.2f}")
```

**Neural Structures:**
| Structure | Function |
|-----------|----------|
| Insula | Primary interoceptive cortex; processes body signals |
| SMA | Body-environment interaction; motor timing |
| Basal ganglia | Interval timing; temporal working memory |
| Cerebellum | Precise timing (<1s); motor coordination |

### 2. Interval Timing (`src/interval_timing.py`)
Computational models of timing:

```python
from src import IntervalTimer, TimingMode

timer = IntervalTimer()

# Estimate a duration
result = timer.estimate_duration(10.0, attention=0.7)
print(f"Estimated: {result.estimated_duration:.2f}s")

# Produce a target duration
produced = timer.produce_duration(5.0)
print(f"Produced: {produced.estimated_duration:.2f}s")
```

**Models:**
- Pacemaker-Accumulator: Classic clock with pulse counting
- Striatal Beat Frequency: Oscillator-based timing

### 3. Time Modulation (`src/time_modulation.py`)
Factors affecting time perception:

```python
from src import TimeModulationSystem, EmotionalState

modulator = TimeModulationSystem()

perceived, effects = modulator.modulate_duration(
    actual_duration=10.0,
    emotional_state=EmotionalState.FEAR,
    attention=0.8,
    dopamine_level=1.2,
    age=30
)

print(f"Perceived: {perceived:.2f}s (actual: 10s)")
```

**Modulation Factors:**
| Factor | Effect on Perceived Time |
|--------|--------------------------|
| Emotion | High arousal → time slows |
| Attention | More attention → longer perceived duration |
| Dopamine | Dopamine increase → time speeds up |
| Age | Aging → subjective time acceleration |

### 4. Embodied Time (`src/embodied_time.py`)
Time grounded in body experience:

```python
from src import EmbodiedTimeSystem, BodyState

system = EmbodiedTimeSystem()

# Set body state
system.set_body_state(BodyState(
    heart_rate=80,
    breathing_rate=18
))

# Estimate duration using body signals
estimate, components = system.estimate_duration(
    actual_duration=30.0,
    movement_present=True,
    external_rhythm=1.0  # 1 Hz beat
)

print(f"Estimate: {estimate:.2f}s")
print(f"Entrainment: {components.get('entrainment', 0):.2f}")
```

### 5. Temporal Integration (`src/temporal_integration.py`)
Complete time perception system:

```python
from src import TimePerceptionOrchestrator

orchestrator = TimePerceptionOrchestrator()

# Use preset scenarios
baseline = orchestrator.estimate(10.0, scenario='baseline')
fear = orchestrator.estimate(10.0, scenario='fear')
flow = orchestrator.estimate(10.0, scenario='flow')

print(f"Baseline: {baseline.perceived_duration:.2f}s")
print(f"Fear (time slows): {fear.perceived_duration:.2f}s")
print(f"Flow (time flies): {flow.perceived_duration:.2f}s")
```

**Available Scenarios:**
- `baseline` - Normal adult, neutral state
- `fear` - Threatening situation (time slows)
- `boredom` - Waiting, nothing to do (time drags)
- `flow` - Absorbed in engaging task (time flies)
- `elderly` - 70 year old (time accelerates with age)
- `child` - 8 year old (time feels slower)
- `stimulant` - Under stimulant effects (faster clock)
- `parkinsons` - Low dopamine (slower clock)

## Installation

```bash
cd neuro-module-11-time-perception
pip install -r requirements.txt
```

## Running Tests

```bash
pytest tests/ -v
```

## Examples

### Time Distortion Demo
```bash
python examples/time_distortion_demo.py
```
Demonstrates how emotions, attention, dopamine, and age affect time perception.

### Embodied Time Demo
```bash
python examples/embodied_time_demo.py
```
Shows time perception grounded in body experience (heartbeat counting, motor timing, entrainment).

## Research Findings (2025)

| Finding | Implementation |
|---------|----------------|
| Insula processes body signals for timing | `InteroceptiveTimer` uses heartbeat/breathing |
| SMA handles body-environment interaction | `MotorTimer` and `BodyEnvironmentCoupler` |
| Basal ganglia for interval timing | `BasalGanglia` with dopamine modulation |
| Cerebellum for sub-second precision | `Cerebellum` with degradation for longer durations |
| High arousal slows time | `EmotionalModulator` with arousal-duration mapping |
| Attention lengthens perceived duration | `AttentionalModulator` with allocation model |
| Dopamine speeds internal clock | `DopamineModulator` affects clock rate |
| Age accelerates subjective time | `AgeModulator` with proportional theory |

## Weber's Law in Timing

Timing variability scales proportionally with duration (scalar property):
- Coefficient of variation remains constant across durations
- Implemented in `PacemakerAccumulator` with duration-scaled noise
