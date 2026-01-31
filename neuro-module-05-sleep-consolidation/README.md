# Module 5: Sleep and Memory Consolidation

Working implementation of sleep's role in memory processing based on 2025 neuroscience research.

## Core Components

### 1. Sleep Stages (`src/sleep_stages.py`)
Sleep stage controller with characteristic oscillations:

```python
from src import SleepStageController, SleepStage

controller = SleepStageController()
controller.start_sleep()  # Begin NREM1

# Advance through sleep
controller.advance_time(dt=5.0)  # 5 minutes

# Generate oscillations for current stage
events = controller.generate_oscillation(duration=1.0)  # 1 second
```

**Sleep Stages:**
- NREM1: Light sleep, theta waves (4-8 Hz)
- NREM2: Sleep spindles (12-16 Hz), K-complexes
- SWS: Slow waves (0.5-4 Hz), sharp-wave ripples (80-120 Hz)
- REM: Theta waves, dreaming

### 2. Memory Replay (`src/memory_replay.py`)
Hippocampal memory reactivation in compressed time (~20x faster):

```python
from src import HippocampalReplay, MemoryTrace

replay = HippocampalReplay(compression_factor=20.0)

# Encode experiences
trace = replay.encode_experience(
    pattern=np.random.randn(20),
    emotional_salience=0.7
)

# Select and replay memories
selected = replay.select_for_replay(n_select=3)
for memory in selected:
    replay.replay_memory(memory, ripple_present=True)
```

### 3. Systems Consolidation (`src/systems_consolidation.py`)
Hippocampal-cortical memory transfer:

```python
from src import HippocampalCorticalDialogue, ConsolidationWindow

dialogue = HippocampalCorticalDialogue(transfer_threshold=0.7)

# Encode in hippocampus
memory_id = dialogue.hippocampus.encode(trace)

# Consolidate during optimal window
dialogue.initiate_dialogue(slow_osc_phase="up", spindle=True, ripple=True)
dialogue.consolidate_memory(memory_id, replay_strength=1.0)

# Check transfer to cortex
dialogue.transfer_memory(memory_id)
```

### 4. Synaptic Homeostasis (`src/synaptic_homeostasis.py`)
Sleep-dependent synaptic downscaling:

```python
from src import SynapticHomeostasis, SelectiveConsolidation

homeostasis = SynapticHomeostasis(n_neurons=100, connectivity=0.1)

# Wake: learning increases weights
homeostasis.hebbian_potentiation(pre_activity, post_activity, learning_rate=0.1)

# Tag important synapses
selective = SelectiveConsolidation(homeostasis)
selective.tag_by_weight(percentile=90)

# Sleep: downscale untagged synapses
homeostasis.downscale(factor=0.8)
```

### 5. Dream Generator (`src/dream_generator.py`)
Dreams as by-product of consolidation:

```python
from src import DreamGenerator

dream_gen = DreamGenerator(pfc_suppression=0.6)

# Generate dream from replaying memories
dream = dream_gen.generate_dream(replaying_memories, duration=10.0)

print(f"Bizarreness: {dream.bizarreness_index}")
print(f"Emotional tone: {dream.emotional_tone}")
```

### 6. Sleep Cycle Orchestrator (`src/sleep_cycle.py`)
Complete sleep-wake cycle management:

```python
from src import SleepCycleOrchestrator

orchestrator = SleepCycleOrchestrator(n_neurons=100)

# Encode experiences during wake
memories = orchestrator.wake_encoding(experiences, emotional_saliences)

# Run overnight consolidation
night_stats = orchestrator.sleep_consolidation(sleep_hours=8.0)

print(f"Memories consolidated: {night_stats.total_memories_consolidated}")
print(f"Dreams generated: {night_stats.total_dreams}")
```

## Installation

```bash
cd neuro-module-05-sleep-consolidation
pip install -r requirements.txt
```

## Running Tests

```bash
pytest tests/ -v
```

## Examples

### Overnight Consolidation Demo
```bash
python examples/overnight_consolidation.py
```
Demonstrates memory improvement after full night of sleep.

### Sleep Deprivation Demo
```bash
python examples/sleep_deprivation.py
```
Shows effects of missing specific sleep stages.

## Key Research Findings (2025)

| Finding | Implementation |
|---------|----------------|
| SWS AND REM both contribute to emotional memory | Both stages process emotional traces |
| REM/SWS ratio predicts memory transformation | Higher ratio → more abstraction |
| TMR during REM impairs memory | Reactivation timing matters |
| Theta/beta during REM for schema formation | Oscillation-function coupling |
| Dreams = consolidation by-product | Dream content tracks replay |

## Sleep Architecture

- **Cycle duration**: ~90 minutes
- **SWS**: Dominates early night, decreases across cycles
- **REM**: Increases across the night
- **Sharp-wave ripples**: Coordinate memory replay during SWS
- **Sleep spindles**: Mark memory consolidation in NREM2

## Synaptic Homeostasis Hypothesis

- **Wake**: Net synaptic potentiation (learning increases weights)
- **Sleep**: Global downscaling (restores baseline)
- **Result**: Signal-to-noise ratio maintained
- **Protection**: Important synapses tagged to resist downscaling
