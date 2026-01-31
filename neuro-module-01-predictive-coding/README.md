# Module 1: Mathematical Foundations of Thought

Working implementation of the brain's core prediction machinery based on the Free Energy Principle and predictive coding theory.

## Core Components

### 1. Predictive Hierarchy (`src/predictive_hierarchy.py`)
Hierarchical predictive coding engine implementing the Free Energy equations:
- `y = g(x,v,θ) + z` (observations from hidden states)
- `ẋ = f(x,v,θ) + w` (hidden state dynamics)

```python
from src import PredictiveHierarchy

hierarchy = PredictiveHierarchy(
    layer_dims=[10, 8, 6, 4],  # Input -> deep layers
    learning_rate=0.1,
    timescale_factor=2.0
)

result = hierarchy.step(observation, dt=0.1)
prediction = hierarchy.predict_next()
```

### 2. Precision Weighting (`src/precision_weighting.py`)
Confidence-based error weighting: `ξ = Π * ε`

```python
from src import AdaptivePrecision

precision = AdaptivePrecision(dim=5, learning_rate=0.1)
prec, volatility = precision.update(error)
weighted_error = precision.get_weighted_error(error)
```

### 3. Cognitive Manifold (`src/cognitive_manifold.py`)
Geometric representation where thinking = gradient descent on cognitive potential.

```python
from src import CognitiveManifold, DualProcess

manifold = CognitiveManifold(
    dim=5,
    accuracy_weight=1.0,
    parsimony_weight=0.1,
    utility_weight=0.5
)

manifold.set_goal(target_state)
manifold.flow(dt=0.1)  # One "thought" step

# Dual-process cognition emerges from geometry
dp = DualProcess(manifold)
final_state, systems_used = dp.think(max_steps=100)
```

### 4. Temporal Dynamics (`src/temporal_dynamics.py`)
Multi-timescale processing with temporal hierarchy.

```python
from src import TemporalHierarchy

temporal = TemporalHierarchy(
    layer_dims=[10, 8, 6],
    base_timescale=0.01,  # 10ms at lowest layer
    timescale_factor=10.0  # Each layer 10x slower
)

states = temporal.step(input_signal, dt=0.01)
```

### 5. Omission Detector (`src/omission_detector.py`)
Detects when expected inputs fail to arrive.

```python
from src import OmissionDetector

detector = OmissionDetector(input_dim=5)
detector.add_expectation(expected_value, time_window=0.5)
omission = detector.check_omissions(current_time)
```

## Installation

```bash
cd neuro-module-01-predictive-coding
pip install -r requirements.txt
```

## Running Tests

```bash
pytest tests/ -v
```

## Examples

### Sensory Prediction Demo
Demonstrates sequence learning, omission detection, and lesion simulation:
```bash
python examples/sensory_prediction.py
```

### Belief Updating Demo
Demonstrates gradient flow, dual-process emergence, and attractor dynamics:
```bash
python examples/belief_updating.py
```

## Key Insights

### Dual-Process Cognition
System 1 (fast) and System 2 (slow) emerge from the SAME manifold geometry:
- **System 1**: Steep gradient regions → fast, automatic responses
- **System 2**: Shallow gradient regions → slow, deliberate processing
- No separate modules required - pure geometric emergence

### Precision Modulates Learning
High precision (reliable predictions) → errors weighted heavily
Low precision (uncertainty) → errors downweighted
Enables adaptive learning in volatile environments

### Temporal Hierarchy
- Lower layers: fast timescales (~10ms) for sensory processing
- Higher layers: slow timescales (seconds to minutes) for abstract concepts
- Information flows bidirectionally: predictions down, errors up

### Omission Responses
The system generates "vigorous prediction error" for ABSENT expected stimuli, matching empirical findings from neuroscience.

## References

Based on theoretical frameworks from:
- Friston, K. (2010). The free-energy principle: a unified brain theory?
- Clark, A. (2013). Whatever next? Predictive brains, situated agents, and the future of cognitive science.
- Rao, R. P., & Ballard, D. H. (1999). Predictive coding in the visual cortex.
