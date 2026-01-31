# neuro-meta-reasoning

Meta-reasoning orchestration for AGI systems.

## Overview

neuro-meta-reasoning provides meta-level reasoning control including:

- **Problem Classification**: Identify problem types and complexity
- **Style Selection**: Choose appropriate reasoning styles
- **Efficiency Monitoring**: Track cost/benefit and early termination
- **Dynamic Orchestration**: Coordinate reasoning across modules
- **Fallacy Detection**: Detect and correct reasoning errors

## Installation

```bash
cd neuro-meta-reasoning
pip install -r requirements.txt
```

## Quick Start

```python
from neuro_meta_reasoning import MetaReasoningController, MetaReasoningConfig
import numpy as np

# Create controller
config = MetaReasoningConfig(
    exploration_rate=0.1,
    enable_fallacy_detection=True,
)
controller = MetaReasoningController(config=config)

# Create reasoning session
problem_embedding = np.random.randn(128)
session = controller.create_session(problem_embedding)

print(f"Problem type: {session.problem_analysis.problem_type}")
print(f"Selected style: {session.style_selection.primary_style}")

# Execute reasoning
result = controller.execute_session(session.session_id, {"data": problem_data})
print(f"Success: {result.success}, Quality: {result.final_quality}")

# Record feedback for learning
controller.record_feedback(
    session.session_id,
    success=result.success,
    efficiency=0.8,
    quality=result.final_quality,
)
```

## Components

### ProblemClassifier

Classify problems by type and complexity:

```python
from neuro_meta_reasoning import ProblemClassifier, ProblemClassifierConfig

config = ProblemClassifierConfig(embedding_dim=128)
classifier = ProblemClassifier(config=config)

analysis = classifier.classify(problem_embedding, context)
print(f"Type: {analysis.problem_type}")
print(f"Difficulty: {analysis.difficulty}")
print(f"Estimated steps: {analysis.estimated_steps}")
```

### StyleSelector

Select appropriate reasoning styles:

```python
from neuro_meta_reasoning import StyleSelector, StyleSelectorConfig

config = StyleSelectorConfig(exploration_rate=0.1)
selector = StyleSelector(config=config)

selection = selector.select_style(analysis, constraints)
print(f"Primary: {selection.primary_style}")
print(f"Fitness: {selection.primary_fitness}")

# Record feedback for adaptive selection
selector.record_feedback(
    style=selection.primary_style,
    problem_type=analysis.problem_type,
    success=True,
    efficiency=0.8,
    quality=0.9,
)
```

### EfficiencyMonitor

Monitor reasoning efficiency:

```python
from neuro_meta_reasoning import EfficiencyMonitor, EfficiencyConfig, ReasoningStyle

config = EfficiencyConfig(time_limit_seconds=60.0, cost_limit=100.0)
monitor = EfficiencyMonitor(config=config)

monitor.start_monitoring("problem1", ReasoningStyle.DEDUCTIVE)
monitor.update_progress("problem1", progress=0.5, confidence=0.7, cost=5.0)

should_stop, reason = monitor.should_terminate_early("problem1")
if should_stop:
    print(f"Terminating: {reason}")

report = monitor.get_efficiency_report("problem1")
print(f"Efficiency: {report.efficiency_score}")
```

### DynamicOrchestrator

Orchestrate reasoning execution:

```python
from neuro_meta_reasoning import DynamicOrchestrator, OrchestratorConfig

config = OrchestratorConfig(
    max_steps=100,
    checkpoint_frequency=5,
    enable_dynamic_switching=True,
)
orchestrator = DynamicOrchestrator(config=config)

# Register reasoning modules
orchestrator.register_module("logical_reasoner", logical_fn)
orchestrator.register_module("heuristic_solver", heuristic_fn)

# Create and execute plan
plan = orchestrator.create_plan(analysis, style_selection)
result = orchestrator.execute_plan(plan, problem)
```

### FallacyDetector

Detect reasoning errors:

```python
from neuro_meta_reasoning import FallacyDetector, FallacyDetectorConfig, ReasoningStep

config = FallacyDetectorConfig(confirmation_bias_threshold=0.7)
detector = FallacyDetector(config=config)

# Detect fallacies in reasoning trace
fallacies = detector.detect_fallacies(reasoning_trace)

for fallacy in fallacies:
    print(f"Detected: {fallacy.fallacy_type} (confidence: {fallacy.confidence})")

# Get correction suggestions
corrections = detector.suggest_corrections(fallacies)
```

**Supported Fallacy Types:**
- Confirmation bias
- Anchoring
- Availability bias
- Base rate neglect
- Circular reasoning
- False dichotomy
- Hasty generalization
- Sunk cost fallacy
- Overfitting
- Premature termination

## Integration Points

- **neuro-module-03-reasoning-types**: Route problems to reasoning modules
- **neuro-module-02-dual-process**: System 1/2 in style selection
- **neuro-module-16-consciousness**: Metacognitive feedback
- **neuro-robust**: Uncertainty for confidence estimates

## Tests

```bash
python -m pytest tests/ -v
```

84 tests covering:
- Problem classification and analysis
- Style selection and adaptation
- Efficiency monitoring and termination
- Dynamic orchestration and switching
- Fallacy detection and correction
- End-to-end meta-reasoning

## License

MIT
