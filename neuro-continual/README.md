# neuro-continual

Continual learning controller for AGI systems.

## Overview

neuro-continual provides lifelong learning capabilities including:

- **Task Inference**: Automatic task boundary detection and identification
- **Selective Consolidation**: Performance-based memory consolidation
- **Forgetting Prevention**: EWC, PackNet, and Synaptic Intelligence
- **Experience Replay**: Importance-weighted replay with task balancing
- **Capability Tracking**: Monitor skill regression and interference

## Installation

```bash
cd neuro-continual
pip install -r requirements.txt
```

## Quick Start

```python
from neuro_continual import ContinualLearningController, ContinualLearningConfig

# Create controller
config = ContinualLearningConfig(
    auto_detect_tasks=True,
    forgetting_method=ForgettingPreventionMethod.EWC,
    ewc_lambda=1000.0,
)
controller = ContinualLearningController(config=config)

# Process observations during learning
result = controller.observe(
    state=current_state,
    action=action,
    reward=reward,
    next_state=next_state,
    done=done,
    params=model_parameters,  # For forgetting prevention
)

if result["task_changed"]:
    print(f"Task changed: {result['previous_task']} -> {result['task_id']}")

# Sample replay batch for training
batch = controller.sample_replay(batch_size=32)
for experience, importance_weight in batch:
    # Train with importance weighting
    pass

# Compute forgetting prevention loss
total_loss, losses = controller.compute_forgetting_loss(current_params)
```

## Components

### TaskInference

Automatic task boundary detection:

```python
from neuro_continual import TaskInference, TaskInferenceConfig

config = TaskInferenceConfig(
    change_threshold=0.5,
    embedding_dim=64,
)
inference = TaskInference(config=config)

# Detect task changes
if inference.detect_task_change(current_state):
    task_id = inference.infer_task_id(current_state)
    print(f"New task detected: {task_id}")

# Merge similar tasks
merged = inference.merge_similar_tasks(threshold=0.8)
```

### CatastrophicForgettingPrevention

Multiple forgetting prevention strategies:

```python
from neuro_continual import (
    CatastrophicForgettingPrevention,
    ForgettingPreventionConfig,
    ForgettingPreventionMethod,
)

config = ForgettingPreventionConfig(
    method=ForgettingPreventionMethod.EWC,
    ewc_lambda=1000.0,
)
prevention = CatastrophicForgettingPrevention(config=config)

# Register task parameters
prevention.register_task("task1", model_params, fisher_info)

# Compute EWC loss during training on new task
ewc_loss = prevention.compute_ewc_loss(current_params)
```

### ImportanceWeightedReplay

Prioritized experience replay:

```python
from neuro_continual import ImportanceWeightedReplay, ReplayConfig, ReplayStrategy

config = ReplayConfig(
    buffer_size=10000,
    strategy=ReplayStrategy.PRIORITIZED,
)
replay = ImportanceWeightedReplay(config=config)

# Add experiences
replay.add(state, action, reward, next_state, done, task_id, td_error)

# Sample batch with importance weights
batch = replay.sample_batch(batch_size=32, task_balance={"task1": 0.5, "task2": 0.5})

# Update priorities after learning
replay.update_priorities(indices, new_td_errors)
```

### SelectiveConsolidation

Performance-based consolidation:

```python
from neuro_continual import SelectiveConsolidation, ConsolidationConfig

config = ConsolidationConfig(
    strategy=ConsolidationStrategy.PERFORMANCE_WEIGHTED,
    min_gap_for_consolidation=0.1,
)
consolidation = SelectiveConsolidation(config=config)

# Track performance
consolidation.update_performance("task1", current_performance)

# Create consolidation plan
plan = consolidation.create_consolidation_plan(total_budget=1000)
for task_id, priority in plan.task_priorities:
    budget = plan.budget_allocation.get(task_id, 0)
    # Allocate replay budget to task
```

### CapabilityTracker

Monitor skill regression:

```python
from neuro_continual import CapabilityTracker, CapabilityConfig

config = CapabilityConfig(
    regression_threshold=0.1,
    interference_threshold=0.3,
)
tracker = CapabilityTracker(config=config)

# Measure capabilities
metric = tracker.measure_capability("reasoning", test_results)

# Check for regression
if tracker.detect_regression("reasoning"):
    suggestions = tracker.suggest_remediation(["reasoning"])

# Identify interference
reports = tracker.identify_interference([("cap1", "cap2")])
```

## Integration Points

- **neuro-module-05-sleep-consolidation**: Extends MetaLearningController
- **neuro-module-12-learning**: Apply EWC to HebbianLearning
- **neuro-bench**: Capability measurement

## Tests

```bash
python -m pytest tests/ -v
```

88 tests covering:
- Task inference and boundary detection
- Selective consolidation strategies
- Forgetting prevention (EWC, PackNet, SI)
- Experience replay mechanisms
- Capability tracking and regression detection
- End-to-end continual learning

## License

MIT
