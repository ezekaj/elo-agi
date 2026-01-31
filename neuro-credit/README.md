# neuro-credit

Unified credit assignment for AGI systems.

## Overview

neuro-credit provides temporal credit assignment capabilities including:

- **Eligibility Traces**: TD(lambda) style traces for temporal credit
- **Policy Gradients**: GAE and cross-module gradient computation
- **Blame Assignment**: Failure attribution with counterfactual analysis
- **Surprise Modulation**: Adaptive learning rates based on prediction errors
- **Contribution Accounting**: Shapley values for fair credit distribution

## Installation

```bash
cd neuro-credit
pip install -r requirements.txt
```

## Quick Start

```python
from neuro_credit import CreditAssignmentSystem, CreditConfig, FailureType

# Create credit assignment system
config = CreditConfig(
    gamma=0.99,
    trace_lambda=0.9,
    use_shapley=True,
)
system = CreditAssignmentSystem(config=config)

# Register modules
system.register_module("perception")
system.register_module("decision")
system.register_module("action")

# Record actions during execution
system.record_action("perception", state, "process_input")
system.record_action("decision", features, "select_action")
system.record_action("action", decision, "execute")

# Distribute credit when reward is received
credits = system.receive_reward(reward=1.0)
print(f"Credits: {credits}")

# Handle failures with blame assignment
blame = system.handle_failure(
    failure_type=FailureType.GOAL_UNREACHED,
    description="Failed to reach target",
    severity=0.8,
)

# Get adaptive learning rate based on surprise
lr = system.get_adaptive_learning_rate("decision", predicted, actual)
```

## Components

### EligibilityTraceManager

TD(lambda) eligibility traces for temporal credit:

```python
from neuro_credit import EligibilityTraceManager, TraceConfig, TraceType

config = TraceConfig(
    trace_type=TraceType.ACCUMULATING,
    lambda_param=0.9,
    gamma=0.99,
)
traces = EligibilityTraceManager(config=config)

# Mark state-action pairs as eligible
traces.mark_eligible(state, action, "module_id")

# Distribute credit when reward arrives
credits = traces.distribute_credit(reward=1.0)

# Decay traces over time
traces.decay_traces()
```

### CrossModulePolicyGradient

Policy gradient computation with GAE:

```python
from neuro_credit import CrossModulePolicyGradient, GAEConfig

config = GAEConfig(gamma=0.99, lambda_param=0.95)
pg = CrossModulePolicyGradient(config=config)

# Compute advantages
advantages, value_targets = pg.compute_advantage(
    rewards=trajectory_rewards,
    values=value_predictions,
)

# Compute policy loss
loss = pg.compute_policy_loss(log_probs, advantages)

# Process full trajectory
result = pg.compute_full_result(trajectory, "module_id")
```

### BlameAssignment

Failure attribution with counterfactual analysis:

```python
from neuro_credit import BlameAssignment, Failure, FailureType

blame = BlameAssignment()

# Set world model for counterfactual reasoning
blame.set_world_model(world_model_fn)

# Identify root cause of failure
root_module, confidence = blame.identify_root_cause(
    failure, action_trajectory, module_ids
)

# Get detailed blame for specific module
result = blame.compute_blame(failure, trajectory, "module_id")
print(f"Blame: {result.blame_score}, Evidence: {result.evidence}")
```

### SurpriseModulatedLearning

Adaptive learning rates based on surprise:

```python
from neuro_credit import SurpriseModulatedLearning, SurpriseConfig

config = SurpriseConfig(
    base_learning_rate=0.01,
    surprise_scale=1.0,
)
surprise = SurpriseModulatedLearning(config=config)

# Compute surprise and get modulated learning rate
metrics = surprise.compute_surprise(predicted, actual)
print(f"Surprise: {metrics.surprise_value}")
print(f"Modulated LR: {metrics.modulated_lr}")
print(f"Should consolidate: {metrics.should_consolidate}")
```

### ContributionAccountant

Shapley values for fair credit distribution:

```python
from neuro_credit import ContributionAccountant, ShapleyConfig

config = ShapleyConfig(use_approximation=True, num_samples=100)
accountant = ContributionAccountant(config=config)

# Compute Shapley values
def value_function(coalition):
    return compute_coalition_value(coalition)

shapley = accountant.compute_shapley_values(
    outcome=total_reward,
    active_modules=["m1", "m2", "m3"],
    value_function=value_function,
)

# Identify underperforming modules
underperforming = accountant.identify_underperforming_modules(threshold=0.0)
```

## Integration Points

- **neuro-module-12-learning**: Extends `RewardModulatedSTDP`
- **neuro-module-06-motivation**: `DopamineSystem` prediction errors
- **neuro-causal**: Counterfactual blame assignment

## Tests

```bash
python -m pytest tests/ -v
```

85 tests covering:
- Eligibility trace management
- Policy gradient computation
- Blame assignment accuracy
- Surprise modulation
- Shapley value properties
- End-to-end credit assignment

## License

MIT
