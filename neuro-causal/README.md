# neuro-causal

Differentiable causal reasoning for AGI systems.

## Overview

neuro-causal provides causal inference capabilities including:

- **Differentiable SCMs**: Gradient-based learning of causal mechanisms
- **Counterfactual Reasoning**: "What if?" queries with nested counterfactuals
- **Causal Discovery**: Learn causal structure from observational data
- **Active Inference Integration**: Causal reasoning for planning and decision-making

## Installation

```bash
cd neuro-causal
pip install -r requirements.txt
```

## Quick Start

```python
from src.differentiable_scm import DifferentiableSCM, CausalMechanism
from src.counterfactual import NestedCounterfactual
from src.causal_discovery import CausalDiscovery

# Create a causal model
scm = DifferentiableSCM()
scm.add_variable("X")
scm.add_variable("Y", parents=["X"])
scm.add_variable("Z", parents=["Y"])

# Intervention: do(X=2)
intervened = scm.intervene({"X": 2.0})
result = intervened.forward()

# Counterfactual query
cf = NestedCounterfactual(scm)
answer = cf.query(
    evidence={"X": 1.0, "Y": 2.5, "Z": 4.0},
    intervention={"X": 3.0},
    target="Z"
)

# Discover causal structure from data
discovery = CausalDiscovery()
dag = discovery.pc_algorithm(data, alpha=0.01)
```

## Components

### DifferentiableSCM

Neural network-based structural causal model with:
- Learnable causal mechanisms (f_i: PA_i -> V_i)
- Topological forward pass
- Intervention via mechanism surgery
- Gradient-based causal effect estimation

### NestedCounterfactual

Pearl's three-step counterfactual reasoning:
1. Abduction: Infer noise from evidence
2. Action: Apply intervention
3. Prediction: Compute counterfactual outcome

Supports nested queries: "What would Y have been if X=x, given that if X=x', Y would have been y'?"

### CausalDiscovery

Structure learning algorithms:
- PC algorithm for constraint-based discovery
- Edge orientation using v-structures
- Structure uncertainty quantification

### CausalActiveInference

Integration with free energy principle:
- Causal model as generative model
- Imagination-based planning
- Intervention selection for goal achievement

## API Reference

See [docs/api.md](docs/api.md) for detailed documentation.

## Theory

The module implements Pearl's causal hierarchy:
1. **Association** (seeing): P(Y|X)
2. **Intervention** (doing): P(Y|do(X))
3. **Counterfactuals** (imagining): P(Y_x|X=x', Y=y')

See [docs/theory.md](docs/theory.md) for theoretical background.

## Tests

```bash
python -m pytest tests/ -v
```

124 tests covering:
- Differentiable SCM operations
- Counterfactual computation
- Causal discovery algorithms
- Integration with other modules

## License

MIT
