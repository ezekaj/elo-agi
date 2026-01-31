# neuro-abstract

Neuro-symbolic abstraction and compositional generalization.

## Overview

neuro-abstract provides compositional abstraction capabilities including:

- **Symbol Binding**: Neural-symbolic binding with role-filler structure
- **Composition Types**: Type-safe compositional operations
- **Program Synthesis**: Generate programs from input-output examples
- **Analogy Engine**: Structure-mapping for analogical reasoning

## Installation

```bash
cd neuro-abstract
pip install -r requirements.txt
```

## Quick Start

```python
from src.symbolic_binder import SymbolicBinder, CompositeBinding
from src.program_synthesis import ProgramSynthesizer
from src.abstraction_engine import AbstractionEngine

# Symbol binding with roles
binder = SymbolicBinder(embedding_dim=256)

# Bind "dog" to neural representation with "agent" role
binding = binder.bind(
    symbol="dog",
    neural_rep=dog_embedding,
    roles={"agent": agent_vector}
)

# Compose bindings into structured representation
chase_event = binder.compose([
    ("chase", verb_embedding, {"predicate": ...}),
    ("dog", dog_embedding, {"agent": ...}),
    ("cat", cat_embedding, {"patient": ...})
])

# Retrieve by role
agent = binder.retrieve_by_role("chase_event", "agent")

# Program synthesis from examples
synth = ProgramSynthesizer()
synth.register_primitive("double", lambda x: x * 2)
synth.register_primitive("add_one", lambda x: x + 1)

examples = [(1, 3), (2, 5), (3, 7)]  # f(x) = 2x + 1
program = synth.synthesize(examples)
```

## Components

### SymbolicBinder

Tensor product variable binding:
- Bind symbols to continuous representations
- Role-filler decomposition (agent, patient, theme, etc.)
- Compositional structure via tensor products
- Unbinding and retrieval operations

### CompositionType

Type-safe compositional semantics:
- Atomic types (int, float, string)
- Function types (A -> B)
- Structured types ({field: Type})
- Type inference and checking

### ProgramSynthesizer

Type-guided program search:
- Register primitive operations
- Enumerate well-typed programs
- Verify against input-output examples
- Return minimal satisfying program

### AbstractionEngine

Hierarchical abstraction and analogy:
- Extract abstract patterns from examples
- Find structure-preserving mappings
- Transfer concepts across domains
- Build hierarchical concept lattices

## API Reference

See [docs/api.md](docs/api.md) for detailed documentation.

## Theory

The module implements ideas from:
- Tensor Product Representations (Smolensky)
- Compositional Semantics (Montague)
- Structure-Mapping Theory (Gentner)
- Program Synthesis (Gulwani)

See [docs/theory.md](docs/theory.md) for theoretical background.

## Tests

```bash
python -m pytest tests/ -v
```

144 tests covering:
- Symbol binding operations
- Type system correctness
- Program synthesis quality
- Analogy completion accuracy

## License

MIT
