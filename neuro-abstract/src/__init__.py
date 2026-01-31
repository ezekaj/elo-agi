"""
Neuro-Abstract: Compositional generalization and symbol grounding.

Implements ARC Prize-inspired capabilities:
- Type-safe symbolic binding with roles
- Hierarchical composition
- Program synthesis from examples
- Abstraction and analogy transfer
- Integration with SharedSpace
"""

from .symbolic_binder import (
    SymbolicBinder,
    CompositeBinding,
    RoleBinding,
)
from .composition_types import (
    CompositionType,
    AtomicType,
    FunctionType,
    StructuredType,
    ListType,
)
from .program_synthesis import (
    ProgramSynthesizer,
    Primitive,
    Program,
)
from .abstraction_engine import (
    AbstractionEngine,
    Abstraction,
    StructureMapping,
    Principle,
)
from .integration import (
    SharedSpaceIntegration,
    AbstractSemanticEmbedding,
)

__all__ = [
    # Symbolic Binding
    "SymbolicBinder",
    "CompositeBinding",
    "RoleBinding",
    # Type System
    "CompositionType",
    "AtomicType",
    "FunctionType",
    "StructuredType",
    "ListType",
    # Program Synthesis
    "ProgramSynthesizer",
    "Primitive",
    "Program",
    # Abstraction
    "AbstractionEngine",
    "Abstraction",
    "StructureMapping",
    "Principle",
    # Integration
    "SharedSpaceIntegration",
    "AbstractSemanticEmbedding",
]
