"""
Compositional Type System.

Implements a type system for compositional generalization:
- Atomic types (int, str, bool, etc.)
- Function types (A -> B)
- Structured types (records)
- List/sequence types
- Type inference and checking
- Neural type representations
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np


class TypeKind(Enum):
    """Kinds of types in the system."""

    ATOMIC = "atomic"
    FUNCTION = "function"
    STRUCTURED = "structured"
    LIST = "list"
    UNION = "union"
    VARIABLE = "variable"  # Type variable for polymorphism


class CompositionType(ABC):
    """
    Abstract base class for compositional types.

    Types support:
    - Composition with other types
    - Validation of values
    - Neural representation
    """

    kind: TypeKind

    @abstractmethod
    def compose(self, *args: "CompositionType") -> "CompositionType":
        """Compose this type with other types."""
        pass

    @abstractmethod
    def validate(self, value: Any) -> bool:
        """Validate that a value conforms to this type."""
        pass

    @abstractmethod
    def to_neural(self, embedding_dim: int = 64) -> np.ndarray:
        """Convert type to neural representation."""
        pass

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """Check type equality."""
        pass

    @abstractmethod
    def __hash__(self) -> int:
        """Hash for type."""
        pass

    @abstractmethod
    def __repr__(self) -> str:
        """String representation."""
        pass

    def is_subtype_of(self, other: "CompositionType") -> bool:
        """Check if this type is a subtype of another."""
        return self == other

    def unify(self, other: "CompositionType") -> Optional["CompositionType"]:
        """Attempt to unify two types, returning the unified type or None."""
        if self == other:
            return self
        return None


@dataclass(frozen=True)
class AtomicType(CompositionType):
    """
    An atomic (primitive) type.

    Examples: int, str, bool, float
    """

    name: str
    python_type: Optional[type] = None

    def __post_init__(self):
        object.__setattr__(self, "kind", TypeKind.ATOMIC)

    def compose(self, *args: CompositionType) -> CompositionType:
        """Atomic types don't compose directly."""
        if not args:
            return self
        raise TypeError(f"Cannot compose atomic type {self.name}")

    def validate(self, value: Any) -> bool:
        """Validate value against atomic type."""
        if self.python_type is not None:
            return isinstance(value, self.python_type)

        # String-based validation
        type_validators = {
            "int": lambda v: isinstance(v, int),
            "float": lambda v: isinstance(v, (int, float)),
            "str": lambda v: isinstance(v, str),
            "bool": lambda v: isinstance(v, bool),
            "any": lambda v: True,
            "none": lambda v: v is None,
        }

        validator = type_validators.get(self.name.lower(), lambda v: True)
        return validator(value)

    def to_neural(self, embedding_dim: int = 64) -> np.ndarray:
        """Convert to neural representation using type name hash."""
        np.random.seed(hash(self.name) % (2**32))
        rep = np.random.randn(embedding_dim)
        return rep / np.linalg.norm(rep)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, AtomicType):
            return self.name == other.name
        return False

    def __hash__(self) -> int:
        return hash(("atomic", self.name))

    def __repr__(self) -> str:
        return self.name


# Common atomic types
INT = AtomicType("int", int)
FLOAT = AtomicType("float")  # No python_type so int is also accepted as float
STR = AtomicType("str", str)
BOOL = AtomicType("bool", bool)
ANY = AtomicType("any", None)
NONE = AtomicType("none", type(None))


@dataclass(frozen=True)
class FunctionType(CompositionType):
    """
    A function type: input_types -> output_type

    Represents callable functions with typed inputs and outputs.
    """

    input_types: Tuple[CompositionType, ...]
    output_type: CompositionType

    def __post_init__(self):
        object.__setattr__(self, "kind", TypeKind.FUNCTION)

    def compose(self, *args: CompositionType) -> CompositionType:
        """
        Apply function type to arguments.

        Returns output type if arguments match input types.
        """
        if len(args) != len(self.input_types):
            raise TypeError(f"Expected {len(self.input_types)} arguments, got {len(args)}")

        for i, (arg, expected) in enumerate(zip(args, self.input_types)):
            if not arg.is_subtype_of(expected):
                raise TypeError(f"Argument {i}: expected {expected}, got {arg}")

        return self.output_type

    def validate(self, value: Any) -> bool:
        """Validate that value is a callable."""
        return callable(value)

    def to_neural(self, embedding_dim: int = 64) -> np.ndarray:
        """Convert to neural representation."""
        # Combine input and output type representations
        parts = []
        for inp in self.input_types:
            parts.append(inp.to_neural(embedding_dim // 4))

        parts.append(self.output_type.to_neural(embedding_dim // 4))

        # Pad or truncate to embedding_dim
        combined = np.concatenate(parts)
        if len(combined) < embedding_dim:
            combined = np.pad(combined, (0, embedding_dim - len(combined)))
        else:
            combined = combined[:embedding_dim]

        return combined / (np.linalg.norm(combined) + 1e-8)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FunctionType):
            return self.input_types == other.input_types and self.output_type == other.output_type
        return False

    def __hash__(self) -> int:
        return hash(("function", self.input_types, self.output_type))

    def __repr__(self) -> str:
        inputs = ", ".join(repr(t) for t in self.input_types)
        return f"({inputs}) -> {self.output_type}"

    @property
    def arity(self) -> int:
        """Number of input arguments."""
        return len(self.input_types)

    def curry(self) -> "FunctionType":
        """Convert to curried form: (A, B) -> C becomes A -> (B -> C)."""
        if len(self.input_types) <= 1:
            return self

        first = self.input_types[0]
        rest = self.input_types[1:]
        inner = FunctionType(rest, self.output_type)
        return FunctionType((first,), inner)


@dataclass(frozen=True)
class StructuredType(CompositionType):
    """
    A structured type (record/object type).

    Represents types with named fields.
    """

    name: str
    fields: Tuple[Tuple[str, CompositionType], ...]  # Immutable field list

    def __post_init__(self):
        object.__setattr__(self, "kind", TypeKind.STRUCTURED)

    def compose(self, *args: CompositionType) -> CompositionType:
        """Compose structured types (merge fields)."""
        if not args:
            return self

        all_fields = dict(self.fields)
        for arg in args:
            if isinstance(arg, StructuredType):
                for field_name, field_type in arg.fields:
                    if field_name in all_fields:
                        # Type must match
                        if all_fields[field_name] != field_type:
                            raise TypeError(f"Conflicting types for field {field_name}")
                    else:
                        all_fields[field_name] = field_type

        return StructuredType(f"{self.name}_composed", tuple(all_fields.items()))

    def validate(self, value: Any) -> bool:
        """Validate that value has required fields with correct types."""
        if not isinstance(value, dict):
            # Try to access as object attributes
            for field_name, field_type in self.fields:
                if not hasattr(value, field_name):
                    return False
                if not field_type.validate(getattr(value, field_name)):
                    return False
            return True

        # Dict validation
        for field_name, field_type in self.fields:
            if field_name not in value:
                return False
            if not field_type.validate(value[field_name]):
                return False
        return True

    def to_neural(self, embedding_dim: int = 64) -> np.ndarray:
        """Convert to neural representation."""
        parts = []
        for field_name, field_type in self.fields:
            # Hash field name
            np.random.seed(hash(field_name) % (2**32))
            name_rep = np.random.randn(embedding_dim // (2 * len(self.fields) + 1))

            # Field type representation
            type_rep = field_type.to_neural(embedding_dim // (2 * len(self.fields) + 1))

            parts.extend([name_rep, type_rep])

        combined = np.concatenate(parts) if parts else np.zeros(embedding_dim)
        if len(combined) < embedding_dim:
            combined = np.pad(combined, (0, embedding_dim - len(combined)))
        else:
            combined = combined[:embedding_dim]

        return combined / (np.linalg.norm(combined) + 1e-8)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, StructuredType):
            return self.fields == other.fields
        return False

    def __hash__(self) -> int:
        return hash(("structured", self.fields))

    def __repr__(self) -> str:
        fields_str = ", ".join(f"{n}: {t}" for n, t in self.fields)
        return f"{{{fields_str}}}"

    def get_field_type(self, field_name: str) -> Optional[CompositionType]:
        """Get type of a specific field."""
        for name, typ in self.fields:
            if name == field_name:
                return typ
        return None

    def has_field(self, field_name: str) -> bool:
        """Check if field exists."""
        return any(name == field_name for name, _ in self.fields)

    def is_subtype_of(self, other: "CompositionType") -> bool:
        """Structural subtyping: more fields is a subtype."""
        if not isinstance(other, StructuredType):
            return False

        # Self must have all fields of other with compatible types
        other_fields = dict(other.fields)
        for name, typ in self.fields:
            if name in other_fields:
                if not typ.is_subtype_of(other_fields[name]):
                    return False

        # Other must not have fields self doesn't have
        self_field_names = {name for name, _ in self.fields}
        for name in other_fields:
            if name not in self_field_names:
                return False

        return True


@dataclass(frozen=True)
class ListType(CompositionType):
    """
    A list/sequence type.

    Represents homogeneous collections.
    """

    element_type: CompositionType

    def __post_init__(self):
        object.__setattr__(self, "kind", TypeKind.LIST)

    def compose(self, *args: CompositionType) -> CompositionType:
        """Compose list types (unify element types)."""
        if not args:
            return self

        unified_element = self.element_type
        for arg in args:
            if isinstance(arg, ListType):
                result = unified_element.unify(arg.element_type)
                if result is None:
                    raise TypeError(
                        f"Cannot unify list element types: {unified_element} and {arg.element_type}"
                    )
                unified_element = result

        return ListType(unified_element)

    def validate(self, value: Any) -> bool:
        """Validate that value is a list with correct element types."""
        if not isinstance(value, (list, tuple)):
            return False

        return all(self.element_type.validate(elem) for elem in value)

    def to_neural(self, embedding_dim: int = 64) -> np.ndarray:
        """Convert to neural representation."""
        # List marker + element type
        np.random.seed(hash("list_marker") % (2**32))
        marker = np.random.randn(embedding_dim // 2)

        element_rep = self.element_type.to_neural(embedding_dim // 2)

        combined = np.concatenate([marker, element_rep])
        return combined / (np.linalg.norm(combined) + 1e-8)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ListType):
            return self.element_type == other.element_type
        return False

    def __hash__(self) -> int:
        return hash(("list", self.element_type))

    def __repr__(self) -> str:
        return f"[{self.element_type}]"


@dataclass(frozen=True)
class UnionType(CompositionType):
    """
    A union type: A | B

    Value can be any of the member types.
    """

    types: Tuple[CompositionType, ...]

    def __post_init__(self):
        object.__setattr__(self, "kind", TypeKind.UNION)

    def compose(self, *args: CompositionType) -> CompositionType:
        """Compose union types."""
        if not args:
            return self

        all_types = set(self.types)
        for arg in args:
            if isinstance(arg, UnionType):
                all_types.update(arg.types)
            else:
                all_types.add(arg)

        return UnionType(tuple(all_types))

    def validate(self, value: Any) -> bool:
        """Value must match at least one member type."""
        return any(t.validate(value) for t in self.types)

    def to_neural(self, embedding_dim: int = 64) -> np.ndarray:
        """Average of member type representations."""
        if not self.types:
            return np.zeros(embedding_dim)

        reps = [t.to_neural(embedding_dim) for t in self.types]
        combined = np.mean(reps, axis=0)
        return combined / (np.linalg.norm(combined) + 1e-8)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, UnionType):
            return set(self.types) == set(other.types)
        return False

    def __hash__(self) -> int:
        return hash(("union", frozenset(self.types)))

    def __repr__(self) -> str:
        return " | ".join(repr(t) for t in self.types)


@dataclass
class TypeVariable(CompositionType):
    """
    A type variable for polymorphism.

    Example: In map: (A -> B) -> [A] -> [B], A and B are type variables.
    """

    name: str
    bound: Optional[CompositionType] = None  # Upper bound

    def __post_init__(self):
        self.kind = TypeKind.VARIABLE

    def compose(self, *args: CompositionType) -> CompositionType:
        """Type variables compose by unification."""
        if not args:
            return self
        return args[0]  # Instantiate to first argument

    def validate(self, value: Any) -> bool:
        """Type variables accept any value (or value matching bound)."""
        if self.bound is not None:
            return self.bound.validate(value)
        return True

    def to_neural(self, embedding_dim: int = 64) -> np.ndarray:
        """Variable representation."""
        np.random.seed(hash(f"var_{self.name}") % (2**32))
        rep = np.random.randn(embedding_dim)
        return rep / np.linalg.norm(rep)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TypeVariable):
            return self.name == other.name
        return False

    def __hash__(self) -> int:
        return hash(("variable", self.name))

    def __repr__(self) -> str:
        if self.bound:
            return f"{self.name} <: {self.bound}"
        return self.name


class TypeInferencer:
    """
    Type inference engine.

    Infers types from values and expressions.
    """

    def __init__(self):
        self._type_variables: Dict[str, CompositionType] = {}
        self._var_counter = 0

    def fresh_variable(self, prefix: str = "T") -> TypeVariable:
        """Create a fresh type variable."""
        self._var_counter += 1
        return TypeVariable(f"{prefix}{self._var_counter}")

    def infer(self, value: Any) -> CompositionType:
        """Infer the type of a value."""
        if value is None:
            return NONE

        if isinstance(value, bool):
            return BOOL
        if isinstance(value, int):
            return INT
        if isinstance(value, float):
            return FLOAT
        if isinstance(value, str):
            return STR

        if isinstance(value, (list, tuple)):
            if not value:
                return ListType(self.fresh_variable("E"))
            element_types = [self.infer(elem) for elem in value]
            unified = self._unify_all(element_types)
            return ListType(unified)

        if isinstance(value, dict):
            fields = tuple((k, self.infer(v)) for k, v in value.items())
            return StructuredType("inferred", fields)

        if callable(value):
            return FunctionType((ANY,), ANY)

        return ANY

    def _unify_all(self, types: List[CompositionType]) -> CompositionType:
        """Unify a list of types."""
        if not types:
            return self.fresh_variable()

        result = types[0]
        for t in types[1:]:
            unified = result.unify(t)
            if unified is not None:
                result = unified
            else:
                # Create union type
                result = UnionType((result, t))

        return result

    def check(
        self,
        value: Any,
        expected_type: CompositionType,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if value conforms to expected type.

        Returns (is_valid, error_message)
        """
        if expected_type.validate(value):
            return True, None

        inferred = self.infer(value)
        return False, f"Expected {expected_type}, got {inferred}"

    def statistics(self) -> Dict[str, Any]:
        """Get inferencer statistics."""
        return {
            "variables_created": self._var_counter,
        }
