"""
Program Synthesis from Examples.

Implements type-guided program synthesis:
- Primitive library
- Type-guided enumeration
- Example-based synthesis
- Program verification
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable, Iterator
from itertools import product

from .composition_types import (
    CompositionType,
    FunctionType,
    ListType,
    INT,
    FLOAT,
    STR,
    BOOL,
    ANY,
)


@dataclass
class Primitive:
    """
    A primitive operation in the DSL.

    Primitives are the building blocks for synthesized programs.
    """

    name: str
    input_types: List[CompositionType]
    output_type: CompositionType
    executable: Callable[..., Any]
    cost: float = 1.0  # Search cost
    description: str = ""

    @property
    def type(self) -> FunctionType:
        """Get function type of primitive."""
        return FunctionType(tuple(self.input_types), self.output_type)

    def __call__(self, *args) -> Any:
        """Execute the primitive."""
        return self.executable(*args)

    def __repr__(self) -> str:
        return f"{self.name}: {self.type}"


@dataclass
class Program:
    """
    A synthesized program.

    Programs are trees of primitive applications.
    """

    primitive: Optional[Primitive]
    arguments: List["Program"] = field(default_factory=list)
    is_variable: bool = False
    variable_index: int = -1
    constant_value: Any = None

    @property
    def is_constant(self) -> bool:
        return self.primitive is None and not self.is_variable

    def execute(self, inputs: List[Any]) -> Any:
        """Execute the program on inputs."""
        if self.is_variable:
            return inputs[self.variable_index]

        if self.is_constant:
            return self.constant_value

        # Execute arguments first
        arg_values = [arg.execute(inputs) for arg in self.arguments]

        # Execute primitive
        return self.primitive(*arg_values)

    def __repr__(self) -> str:
        if self.is_variable:
            return f"x{self.variable_index}"
        if self.is_constant:
            return repr(self.constant_value)
        if not self.arguments:
            return self.primitive.name
        args_str = ", ".join(repr(arg) for arg in self.arguments)
        return f"{self.primitive.name}({args_str})"

    @property
    def size(self) -> int:
        """Size of program (number of nodes)."""
        if self.is_variable or self.is_constant:
            return 1
        return 1 + sum(arg.size for arg in self.arguments)

    @property
    def depth(self) -> int:
        """Depth of program tree."""
        if self.is_variable or self.is_constant:
            return 0
        if not self.arguments:
            return 1
        return 1 + max(arg.depth for arg in self.arguments)

    def cost(self) -> float:
        """Total cost of program."""
        if self.is_variable:
            return 0.1
        if self.is_constant:
            return 0.5
        base = self.primitive.cost
        return base + sum(arg.cost() for arg in self.arguments)


@dataclass
class Example:
    """An input-output example for synthesis."""

    inputs: List[Any]
    output: Any

    def __repr__(self) -> str:
        inputs_str = ", ".join(repr(i) for i in self.inputs)
        return f"({inputs_str}) -> {self.output}"


class PrimitiveLibrary:
    """
    Library of primitive operations for synthesis.
    """

    def __init__(self):
        self._primitives: Dict[str, Primitive] = {}
        self._load_default_primitives()

    def _load_default_primitives(self) -> None:
        """Load default primitive library."""
        # Arithmetic
        self.register(
            Primitive(
                name="add",
                input_types=[INT, INT],
                output_type=INT,
                executable=lambda x, y: x + y,
                description="Integer addition",
            )
        )

        self.register(
            Primitive(
                name="sub",
                input_types=[INT, INT],
                output_type=INT,
                executable=lambda x, y: x - y,
                description="Integer subtraction",
            )
        )

        self.register(
            Primitive(
                name="mul",
                input_types=[INT, INT],
                output_type=INT,
                executable=lambda x, y: x * y,
                description="Integer multiplication",
            )
        )

        self.register(
            Primitive(
                name="div",
                input_types=[INT, INT],
                output_type=INT,
                executable=lambda x, y: x // y if y != 0 else 0,
                description="Integer division",
            )
        )

        self.register(
            Primitive(
                name="mod",
                input_types=[INT, INT],
                output_type=INT,
                executable=lambda x, y: x % y if y != 0 else 0,
                description="Modulo",
            )
        )

        self.register(
            Primitive(
                name="neg",
                input_types=[INT],
                output_type=INT,
                executable=lambda x: -x,
                description="Negation",
            )
        )

        self.register(
            Primitive(
                name="abs",
                input_types=[INT],
                output_type=INT,
                executable=lambda x: abs(x),
                description="Absolute value",
            )
        )

        # Comparison
        self.register(
            Primitive(
                name="eq",
                input_types=[INT, INT],
                output_type=BOOL,
                executable=lambda x, y: x == y,
                description="Equality",
            )
        )

        self.register(
            Primitive(
                name="lt",
                input_types=[INT, INT],
                output_type=BOOL,
                executable=lambda x, y: x < y,
                description="Less than",
            )
        )

        self.register(
            Primitive(
                name="gt",
                input_types=[INT, INT],
                output_type=BOOL,
                executable=lambda x, y: x > y,
                description="Greater than",
            )
        )

        # Boolean
        self.register(
            Primitive(
                name="and",
                input_types=[BOOL, BOOL],
                output_type=BOOL,
                executable=lambda x, y: x and y,
                description="Logical AND",
            )
        )

        self.register(
            Primitive(
                name="or",
                input_types=[BOOL, BOOL],
                output_type=BOOL,
                executable=lambda x, y: x or y,
                description="Logical OR",
            )
        )

        self.register(
            Primitive(
                name="not",
                input_types=[BOOL],
                output_type=BOOL,
                executable=lambda x: not x,
                description="Logical NOT",
            )
        )

        # Conditionals
        self.register(
            Primitive(
                name="if",
                input_types=[BOOL, INT, INT],
                output_type=INT,
                executable=lambda c, t, f: t if c else f,
                cost=2.0,
                description="Conditional",
            )
        )

        # List operations
        self.register(
            Primitive(
                name="head",
                input_types=[ListType(INT)],
                output_type=INT,
                executable=lambda xs: xs[0] if xs else 0,
                description="First element",
            )
        )

        self.register(
            Primitive(
                name="tail",
                input_types=[ListType(INT)],
                output_type=ListType(INT),
                executable=lambda xs: xs[1:],
                description="All but first",
            )
        )

        self.register(
            Primitive(
                name="cons",
                input_types=[INT, ListType(INT)],
                output_type=ListType(INT),
                executable=lambda x, xs: [x] + list(xs),
                description="Prepend element",
            )
        )

        self.register(
            Primitive(
                name="len",
                input_types=[ListType(INT)],
                output_type=INT,
                executable=lambda xs: len(xs),
                description="List length",
            )
        )

        self.register(
            Primitive(
                name="sum",
                input_types=[ListType(INT)],
                output_type=INT,
                executable=lambda xs: sum(xs),
                description="Sum of list",
            )
        )

        self.register(
            Primitive(
                name="max",
                input_types=[ListType(INT)],
                output_type=INT,
                executable=lambda xs: max(xs) if xs else 0,
                description="Maximum",
            )
        )

        self.register(
            Primitive(
                name="min",
                input_types=[ListType(INT)],
                output_type=INT,
                executable=lambda xs: min(xs) if xs else 0,
                description="Minimum",
            )
        )

        self.register(
            Primitive(
                name="reverse",
                input_types=[ListType(INT)],
                output_type=ListType(INT),
                executable=lambda xs: list(reversed(xs)),
                description="Reverse list",
            )
        )

        self.register(
            Primitive(
                name="sort",
                input_types=[ListType(INT)],
                output_type=ListType(INT),
                executable=lambda xs: sorted(xs),
                description="Sort list",
            )
        )

        # Constants
        self.register(
            Primitive(
                name="zero",
                input_types=[],
                output_type=INT,
                executable=lambda: 0,
                cost=0.1,
                description="Zero constant",
            )
        )

        self.register(
            Primitive(
                name="one",
                input_types=[],
                output_type=INT,
                executable=lambda: 1,
                cost=0.1,
                description="One constant",
            )
        )

        self.register(
            Primitive(
                name="empty",
                input_types=[],
                output_type=ListType(INT),
                executable=lambda: [],
                cost=0.1,
                description="Empty list",
            )
        )

    def register(self, primitive: Primitive) -> None:
        """Register a primitive."""
        self._primitives[primitive.name] = primitive

    def get(self, name: str) -> Optional[Primitive]:
        """Get primitive by name."""
        return self._primitives.get(name)

    def all_primitives(self) -> List[Primitive]:
        """Get all primitives."""
        return list(self._primitives.values())

    def primitives_with_output(self, output_type: CompositionType) -> List[Primitive]:
        """Get primitives that produce a given output type."""
        return [
            p
            for p in self._primitives.values()
            if p.output_type == output_type or output_type == ANY
        ]


class ProgramSynthesizer:
    """
    Type-guided program synthesizer.

    Synthesizes programs from input-output examples using:
    - Bottom-up enumeration
    - Type-guided search
    - Observational equivalence pruning
    """

    def __init__(
        self,
        library: Optional[PrimitiveLibrary] = None,
        max_size: int = 10,
        max_depth: int = 5,
        timeout_per_example: float = 1.0,
    ):
        self.library = library or PrimitiveLibrary()
        self.max_size = max_size
        self.max_depth = max_depth
        self.timeout_per_example = timeout_per_example

        # Statistics
        self._programs_enumerated = 0
        self._programs_evaluated = 0

    def register_primitive(self, primitive: Primitive) -> None:
        """Register a new primitive."""
        self.library.register(primitive)

    def synthesize_from_examples(
        self,
        examples: List[Example],
        input_types: List[CompositionType],
        output_type: CompositionType,
    ) -> Optional[Program]:
        """
        Synthesize a program from input-output examples.

        Args:
            examples: List of input-output examples
            input_types: Types of input variables
            output_type: Expected output type

        Returns:
            A program that satisfies all examples, or None
        """
        # Generate candidate programs
        for program in self.generate_candidates(input_types, output_type):
            if self.verify_program(program, examples):
                return program

        return None

    def generate_candidates(
        self,
        input_types: List[CompositionType],
        output_type: CompositionType,
    ) -> Iterator[Program]:
        """Generate candidate programs in order of increasing size."""
        # Bottom-up enumeration
        for size in range(1, self.max_size + 1):
            for program in self._enumerate_size(size, input_types, output_type, 0):
                self._programs_enumerated += 1
                yield program

    def _enumerate_size(
        self,
        size: int,
        input_types: List[CompositionType],
        target_type: CompositionType,
        depth: int,
    ) -> Iterator[Program]:
        """Enumerate programs of a specific size."""
        if depth > self.max_depth:
            return

        if size == 1:
            # Variables
            for i, typ in enumerate(input_types):
                if typ == target_type or target_type == ANY:
                    yield Program(
                        primitive=None,
                        is_variable=True,
                        variable_index=i,
                    )

            # Zero-argument primitives
            for prim in self.library.primitives_with_output(target_type):
                if len(prim.input_types) == 0:
                    yield Program(primitive=prim)

            return

        # Primitives with arguments
        for prim in self.library.primitives_with_output(target_type):
            if len(prim.input_types) == 0:
                continue

            n_args = len(prim.input_types)
            remaining_size = size - 1

            # Distribute size among arguments
            for size_dist in self._distribute_size(remaining_size, n_args):
                arg_programs_list = []
                valid = True

                for i, arg_size in enumerate(size_dist):
                    arg_type = prim.input_types[i]
                    arg_programs = list(
                        self._enumerate_size(arg_size, input_types, arg_type, depth + 1)
                    )

                    if not arg_programs:
                        valid = False
                        break

                    arg_programs_list.append(arg_programs)

                if not valid:
                    continue

                # Generate all combinations of arguments
                for args in product(*arg_programs_list):
                    yield Program(primitive=prim, arguments=list(args))

    def _distribute_size(
        self,
        total: int,
        n_parts: int,
    ) -> Iterator[Tuple[int, ...]]:
        """Distribute total size among n parts."""
        if n_parts == 1:
            yield (total,)
            return

        for first in range(1, total - n_parts + 2):
            for rest in self._distribute_size(total - first, n_parts - 1):
                yield (first,) + rest

    def verify_program(
        self,
        program: Program,
        examples: List[Example],
    ) -> bool:
        """Verify that program satisfies all examples."""
        self._programs_evaluated += 1

        for example in examples:
            try:
                result = program.execute(example.inputs)
                if result != example.output:
                    return False
            except Exception:
                return False

        return True

    def rank_programs(
        self,
        programs: List[Program],
        examples: List[Example],
    ) -> List[Tuple[Program, float]]:
        """
        Rank programs by how many examples they satisfy.

        Returns list of (program, score) sorted by score descending.
        """
        scores = []

        for program in programs:
            score = 0.0
            for example in examples:
                try:
                    result = program.execute(example.inputs)
                    if result == example.output:
                        score += 1.0
                except Exception:
                    pass

            score /= len(examples)  # Normalize to [0, 1]
            scores.append((program, score))

        scores.sort(key=lambda x: (-x[1], x[0].cost()))
        return scores

    def infer_type(self, examples: List[Example]) -> Tuple[List[CompositionType], CompositionType]:
        """Infer input and output types from examples."""
        if not examples:
            return [ANY], ANY

        # Infer input types
        n_inputs = len(examples[0].inputs)
        input_types = []

        for i in range(n_inputs):
            values = [ex.inputs[i] for ex in examples]
            input_types.append(self._infer_type_from_values(values))

        # Infer output type
        outputs = [ex.output for ex in examples]
        output_type = self._infer_type_from_values(outputs)

        return input_types, output_type

    def _infer_type_from_values(self, values: List[Any]) -> CompositionType:
        """Infer type from a list of values."""
        if not values:
            return ANY

        sample = values[0]

        if isinstance(sample, bool):
            return BOOL
        if isinstance(sample, int):
            return INT
        if isinstance(sample, float):
            return FLOAT
        if isinstance(sample, str):
            return STR
        if isinstance(sample, list):
            if sample:
                elem_type = self._infer_type_from_values([sample[0]])
            else:
                elem_type = INT  # Default
            return ListType(elem_type)

        return ANY

    def synthesize_auto(
        self,
        examples: List[Example],
    ) -> Optional[Program]:
        """Synthesize with automatic type inference."""
        input_types, output_type = self.infer_type(examples)
        return self.synthesize_from_examples(examples, input_types, output_type)

    def statistics(self) -> Dict[str, Any]:
        """Get synthesizer statistics."""
        return {
            "programs_enumerated": self._programs_enumerated,
            "programs_evaluated": self._programs_evaluated,
            "max_size": self.max_size,
            "max_depth": self.max_depth,
            "n_primitives": len(self.library.all_primitives()),
        }
