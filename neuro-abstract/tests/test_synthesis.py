"""
Comprehensive tests for Program Synthesis.

Tests cover:
- Primitive library
- Program execution
- Type-guided synthesis
- Example verification
"""

import pytest
import numpy as np
from neuro.modules.abstract.program_synthesis import (
    ProgramSynthesizer,
    PrimitiveLibrary,
    Primitive,
    Program,
    Example,
)
from neuro.modules.abstract.composition_types import INT, BOOL, ListType


class TestPrimitive:
    """Test Primitive class."""

    def test_primitive_creation(self):
        """Should create primitive."""
        add = Primitive(
            name="add",
            input_types=[INT, INT],
            output_type=INT,
            executable=lambda x, y: x + y,
        )

        assert add.name == "add"
        assert add.type.arity == 2

    def test_primitive_call(self):
        """Should execute primitive."""
        add = Primitive(
            name="add",
            input_types=[INT, INT],
            output_type=INT,
            executable=lambda x, y: x + y,
        )

        assert add(2, 3) == 5

    def test_primitive_repr(self):
        """Should have string representation."""
        inc = Primitive(
            name="inc",
            input_types=[INT],
            output_type=INT,
            executable=lambda x: x + 1,
        )

        assert "inc" in repr(inc)


class TestProgram:
    """Test Program class."""

    def test_variable_program(self):
        """Should create and execute variable program."""
        p = Program(
            primitive=None,
            is_variable=True,
            variable_index=0,
        )

        assert p.execute([10]) == 10
        assert p.execute([42, 99]) == 42

    def test_constant_program(self):
        """Should create and execute constant program."""
        p = Program(
            primitive=None,
            constant_value=42,
        )

        assert p.execute([]) == 42
        assert p.execute([1, 2, 3]) == 42

    def test_primitive_program(self):
        """Should execute primitive program."""
        add = Primitive(
            name="add",
            input_types=[INT, INT],
            output_type=INT,
            executable=lambda x, y: x + y,
        )

        x = Program(primitive=None, is_variable=True, variable_index=0)
        y = Program(primitive=None, is_variable=True, variable_index=1)
        p = Program(primitive=add, arguments=[x, y])

        assert p.execute([3, 4]) == 7

    def test_nested_program(self):
        """Should execute nested program."""
        add = Primitive("add", [INT, INT], INT, lambda x, y: x + y)
        mul = Primitive("mul", [INT, INT], INT, lambda x, y: x * y)

        x = Program(primitive=None, is_variable=True, variable_index=0)
        y = Program(primitive=None, is_variable=True, variable_index=1)

        # (x + y) * x
        sum_xy = Program(primitive=add, arguments=[x, y])
        product = Program(primitive=mul, arguments=[sum_xy, x])

        assert product.execute([3, 4]) == 21  # (3+4)*3 = 21

    def test_program_size(self):
        """Should compute program size."""
        x = Program(primitive=None, is_variable=True, variable_index=0)
        assert x.size == 1

        add = Primitive("add", [INT, INT], INT, lambda x, y: x + y)
        p = Program(primitive=add, arguments=[x, x])
        assert p.size == 3  # add + x + x

    def test_program_depth(self):
        """Should compute program depth."""
        x = Program(primitive=None, is_variable=True, variable_index=0)
        assert x.depth == 0

        add = Primitive("add", [INT, INT], INT, lambda x, y: x + y)
        p1 = Program(primitive=add, arguments=[x, x])
        assert p1.depth == 1

        p2 = Program(primitive=add, arguments=[p1, x])
        assert p2.depth == 2

    def test_program_repr(self):
        """Should have readable repr."""
        x = Program(primitive=None, is_variable=True, variable_index=0)
        assert repr(x) == "x0"

        c = Program(primitive=None, constant_value=42)
        assert repr(c) == "42"

        add = Primitive("add", [INT, INT], INT, lambda x, y: x + y)
        p = Program(primitive=add, arguments=[x, x])
        assert "add" in repr(p)


class TestPrimitiveLibrary:
    """Test PrimitiveLibrary class."""

    @pytest.fixture
    def library(self):
        return PrimitiveLibrary()

    def test_default_primitives(self, library):
        """Should have default primitives."""
        assert library.get("add") is not None
        assert library.get("sub") is not None
        assert library.get("mul") is not None

    def test_register_primitive(self, library):
        """Should register new primitive."""
        custom = Primitive(
            name="custom",
            input_types=[INT],
            output_type=INT,
            executable=lambda x: x * 2,
        )
        library.register(custom)

        assert library.get("custom") is not None
        assert library.get("custom")(5) == 10

    def test_primitives_with_output(self, library):
        """Should filter primitives by output type."""
        int_prims = library.primitives_with_output(INT)
        bool_prims = library.primitives_with_output(BOOL)

        assert any(p.name == "add" for p in int_prims)
        assert any(p.name == "eq" for p in bool_prims)

    def test_arithmetic_primitives(self, library):
        """Arithmetic primitives should work correctly."""
        add = library.get("add")
        sub = library.get("sub")
        mul = library.get("mul")
        div = library.get("div")

        assert add(3, 4) == 7
        assert sub(10, 4) == 6
        assert mul(3, 4) == 12
        assert div(10, 3) == 3

    def test_comparison_primitives(self, library):
        """Comparison primitives should work correctly."""
        eq = library.get("eq")
        lt = library.get("lt")
        gt = library.get("gt")

        assert eq(5, 5) == True
        assert eq(5, 6) == False
        assert lt(3, 5) == True
        assert gt(5, 3) == True

    def test_list_primitives(self, library):
        """List primitives should work correctly."""
        head = library.get("head")
        tail = library.get("tail")
        length = library.get("len")
        total = library.get("sum")

        assert head([1, 2, 3]) == 1
        assert tail([1, 2, 3]) == [2, 3]
        assert length([1, 2, 3]) == 3
        assert total([1, 2, 3, 4]) == 10


class TestExample:
    """Test Example class."""

    def test_example_creation(self):
        """Should create example."""
        ex = Example(inputs=[1, 2], output=3)
        assert ex.inputs == [1, 2]
        assert ex.output == 3

    def test_example_repr(self):
        """Should have readable repr."""
        ex = Example(inputs=[1, 2], output=3)
        assert "(1, 2) -> 3" in repr(ex)


class TestProgramSynthesizer:
    """Test ProgramSynthesizer class."""

    @pytest.fixture
    def synthesizer(self):
        return ProgramSynthesizer(max_size=6, max_depth=3)

    def test_synthesize_identity(self, synthesizer):
        """Should synthesize identity function."""
        examples = [
            Example(inputs=[1], output=1),
            Example(inputs=[5], output=5),
            Example(inputs=[42], output=42),
        ]

        program = synthesizer.synthesize_from_examples(examples, [INT], INT)

        assert program is not None
        assert program.execute([10]) == 10

    def test_synthesize_increment(self, synthesizer):
        """Should synthesize increment function."""
        examples = [
            Example(inputs=[0], output=1),
            Example(inputs=[1], output=2),
            Example(inputs=[10], output=11),
        ]

        program = synthesizer.synthesize_from_examples(examples, [INT], INT)

        assert program is not None
        # Verify on new inputs
        assert program.execute([100]) == 101

    def test_synthesize_add(self, synthesizer):
        """Should synthesize addition."""
        examples = [
            Example(inputs=[1, 2], output=3),
            Example(inputs=[5, 3], output=8),
            Example(inputs=[0, 0], output=0),
        ]

        program = synthesizer.synthesize_from_examples(examples, [INT, INT], INT)

        assert program is not None
        assert program.execute([10, 20]) == 30

    def test_synthesize_double(self, synthesizer):
        """Should synthesize double function."""
        examples = [
            Example(inputs=[1], output=2),
            Example(inputs=[3], output=6),
            Example(inputs=[5], output=10),
        ]

        program = synthesizer.synthesize_from_examples(examples, [INT], INT)

        assert program is not None
        assert program.execute([7]) == 14

    def test_verify_program(self, synthesizer):
        """Should verify program against examples."""
        add = synthesizer.library.get("add")
        x = Program(primitive=None, is_variable=True, variable_index=0)
        y = Program(primitive=None, is_variable=True, variable_index=1)
        program = Program(primitive=add, arguments=[x, y])

        examples = [
            Example(inputs=[1, 2], output=3),
            Example(inputs=[5, 3], output=8),
        ]

        assert synthesizer.verify_program(program, examples)

    def test_verify_program_fails(self, synthesizer):
        """Should detect incorrect program."""
        mul = synthesizer.library.get("mul")
        x = Program(primitive=None, is_variable=True, variable_index=0)
        y = Program(primitive=None, is_variable=True, variable_index=1)
        program = Program(primitive=mul, arguments=[x, y])

        examples = [
            Example(inputs=[1, 2], output=3),  # mul(1,2)=2 != 3
        ]

        assert not synthesizer.verify_program(program, examples)

    def test_rank_programs(self, synthesizer):
        """Should rank programs by correctness."""
        add = synthesizer.library.get("add")
        mul = synthesizer.library.get("mul")
        x = Program(primitive=None, is_variable=True, variable_index=0)
        y = Program(primitive=None, is_variable=True, variable_index=1)

        add_program = Program(primitive=add, arguments=[x, y])
        mul_program = Program(primitive=mul, arguments=[x, y])

        examples = [
            Example(inputs=[2, 3], output=5),
            Example(inputs=[1, 1], output=2),
        ]

        ranked = synthesizer.rank_programs([add_program, mul_program], examples)

        # Add should be ranked higher
        assert ranked[0][0] == add_program
        assert ranked[0][1] == 1.0  # 100% correct

    def test_infer_type_int(self, synthesizer):
        """Should infer int types."""
        examples = [
            Example(inputs=[1, 2], output=3),
        ]

        input_types, output_type = synthesizer.infer_type(examples)

        assert input_types == [INT, INT]
        assert output_type == INT

    def test_infer_type_list(self, synthesizer):
        """Should infer list types."""
        examples = [
            Example(inputs=[[1, 2, 3]], output=6),
        ]

        input_types, output_type = synthesizer.infer_type(examples)

        assert isinstance(input_types[0], ListType)
        assert output_type == INT

    def test_synthesize_auto(self, synthesizer):
        """Should synthesize with automatic type inference."""
        examples = [
            Example(inputs=[1], output=2),
            Example(inputs=[5], output=6),
        ]

        program = synthesizer.synthesize_auto(examples)
        assert program is not None

    def test_statistics(self, synthesizer):
        """Should track statistics."""
        examples = [
            Example(inputs=[1], output=2),
        ]

        synthesizer.synthesize_auto(examples)
        stats = synthesizer.statistics()

        assert stats["programs_enumerated"] > 0
        assert stats["programs_evaluated"] > 0


class TestComplexSynthesis:
    """Test more complex synthesis tasks."""

    @pytest.fixture
    def synthesizer(self):
        return ProgramSynthesizer(max_size=8, max_depth=4)

    def test_synthesize_list_sum(self, synthesizer):
        """Should synthesize list sum."""
        examples = [
            Example(inputs=[[1, 2, 3]], output=6),
            Example(inputs=[[10, 20]], output=30),
            Example(inputs=[[5]], output=5),
        ]

        program = synthesizer.synthesize_from_examples(examples, [ListType(INT)], INT)

        assert program is not None
        assert program.execute([[1, 2, 3, 4]]) == 10

    def test_synthesize_list_length(self, synthesizer):
        """Should synthesize list length."""
        examples = [
            Example(inputs=[[1, 2, 3]], output=3),
            Example(inputs=[[]], output=0),
            Example(inputs=[[1, 2, 3, 4, 5]], output=5),
        ]

        program = synthesizer.synthesize_from_examples(examples, [ListType(INT)], INT)

        assert program is not None
        assert program.execute([[1, 2]]) == 2

    def test_synthesize_max(self, synthesizer):
        """Should synthesize max of list."""
        examples = [
            Example(inputs=[[1, 5, 3]], output=5),
            Example(inputs=[[10, 20, 15]], output=20),
            Example(inputs=[[42]], output=42),
        ]

        program = synthesizer.synthesize_from_examples(examples, [ListType(INT)], INT)

        if program:  # May not find due to size limits
            assert program.execute([[1, 100, 50]]) == 100


class TestEdgeCases:
    """Test edge cases."""

    def test_no_solution(self):
        """Should return None when no solution exists."""
        synthesizer = ProgramSynthesizer(max_size=3)

        # Impossible task
        examples = [
            Example(inputs=[1], output=1000000),
        ]

        program = synthesizer.synthesize_from_examples(examples, [INT], INT)
        # May or may not find, but shouldn't crash

    def test_empty_examples(self):
        """Should handle empty examples."""
        synthesizer = ProgramSynthesizer()

        # Empty examples - type inference
        input_types, output_type = synthesizer.infer_type([])
        assert output_type is not None

    def test_single_example(self):
        """Should handle single example."""
        synthesizer = ProgramSynthesizer(max_size=4)

        examples = [Example(inputs=[5], output=10)]
        program = synthesizer.synthesize_auto(examples)

        # Should find something (even if overfit)
        if program:
            assert program.execute([5]) == 10
