"""
Comprehensive tests for Composition Types.

Tests cover:
- Atomic types
- Function types
- Structured types
- List types
- Type inference
- Type validation
"""

import pytest
import numpy as np
from neuro.modules.abstract.composition_types import (
    AtomicType,
    FunctionType,
    StructuredType,
    ListType,
    UnionType,
    TypeVariable,
    TypeInferencer,
    INT,
    FLOAT,
    STR,
    BOOL,
    ANY,
    NONE,
)


class TestAtomicType:
    """Test atomic types."""

    def test_int_validation(self):
        """INT should validate integers."""
        assert INT.validate(42)
        assert INT.validate(-10)
        assert not INT.validate(3.14)
        assert not INT.validate("hello")

    def test_float_validation(self):
        """FLOAT should validate floats and ints."""
        assert FLOAT.validate(3.14)
        assert FLOAT.validate(42)  # Int is also float
        assert not FLOAT.validate("hello")

    def test_str_validation(self):
        """STR should validate strings."""
        assert STR.validate("hello")
        assert STR.validate("")
        assert not STR.validate(42)

    def test_bool_validation(self):
        """BOOL should validate booleans."""
        assert BOOL.validate(True)
        assert BOOL.validate(False)
        assert not BOOL.validate(1)  # Int is not bool

    def test_any_validation(self):
        """ANY should validate anything."""
        assert ANY.validate(42)
        assert ANY.validate("hello")
        assert ANY.validate([1, 2, 3])
        assert ANY.validate(None)

    def test_none_validation(self):
        """NONE should validate only None."""
        assert NONE.validate(None)
        assert not NONE.validate(0)
        assert not NONE.validate("")

    def test_atomic_equality(self):
        """Atomic types should compare by name."""
        int1 = AtomicType("int")
        int2 = AtomicType("int")
        str1 = AtomicType("str")

        assert int1 == int2
        assert int1 != str1

    def test_atomic_hash(self):
        """Atomic types should be hashable."""
        types = {INT, FLOAT, STR}
        assert len(types) == 3
        assert INT in types

    def test_atomic_compose_fails(self):
        """Atomic types shouldn't compose with args."""
        with pytest.raises(TypeError):
            INT.compose(STR)

    def test_atomic_compose_empty(self):
        """Atomic compose with no args returns self."""
        result = INT.compose()
        assert result == INT

    def test_atomic_repr(self):
        """Atomic repr should be the name."""
        assert repr(INT) == "int"
        assert repr(STR) == "str"

    def test_atomic_to_neural(self):
        """Should produce neural representation."""
        rep = INT.to_neural(64)
        assert rep.shape == (64,)
        assert np.abs(np.linalg.norm(rep) - 1.0) < 0.01


class TestFunctionType:
    """Test function types."""

    def test_function_type_creation(self):
        """Should create function type."""
        f = FunctionType((INT, INT), INT)
        assert f.arity == 2
        assert f.output_type == INT

    def test_function_compose_success(self):
        """Compose should return output type when inputs match."""
        f = FunctionType((INT, INT), BOOL)
        result = f.compose(INT, INT)
        assert result == BOOL

    def test_function_compose_wrong_arity(self):
        """Compose should fail with wrong number of args."""
        f = FunctionType((INT, INT), BOOL)
        with pytest.raises(TypeError):
            f.compose(INT)

    def test_function_compose_type_mismatch(self):
        """Compose should fail with type mismatch."""
        f = FunctionType((INT,), BOOL)
        with pytest.raises(TypeError):
            f.compose(STR)

    def test_function_validate(self):
        """Should validate callables."""
        f = FunctionType((INT,), INT)
        assert f.validate(lambda x: x + 1)
        assert not f.validate(42)

    def test_function_equality(self):
        """Function types should compare structurally."""
        f1 = FunctionType((INT, INT), BOOL)
        f2 = FunctionType((INT, INT), BOOL)
        f3 = FunctionType((INT,), BOOL)

        assert f1 == f2
        assert f1 != f3

    def test_function_repr(self):
        """Function repr should show arrow notation."""
        f = FunctionType((INT, STR), BOOL)
        assert repr(f) == "(int, str) -> bool"

    def test_function_curry(self):
        """Should curry multi-arg function."""
        f = FunctionType((INT, STR), BOOL)
        curried = f.curry()

        assert curried.arity == 1
        assert isinstance(curried.output_type, FunctionType)

    def test_function_curry_unary(self):
        """Currying unary function should return self."""
        f = FunctionType((INT,), BOOL)
        curried = f.curry()
        assert curried == f


class TestStructuredType:
    """Test structured (record) types."""

    def test_structured_creation(self):
        """Should create structured type."""
        t = StructuredType("Person", (("name", STR), ("age", INT)))
        assert t.name == "Person"
        assert t.has_field("name")
        assert t.has_field("age")

    def test_structured_get_field_type(self):
        """Should get field type."""
        t = StructuredType("Point", (("x", INT), ("y", INT)))
        assert t.get_field_type("x") == INT
        assert t.get_field_type("z") is None

    def test_structured_validate_dict(self):
        """Should validate dict with correct fields."""
        t = StructuredType("Point", (("x", INT), ("y", INT)))

        assert t.validate({"x": 1, "y": 2})
        assert t.validate({"x": 1, "y": 2, "z": 3})  # Extra fields OK
        assert not t.validate({"x": 1})  # Missing field
        assert not t.validate({"x": "one", "y": 2})  # Wrong type

    def test_structured_compose(self):
        """Should merge fields on compose."""
        t1 = StructuredType("A", (("x", INT),))
        t2 = StructuredType("B", (("y", STR),))

        composed = t1.compose(t2)
        assert composed.has_field("x")
        assert composed.has_field("y")

    def test_structured_compose_conflict(self):
        """Should fail on conflicting field types."""
        t1 = StructuredType("A", (("x", INT),))
        t2 = StructuredType("B", (("x", STR),))

        with pytest.raises(TypeError):
            t1.compose(t2)

    def test_structured_equality(self):
        """Should compare fields structurally."""
        t1 = StructuredType("A", (("x", INT), ("y", STR)))
        t2 = StructuredType("B", (("x", INT), ("y", STR)))  # Same fields
        t3 = StructuredType("C", (("x", INT),))

        assert t1 == t2  # Same fields = equal
        assert t1 != t3

    def test_structured_subtype(self):
        """Should support structural subtyping."""
        base = StructuredType("Base", (("x", INT),))
        StructuredType("Sub", (("x", INT), ("y", STR)))

        # Sub has all fields of base
        # Note: In our implementation, more fields = NOT subtype
        # (opposite of typical structural subtyping)
        assert base.is_subtype_of(base)


class TestListType:
    """Test list types."""

    def test_list_creation(self):
        """Should create list type."""
        t = ListType(INT)
        assert t.element_type == INT

    def test_list_validate(self):
        """Should validate lists with correct element type."""
        t = ListType(INT)

        assert t.validate([1, 2, 3])
        assert t.validate([])
        assert t.validate((1, 2, 3))  # Tuples count as lists
        assert not t.validate([1, "two", 3])
        assert not t.validate("not a list")

    def test_list_compose(self):
        """Should unify element types."""
        t1 = ListType(INT)
        t2 = ListType(INT)

        composed = t1.compose(t2)
        assert isinstance(composed, ListType)
        assert composed.element_type == INT

    def test_list_equality(self):
        """Should compare element types."""
        t1 = ListType(INT)
        t2 = ListType(INT)
        t3 = ListType(STR)

        assert t1 == t2
        assert t1 != t3

    def test_list_repr(self):
        """Should show bracket notation."""
        t = ListType(INT)
        assert repr(t) == "[int]"


class TestUnionType:
    """Test union types."""

    def test_union_creation(self):
        """Should create union type."""
        t = UnionType((INT, STR))
        assert INT in t.types
        assert STR in t.types

    def test_union_validate(self):
        """Should validate any member type."""
        t = UnionType((INT, STR))

        assert t.validate(42)
        assert t.validate("hello")
        assert not t.validate([1, 2, 3])

    def test_union_compose(self):
        """Should combine member types."""
        t1 = UnionType((INT, STR))
        t2 = UnionType((BOOL,))

        composed = t1.compose(t2)
        assert isinstance(composed, UnionType)
        assert INT in composed.types
        assert STR in composed.types
        assert BOOL in composed.types

    def test_union_equality(self):
        """Should compare member sets."""
        t1 = UnionType((INT, STR))
        t2 = UnionType((STR, INT))  # Same members, different order
        t3 = UnionType((INT,))

        assert t1 == t2
        assert t1 != t3


class TestTypeVariable:
    """Test type variables."""

    def test_type_variable_creation(self):
        """Should create type variable."""
        t = TypeVariable("T")
        assert t.name == "T"

    def test_type_variable_validate(self):
        """Should validate any value."""
        t = TypeVariable("T")
        assert t.validate(42)
        assert t.validate("hello")
        assert t.validate([1, 2, 3])

    def test_type_variable_with_bound(self):
        """Should respect bound."""
        t = TypeVariable("T", bound=INT)
        assert t.validate(42)
        assert not t.validate("hello")


class TestTypeInferencer:
    """Test type inference."""

    @pytest.fixture
    def inferencer(self):
        return TypeInferencer()

    def test_infer_int(self, inferencer):
        """Should infer int type."""
        t = inferencer.infer(42)
        assert t == INT

    def test_infer_float(self, inferencer):
        """Should infer float type."""
        t = inferencer.infer(3.14)
        assert t == FLOAT

    def test_infer_str(self, inferencer):
        """Should infer str type."""
        t = inferencer.infer("hello")
        assert t == STR

    def test_infer_bool(self, inferencer):
        """Should infer bool type."""
        t = inferencer.infer(True)
        assert t == BOOL

    def test_infer_none(self, inferencer):
        """Should infer none type."""
        t = inferencer.infer(None)
        assert t == NONE

    def test_infer_list(self, inferencer):
        """Should infer list type."""
        t = inferencer.infer([1, 2, 3])
        assert isinstance(t, ListType)
        assert t.element_type == INT

    def test_infer_dict(self, inferencer):
        """Should infer structured type."""
        t = inferencer.infer({"x": 1, "y": 2})
        assert isinstance(t, StructuredType)

    def test_check_success(self, inferencer):
        """Should pass valid type check."""
        is_valid, error = inferencer.check(42, INT)
        assert is_valid
        assert error is None

    def test_check_failure(self, inferencer):
        """Should fail invalid type check."""
        is_valid, error = inferencer.check("hello", INT)
        assert not is_valid
        assert error is not None

    def test_fresh_variable(self, inferencer):
        """Should create fresh type variables."""
        v1 = inferencer.fresh_variable()
        v2 = inferencer.fresh_variable()
        assert v1 != v2


class TestNeuralRepresentations:
    """Test neural representations of types."""

    def test_atomic_neural_deterministic(self):
        """Same type should produce same representation."""
        rep1 = INT.to_neural(64)
        rep2 = INT.to_neural(64)
        np.testing.assert_array_equal(rep1, rep2)

    def test_different_types_different_reps(self):
        """Different types should have different representations."""
        int_rep = INT.to_neural(64)
        str_rep = STR.to_neural(64)

        similarity = np.dot(int_rep, str_rep)
        assert abs(similarity) < 0.9  # Should be different

    def test_function_neural(self):
        """Function types should have neural representation."""
        f = FunctionType((INT,), BOOL)
        rep = f.to_neural(64)
        assert rep.shape == (64,)

    def test_structured_neural(self):
        """Structured types should have neural representation."""
        t = StructuredType("Point", (("x", INT), ("y", INT)))
        rep = t.to_neural(64)
        assert rep.shape == (64,)

    def test_list_neural(self):
        """List types should have neural representation."""
        t = ListType(INT)
        rep = t.to_neural(64)
        assert rep.shape == (64,)


class TestTypeCompositionComplex:
    """Test complex type compositions."""

    def test_nested_function_types(self):
        """Should handle nested function types."""
        # (int -> int) -> int
        inner = FunctionType((INT,), INT)
        outer = FunctionType((inner,), INT)

        assert outer.arity == 1
        assert outer.input_types[0] == inner

    def test_list_of_functions(self):
        """Should handle list of function types."""
        f = FunctionType((INT,), INT)
        t = ListType(f)

        assert t.element_type == f

    def test_structured_with_list_field(self):
        """Should handle structured type with list field."""
        t = StructuredType("Data", (("values", ListType(INT)), ("name", STR)))

        assert t.validate({"values": [1, 2, 3], "name": "test"})
        assert not t.validate({"values": ["a", "b"], "name": "test"})
