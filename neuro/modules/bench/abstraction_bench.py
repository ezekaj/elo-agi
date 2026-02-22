"""
Abstraction Benchmark: Tests for compositional generalization and symbol binding.

Evaluates:
- Symbol binding accuracy
- Compositional generalization (train/test split)
- Program synthesis success rate
- Analogy completion
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
import numpy as np

from .base_benchmark import Benchmark, BenchmarkConfig


@dataclass
class SymbolBindingTask:
    """A symbol binding task."""

    symbols: List[str]
    neural_vectors: Dict[str, np.ndarray]
    roles: Dict[str, str]  # symbol -> role (e.g., "dog" -> "agent")
    query_symbol: str
    query_role: str
    expected_vector: np.ndarray


@dataclass
class CompositionTask:
    """A compositional generalization task."""

    primitives: List[str]
    train_compositions: List[Tuple[str, ...]]
    test_composition: Tuple[str, ...]
    primitive_meanings: Dict[str, Callable]
    expected_output: Any


@dataclass
class AnalogyTask:
    """An analogy completion task (A:B :: C:?)."""

    a: str
    b: str
    c: str
    expected_d: str
    domain: str  # e.g., "semantic", "visual", "relational"


class SymbolBindingBenchmark(Benchmark):
    """
    Benchmark for neural-symbolic binding.

    Tests agent's ability to bind symbols to neural representations
    and retrieve them by role.
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        config = config or BenchmarkConfig(
            name="symbol_binding",
            description="Symbol-neural binding benchmark",
            n_trials=50,
        )
        super().__init__(config)
        self.embedding_dim = 64

    @property
    def name(self) -> str:
        return "symbol_binding"

    def generate_trial(self, trial_id: int) -> Tuple[SymbolBindingTask, np.ndarray]:
        """Generate a symbol binding trial."""
        # Create symbols with roles
        roles = ["agent", "patient", "instrument", "location"]
        symbols = [f"sym_{i}" for i in range(len(roles))]

        # Generate random embeddings
        vectors = {}
        for sym in symbols:
            vec = self._rng.standard_normal(self.embedding_dim)
            vec = vec / np.linalg.norm(vec)
            vectors[sym] = vec

        # Assign roles
        role_assignment = dict(zip(symbols, roles))

        # Query: retrieve the vector for a random symbol by its role
        query_idx = self._rng.integers(0, len(symbols))
        query_symbol = symbols[query_idx]
        query_role = roles[query_idx]
        expected = vectors[query_symbol]

        task = SymbolBindingTask(
            symbols=symbols,
            neural_vectors=vectors,
            roles=role_assignment,
            query_symbol=query_symbol,
            query_role=query_role,
            expected_vector=expected,
        )

        return task, expected

    def evaluate(self, expected: np.ndarray, actual: Any) -> Tuple[bool, float]:
        """Evaluate symbol binding retrieval."""
        if actual is None:
            return False, 0.0

        try:
            actual_vec = np.array(actual)
        except (ValueError, TypeError):
            return False, 0.0

        if actual_vec.shape != expected.shape:
            return False, 0.0

        # Cosine similarity
        norm_e = np.linalg.norm(expected)
        norm_a = np.linalg.norm(actual_vec)

        if norm_e < 1e-8 or norm_a < 1e-8:
            return False, 0.0

        similarity = np.dot(expected, actual_vec) / (norm_e * norm_a)
        score = (similarity + 1) / 2  # Map [-1, 1] to [0, 1]

        success = similarity > 0.9

        return success, float(score)


class CompositionalBenchmark(Benchmark):
    """
    Benchmark for compositional generalization.

    Tests agent's ability to generalize to novel compositions
    of known primitives.
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        config = config or BenchmarkConfig(
            name="compositional",
            description="Compositional generalization benchmark",
            n_trials=40,
        )
        super().__init__(config)

    @property
    def name(self) -> str:
        return "compositional"

    def generate_trial(self, trial_id: int) -> Tuple[CompositionTask, int]:
        """Generate a compositional generalization trial."""
        # Primitives: simple arithmetic operations
        primitives = ["double", "add_one", "square", "negate"]

        def double(x):
            return x * 2

        def add_one(x):
            return x + 1

        def square(x):
            return x * x

        def negate(x):
            return -x

        primitive_fns = {
            "double": double,
            "add_one": add_one,
            "square": square,
            "negate": negate,
        }

        # Training compositions (seen during training)
        train_comps = [
            ("double", "add_one"),  # double(add_one(x))
            ("square", "double"),
            ("add_one", "negate"),
            ("negate", "double"),
        ]

        # Test composition (novel)
        test_options = [
            ("add_one", "square"),
            ("double", "negate"),
            ("square", "add_one"),
            ("negate", "square"),
        ]
        test_comp = test_options[trial_id % len(test_options)]

        # Input value
        x = self._rng.integers(1, 10)

        # Compute expected output (apply in order: first, then second)
        result = x
        for op in test_comp:
            result = primitive_fns[op](result)

        task = CompositionTask(
            primitives=primitives,
            train_compositions=train_comps,
            test_composition=test_comp,
            primitive_meanings=primitive_fns,
            expected_output=result,
        )

        # Include input in task for agent
        task_with_input = {
            "primitives": primitives,
            "train_compositions": train_comps,
            "test_composition": test_comp,
            "input_value": x,
        }

        return task_with_input, result

    def evaluate(self, expected: int, actual: Any) -> Tuple[bool, float]:
        """Evaluate compositional output."""
        if actual is None:
            return False, 0.0

        try:
            actual_val = int(actual)
        except (ValueError, TypeError):
            return False, 0.0

        success = actual_val == expected
        score = 1.0 if success else 0.0

        return success, score


class ProgramSynthesisBenchmark(Benchmark):
    """
    Benchmark for program synthesis from examples.

    Tests agent's ability to synthesize programs that
    satisfy input-output examples.
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        config = config or BenchmarkConfig(
            name="program_synthesis",
            description="Program synthesis from examples benchmark",
            n_trials=30,
        )
        super().__init__(config)

    @property
    def name(self) -> str:
        return "program_synthesis"

    def generate_trial(self, trial_id: int) -> Tuple[Dict, Callable]:
        """Generate a program synthesis trial."""
        # Define target functions of increasing complexity
        target_fns = [
            lambda x: x + 1,  # increment
            lambda x: x * 2,  # double
            lambda x: x * x,  # square
            lambda x: (x * 2) + 1,  # 2x + 1
            lambda x: (x + 1) * 2,  # (x+1) * 2
            lambda x: x * x + x,  # x^2 + x
            lambda x: (x + 2) * (x + 2),  # (x+2)^2
            lambda x: x * 3 - 1,  # 3x - 1
        ]

        target_fn = target_fns[trial_id % len(target_fns)]

        # Generate input-output examples
        n_examples = 5
        examples = []
        for i in range(n_examples):
            inp = self._rng.integers(0, 10)
            out = target_fn(inp)
            examples.append((inp, out))

        # Test inputs (for verification)
        test_inputs = [self._rng.integers(0, 20) for _ in range(3)]

        task = {
            "examples": examples,
            "test_inputs": test_inputs,
            "available_ops": ["add", "subtract", "multiply", "constant"],
        }

        return task, target_fn

    def evaluate(self, expected_fn: Callable, actual: Any) -> Tuple[bool, float]:
        """Evaluate synthesized program."""
        if actual is None or not callable(actual):
            # Try to interpret as a simple function
            if isinstance(actual, (list, tuple)):
                # Maybe it's a sequence of operations?
                return False, 0.0
            return False, 0.0

        # Test on random inputs
        test_values = list(range(0, 15))
        correct = 0

        for val in test_values:
            try:
                expected_out = expected_fn(val)
                actual_out = actual(val)
                if actual_out == expected_out:
                    correct += 1
            except Exception:
                pass

        score = correct / len(test_values)
        success = score >= 0.9  # 90% accuracy

        return success, score


class AnalogyBenchmark(Benchmark):
    """
    Benchmark for analogy completion.

    Tests agent's ability to complete analogies of form A:B :: C:?
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        config = config or BenchmarkConfig(
            name="analogy",
            description="Analogy completion benchmark",
            n_trials=50,
        )
        super().__init__(config)

    @property
    def name(self) -> str:
        return "analogy"

    def generate_trial(self, trial_id: int) -> Tuple[AnalogyTask, str]:
        """Generate an analogy trial."""
        # Semantic analogies
        analogies = [
            # Opposites
            ("hot", "cold", "big", "small", "semantic"),
            ("up", "down", "left", "right", "semantic"),
            ("day", "night", "summer", "winter", "semantic"),
            # Part-whole
            ("finger", "hand", "toe", "foot", "semantic"),
            ("page", "book", "brick", "wall", "semantic"),
            # Category
            ("dog", "animal", "rose", "flower", "semantic"),
            ("car", "vehicle", "hammer", "tool", "semantic"),
            # Functional
            ("eye", "see", "ear", "hear", "semantic"),
            ("knife", "cut", "pen", "write", "semantic"),
            # Numerical (relational)
            ("2", "4", "3", "6", "numerical"),  # double
            ("5", "25", "3", "9", "numerical"),  # square
            ("10", "5", "8", "4", "numerical"),  # half
            # Sequence
            ("a", "b", "c", "d", "sequence"),
            ("1", "2", "3", "4", "sequence"),
        ]

        idx = trial_id % len(analogies)
        a, b, c, d, domain = analogies[idx]

        task = AnalogyTask(
            a=a,
            b=b,
            c=c,
            expected_d=d,
            domain=domain,
        )

        return task, d

    def evaluate(self, expected: str, actual: Any) -> Tuple[bool, float]:
        """Evaluate analogy completion."""
        if actual is None:
            return False, 0.0

        actual_str = str(actual).strip().lower()
        expected_str = expected.strip().lower()

        # Exact match
        if actual_str == expected_str:
            return True, 1.0

        # Partial credit for close answers
        # (Could use embeddings for semantic similarity)
        return False, 0.0


class AbstractionLevelBenchmark(Benchmark):
    """
    Benchmark for hierarchical abstraction.

    Tests agent's ability to operate at different levels of abstraction.
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        config = config or BenchmarkConfig(
            name="abstraction_level",
            description="Hierarchical abstraction benchmark",
            n_trials=40,
        )
        super().__init__(config)

    @property
    def name(self) -> str:
        return "abstraction_level"

    def generate_trial(self, trial_id: int) -> Tuple[Dict, str]:
        """Generate an abstraction level trial."""
        # Hierarchies: instance -> category -> supercategory
        hierarchies = [
            ("labrador", "dog", "mammal", "animal"),
            ("rose", "flower", "plant", "organism"),
            ("car", "vehicle", "machine", "artifact"),
            ("hammer", "tool", "implement", "artifact"),
            ("apple", "fruit", "food", "consumable"),
            ("oak", "tree", "plant", "organism"),
        ]

        hierarchy = hierarchies[trial_id % len(hierarchies)]

        # Query: What is X at level Y?
        query_idx = self._rng.integers(0, 3)  # 0, 1, or 2
        target_level = self._rng.integers(query_idx + 1, 4)

        item = hierarchy[query_idx]
        expected = hierarchy[target_level]

        task = {
            "item": item,
            "current_level": query_idx,
            "target_level": target_level,
            "level_names": ["instance", "category", "supercategory", "domain"],
        }

        return task, expected

    def evaluate(self, expected: str, actual: Any) -> Tuple[bool, float]:
        """Evaluate abstraction level answer."""
        if actual is None:
            return False, 0.0

        actual_str = str(actual).strip().lower()
        expected_str = expected.strip().lower()

        success = actual_str == expected_str
        score = 1.0 if success else 0.0

        return success, score


def create_abstraction_benchmark_suite() -> List[Benchmark]:
    """Create all abstraction benchmarks."""
    return [
        SymbolBindingBenchmark(),
        CompositionalBenchmark(),
        ProgramSynthesisBenchmark(),
        AnalogyBenchmark(),
        AbstractionLevelBenchmark(),
    ]


__all__ = [
    "SymbolBindingTask",
    "CompositionTask",
    "AnalogyTask",
    "SymbolBindingBenchmark",
    "CompositionalBenchmark",
    "ProgramSynthesisBenchmark",
    "AnalogyBenchmark",
    "AbstractionLevelBenchmark",
    "create_abstraction_benchmark_suite",
]
