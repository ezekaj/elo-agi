"""
Reasoning Benchmarks: Tests for abstract and logical reasoning.

Includes:
- Pattern completion (ARC-style)
- Analogy solving
- Logical inference
- Rule learning
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from .base_benchmark import Benchmark, BenchmarkConfig


@dataclass
class ReasoningConfig(BenchmarkConfig):
    """Configuration for reasoning benchmarks."""

    grid_size: int = 5
    max_patterns: int = 3
    include_transforms: List[str] = field(
        default_factory=lambda: ["rotate", "flip", "scale", "translate"]
    )


class ReasoningBenchmark(Benchmark):
    """Base class for reasoning benchmarks."""

    @property
    def name(self) -> str:
        return "reasoning"


class PatternCompletion(ReasoningBenchmark):
    """
    Pattern completion benchmark (ARC-style).

    Given input-output pairs, predict the output for a new input.
    Tests abstract reasoning and rule induction.
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        if config is None:
            self.reasoning_config = ReasoningConfig(name="pattern_completion")
        elif isinstance(config, ReasoningConfig):
            self.reasoning_config = config
        else:
            # Wrap generic BenchmarkConfig in ReasoningConfig
            self.reasoning_config = ReasoningConfig(
                name=config.name,
                n_trials=config.n_trials,
                difficulty=config.difficulty,
                random_seed=config.random_seed,
            )
        super().__init__(self.reasoning_config)

    @property
    def name(self) -> str:
        return "pattern_completion"

    def generate_trial(self, trial_id: int) -> Tuple[Dict[str, Any], np.ndarray]:
        """Generate a pattern completion trial."""
        grid_size = self.reasoning_config.grid_size

        # Choose a transformation rule
        rule = self._rng.choice(["fill", "extend", "mirror", "color_map"])

        # Generate examples
        n_examples = self._rng.integers(2, 4)
        examples = []

        for _ in range(n_examples):
            input_grid = self._generate_grid(grid_size)
            output_grid = self._apply_rule(input_grid, rule)
            examples.append({"input": input_grid, "output": output_grid})

        # Generate test case
        test_input = self._generate_grid(grid_size)
        expected_output = self._apply_rule(test_input, rule)

        trial_data = {
            "examples": examples,
            "test_input": test_input,
            "rule": rule,  # For debugging
        }

        return trial_data, expected_output

    def evaluate(self, expected: np.ndarray, actual: Any) -> Tuple[bool, float]:
        """Evaluate pattern completion."""
        if actual is None:
            return False, 0.0

        try:
            actual = np.asarray(actual)

            if actual.shape != expected.shape:
                return False, 0.0

            # Exact match
            if np.array_equal(actual, expected):
                return True, 1.0

            # Partial credit for close matches
            matches = np.sum(actual == expected)
            total = expected.size
            score = matches / total

            return score > 0.9, score

        except Exception:
            return False, 0.0

    def _generate_grid(self, size: int) -> np.ndarray:
        """Generate a random grid pattern."""
        # Simple patterns: shapes, lines, etc.
        grid = np.zeros((size, size), dtype=np.int32)

        pattern_type = self._rng.choice(["square", "line", "diagonal", "scatter"])

        if pattern_type == "square":
            # Random filled square
            x = self._rng.integers(0, size - 2)
            y = self._rng.integers(0, size - 2)
            w = self._rng.integers(1, min(3, size - x))
            h = self._rng.integers(1, min(3, size - y))
            color = self._rng.integers(1, 5)
            grid[y : y + h, x : x + w] = color

        elif pattern_type == "line":
            # Horizontal or vertical line
            if self._rng.random() > 0.5:
                row = self._rng.integers(0, size)
                color = self._rng.integers(1, 5)
                grid[row, :] = color
            else:
                col = self._rng.integers(0, size)
                color = self._rng.integers(1, 5)
                grid[:, col] = color

        elif pattern_type == "diagonal":
            color = self._rng.integers(1, 5)
            for i in range(size):
                grid[i, i] = color

        else:  # scatter
            n_points = self._rng.integers(3, 8)
            for _ in range(n_points):
                x = self._rng.integers(0, size)
                y = self._rng.integers(0, size)
                color = self._rng.integers(1, 5)
                grid[y, x] = color

        return grid

    def _apply_rule(self, grid: np.ndarray, rule: str) -> np.ndarray:
        """Apply a transformation rule to the grid."""
        if rule == "fill":
            # Fill background with most common non-zero color
            non_zero = grid[grid > 0]
            if len(non_zero) > 0:
                fill_color = int(np.bincount(non_zero).argmax())
                result = grid.copy()
                result[result == 0] = fill_color
                return result
            return grid.copy()

        elif rule == "extend":
            # Extend patterns to edges
            result = grid.copy()
            for i in range(grid.shape[0]):
                row = grid[i, :]
                non_zero = np.where(row > 0)[0]
                if len(non_zero) > 0:
                    color = row[non_zero[0]]
                    result[i, :] = color
            return result

        elif rule == "mirror":
            # Mirror horizontally
            return np.fliplr(grid)

        elif rule == "color_map":
            # Map colors: 1->2, 2->3, etc.
            result = grid.copy()
            result[result > 0] = (result[result > 0] % 4) + 1
            return result

        return grid.copy()


class AnalogySolving(ReasoningBenchmark):
    """
    Analogy solving benchmark.

    A is to B as C is to ? (A:B::C:?)
    Tests relational reasoning.
    """

    def __init__(self, config: Optional[ReasoningConfig] = None):
        self.reasoning_config = config or ReasoningConfig(name="analogy_solving")
        super().__init__(self.reasoning_config)

    @property
    def name(self) -> str:
        return "analogy_solving"

    def generate_trial(self, trial_id: int) -> Tuple[Dict[str, Any], np.ndarray]:
        """Generate an analogy trial."""
        # Generate A and transformation to get B
        size = self.reasoning_config.grid_size
        A = self._generate_shape(size)

        # Choose transformation
        transform = self._rng.choice(["rotate90", "double", "invert", "shift"])
        B = self._apply_transform(A, transform)

        # Generate C
        C = self._generate_shape(size)

        # Apply same transformation
        D = self._apply_transform(C, transform)

        trial_data = {
            "A": A,
            "B": B,
            "C": C,
            "transform": transform,
        }

        return trial_data, D

    def evaluate(self, expected: np.ndarray, actual: Any) -> Tuple[bool, float]:
        """Evaluate analogy solution."""
        if actual is None:
            return False, 0.0

        try:
            actual = np.asarray(actual)

            if actual.shape != expected.shape:
                return False, 0.0

            if np.array_equal(actual, expected):
                return True, 1.0

            # Partial credit
            matches = np.sum(actual == expected)
            score = matches / expected.size
            return score > 0.8, score

        except Exception:
            return False, 0.0

    def _generate_shape(self, size: int) -> np.ndarray:
        """Generate a simple shape."""
        grid = np.zeros((size, size), dtype=np.int32)

        shape = self._rng.choice(["cross", "L", "T", "square"])
        color = self._rng.integers(1, 5)
        center = size // 2

        if shape == "cross":
            grid[center, :] = color
            grid[:, center] = color

        elif shape == "L":
            grid[center:, center] = color
            grid[-1, center:] = color

        elif shape == "T":
            grid[0, :] = color
            grid[:, center] = color

        elif shape == "square":
            grid[1:-1, 1:-1] = color

        return grid

    def _apply_transform(self, grid: np.ndarray, transform: str) -> np.ndarray:
        """Apply transformation to grid."""
        if transform == "rotate90":
            return np.rot90(grid)

        elif transform == "double":
            # Double the size (simple upscale)
            result = np.zeros((grid.shape[0], grid.shape[1]), dtype=np.int32)
            for i in range(grid.shape[0] // 2):
                for j in range(grid.shape[1] // 2):
                    val = grid[i * 2, j * 2]
                    result[i, j] = val
            return result

        elif transform == "invert":
            result = grid.copy()
            max_val = grid.max()
            result[result > 0] = max_val - result[result > 0] + 1
            return result

        elif transform == "shift":
            return np.roll(grid, 1, axis=1)

        return grid.copy()


class LogicalInference(ReasoningBenchmark):
    """
    Logical inference benchmark.

    Given premises, determine if conclusion follows.
    Tests deductive reasoning.
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        config = config or BenchmarkConfig(name="logical_inference")
        super().__init__(config)

    @property
    def name(self) -> str:
        return "logical_inference"

    def generate_trial(self, trial_id: int) -> Tuple[Dict[str, Any], bool]:
        """Generate a logical inference trial."""
        # Simple propositional logic
        problem_type = self._rng.choice(["modus_ponens", "modus_tollens", "syllogism", "invalid"])

        if problem_type == "modus_ponens":
            # If P then Q. P. Therefore Q.
            trial_data = {
                "premises": [
                    "If it rains, the ground is wet.",
                    "It is raining.",
                ],
                "conclusion": "The ground is wet.",
                "type": problem_type,
            }
            expected = True

        elif problem_type == "modus_tollens":
            # If P then Q. Not Q. Therefore not P.
            trial_data = {
                "premises": [
                    "If it rains, the ground is wet.",
                    "The ground is not wet.",
                ],
                "conclusion": "It is not raining.",
                "type": problem_type,
            }
            expected = True

        elif problem_type == "syllogism":
            # All A are B. All B are C. Therefore all A are C.
            trial_data = {
                "premises": [
                    "All dogs are mammals.",
                    "All mammals are animals.",
                ],
                "conclusion": "All dogs are animals.",
                "type": problem_type,
            }
            expected = True

        else:  # invalid
            # Invalid inference
            trial_data = {
                "premises": [
                    "If it rains, the ground is wet.",
                    "The ground is wet.",
                ],
                "conclusion": "It is raining.",  # Affirming the consequent (invalid)
                "type": problem_type,
            }
            expected = False

        return trial_data, expected

    def evaluate(self, expected: bool, actual: Any) -> Tuple[bool, float]:
        """Evaluate logical inference."""
        if actual is None:
            return False, 0.0

        try:
            # Convert to boolean
            if isinstance(actual, (bool, np.bool_)):
                actual_bool = bool(actual)
            elif isinstance(actual, str):
                actual_bool = actual.lower() in ["true", "yes", "valid", "1"]
            elif isinstance(actual, (int, float)):
                actual_bool = actual > 0.5
            else:
                actual_bool = bool(actual)

            correct = actual_bool == expected
            return correct, 1.0 if correct else 0.0

        except Exception:
            return False, 0.0
