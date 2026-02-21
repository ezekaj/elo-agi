"""
Causal Reasoning Benchmark: Tests for counterfactual and interventional reasoning.

Evaluates:
- Counterfactual accuracy (Pearl's ladder of causation)
- Intervention prediction
- Causal discovery from observational data
- Nested counterfactual reasoning
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from .base_benchmark import Benchmark, BenchmarkConfig, TrialResult


@dataclass
class CausalGraph:
    """Simple causal graph representation."""
    variables: List[str]
    edges: List[Tuple[str, str]]  # (cause, effect) pairs
    mechanisms: Dict[str, Any] = field(default_factory=dict)

    def parents(self, var: str) -> List[str]:
        """Get parents of a variable."""
        return [cause for cause, effect in self.edges if effect == var]

    def children(self, var: str) -> List[str]:
        """Get children of a variable."""
        return [effect for cause, effect in self.edges if cause == var]


@dataclass
class CausalQuery:
    """A causal reasoning query."""
    query_type: str  # "counterfactual", "intervention", "association"
    graph: CausalGraph
    observation: Dict[str, float]  # Observed values
    intervention: Optional[Dict[str, float]] = None  # do(X=x)
    target_variable: str = ""
    expected_answer: Optional[float] = None


class CounterfactualBenchmark(Benchmark):
    """
    Benchmark for counterfactual reasoning.

    Tests agent's ability to answer "What would Y have been if X had been x?"
    given that we observed X=x' and Y=y'.
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        config = config or BenchmarkConfig(
            name="counterfactual",
            description="Counterfactual reasoning benchmark",
            n_trials=50,
        )
        super().__init__(config)

    @property
    def name(self) -> str:
        return "counterfactual"

    def generate_trial(self, trial_id: int) -> Tuple[CausalQuery, float]:
        """Generate a counterfactual reasoning trial."""
        # Simple 3-variable chain: X -> Y -> Z
        graph = CausalGraph(
            variables=["X", "Y", "Z"],
            edges=[("X", "Y"), ("Y", "Z")],
        )

        # Generate random coefficients for linear mechanisms
        alpha = self._rng.uniform(0.5, 2.0)
        beta = self._rng.uniform(0.5, 2.0)

        # Generate observed values
        x_obs = self._rng.uniform(-2, 2)
        noise_y = self._rng.normal(0, 0.1)
        noise_z = self._rng.normal(0, 0.1)
        y_obs = alpha * x_obs + noise_y
        z_obs = beta * y_obs + noise_z

        # Counterfactual: What would Z have been if X had been x_cf?
        x_cf = self._rng.uniform(-2, 2)

        # Compute ground truth counterfactual
        # Z_cf = beta * (alpha * x_cf + noise_y) + noise_z
        y_cf = alpha * x_cf + noise_y
        z_cf = beta * y_cf + noise_z

        query = CausalQuery(
            query_type="counterfactual",
            graph=graph,
            observation={"X": x_obs, "Y": y_obs, "Z": z_obs},
            intervention={"X": x_cf},
            target_variable="Z",
            expected_answer=z_cf,
        )

        # Store coefficients for evaluation
        query.graph.mechanisms = {"alpha": alpha, "beta": beta}

        return query, z_cf

    def evaluate(self, expected: float, actual: Any) -> Tuple[bool, float]:
        """Evaluate counterfactual prediction."""
        if actual is None:
            return False, 0.0

        try:
            actual_val = float(actual)
        except (ValueError, TypeError):
            return False, 0.0

        # Score based on relative error
        if abs(expected) < 0.01:
            error = abs(actual_val - expected)
        else:
            error = abs(actual_val - expected) / abs(expected)

        # Convert error to score (lower error = higher score)
        score = max(0.0, 1.0 - error)

        # Success if within 10% relative error
        success = error < 0.1

        return success, score


class InterventionBenchmark(Benchmark):
    """
    Benchmark for intervention prediction.

    Tests agent's ability to predict P(Y | do(X=x)).
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        config = config or BenchmarkConfig(
            name="intervention",
            description="Causal intervention prediction benchmark",
            n_trials=50,
        )
        super().__init__(config)

    @property
    def name(self) -> str:
        return "intervention"

    def generate_trial(self, trial_id: int) -> Tuple[CausalQuery, float]:
        """Generate an intervention prediction trial."""
        # Confounded structure: X <- U -> Y, X -> Y
        # Intervention removes X <- U, keeping only X -> Y
        graph = CausalGraph(
            variables=["X", "Y", "U"],
            edges=[("U", "X"), ("U", "Y"), ("X", "Y")],
        )

        # Coefficients
        gamma_ux = self._rng.uniform(0.5, 1.5)
        gamma_uy = self._rng.uniform(0.5, 1.5)
        gamma_xy = self._rng.uniform(0.5, 1.5)

        # Intervention value
        x_do = self._rng.uniform(-2, 2)

        # Under intervention do(X=x_do):
        # Y = gamma_xy * x_do + gamma_uy * E[U] + noise
        # E[U] = 0 (standard normal)
        # E[Y | do(X=x_do)] = gamma_xy * x_do
        expected_y = gamma_xy * x_do

        query = CausalQuery(
            query_type="intervention",
            graph=graph,
            observation={},
            intervention={"X": x_do},
            target_variable="Y",
            expected_answer=expected_y,
        )

        query.graph.mechanisms = {
            "gamma_ux": gamma_ux,
            "gamma_uy": gamma_uy,
            "gamma_xy": gamma_xy,
        }

        return query, expected_y

    def evaluate(self, expected: float, actual: Any) -> Tuple[bool, float]:
        """Evaluate intervention prediction."""
        if actual is None:
            return False, 0.0

        try:
            actual_val = float(actual)
        except (ValueError, TypeError):
            return False, 0.0

        if abs(expected) < 0.01:
            error = abs(actual_val - expected)
        else:
            error = abs(actual_val - expected) / abs(expected)

        score = max(0.0, 1.0 - error)
        success = error < 0.15

        return success, score


class CausalDiscoveryBenchmark(Benchmark):
    """
    Benchmark for causal structure discovery.

    Tests agent's ability to infer causal structure from data.
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        config = config or BenchmarkConfig(
            name="causal_discovery",
            description="Causal structure discovery benchmark",
            n_trials=30,
        )
        super().__init__(config)

    @property
    def name(self) -> str:
        return "causal_discovery"

    def generate_trial(self, trial_id: int) -> Tuple[Dict, CausalGraph]:
        """Generate a causal discovery trial."""
        # Generate a random DAG with 4 variables
        n_vars = 4
        variables = [f"V{i}" for i in range(n_vars)]

        # Random edges (ensure DAG by only allowing i -> j where i < j)
        edges = []
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if self._rng.random() < 0.4:  # 40% chance of edge
                    edges.append((variables[i], variables[j]))

        true_graph = CausalGraph(variables=variables, edges=edges)

        # Generate observational data from this structure
        n_samples = 200
        data = {var: np.zeros(n_samples) for var in variables}

        # Topological order (already sorted by construction)
        coefficients = {}
        for var in variables:
            parents = true_graph.parents(var)
            noise = self._rng.normal(0, 1, n_samples)

            if not parents:
                data[var] = noise
            else:
                data[var] = noise.copy()
                for parent in parents:
                    coef = self._rng.uniform(0.5, 2.0)
                    coefficients[(parent, var)] = coef
                    data[var] += coef * data[parent]

        trial_input = {
            "data": data,
            "variables": variables,
            "n_samples": n_samples,
        }

        return trial_input, true_graph

    def evaluate(self, expected: CausalGraph, actual: Any) -> Tuple[bool, float]:
        """Evaluate discovered causal structure."""
        if actual is None:
            return False, 0.0

        # actual should be a list of edges or a CausalGraph
        if isinstance(actual, CausalGraph):
            predicted_edges = set(actual.edges)
        elif isinstance(actual, (list, set)):
            predicted_edges = set(actual)
        else:
            return False, 0.0

        true_edges = set(expected.edges)

        if not true_edges and not predicted_edges:
            return True, 1.0

        # Compute F1 score
        true_positives = len(true_edges & predicted_edges)
        false_positives = len(predicted_edges - true_edges)
        false_negatives = len(true_edges - predicted_edges)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        # Success if F1 > 0.7
        success = f1 > 0.7

        return success, f1


class NestedCounterfactualBenchmark(Benchmark):
    """
    Benchmark for nested counterfactual reasoning.

    Tests "What would Y have been if X had been x, given that
    if X had been x', Y would have been y'?"
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        config = config or BenchmarkConfig(
            name="nested_counterfactual",
            description="Nested counterfactual reasoning benchmark",
            n_trials=30,
        )
        super().__init__(config)

    @property
    def name(self) -> str:
        return "nested_counterfactual"

    def generate_trial(self, trial_id: int) -> Tuple[Dict, float]:
        """Generate a nested counterfactual trial."""
        # Simple model: Y = alpha * X + U
        alpha = self._rng.uniform(0.5, 2.0)

        # Observed
        x_obs = self._rng.uniform(-2, 2)
        u = self._rng.normal(0, 0.5)  # Noise term
        y_obs = alpha * x_obs + u

        # Primary counterfactual: What if X = x1?
        x1 = self._rng.uniform(-2, 2)
        y1_cf = alpha * x1 + u  # Same noise

        # Nested: Given y1_cf, what if X = x2?
        x2 = self._rng.uniform(-2, 2)
        # The noise u is fixed (abducted from observation)
        y2_cf = alpha * x2 + u

        trial_input = {
            "model": {"alpha": alpha},
            "observation": {"X": x_obs, "Y": y_obs},
            "primary_intervention": {"X": x1, "Y_counterfactual": y1_cf},
            "secondary_intervention": {"X": x2},
            "target": "Y",
        }

        return trial_input, y2_cf

    def evaluate(self, expected: float, actual: Any) -> Tuple[bool, float]:
        """Evaluate nested counterfactual."""
        if actual is None:
            return False, 0.0

        try:
            actual_val = float(actual)
        except (ValueError, TypeError):
            return False, 0.0

        if abs(expected) < 0.01:
            error = abs(actual_val - expected)
        else:
            error = abs(actual_val - expected) / abs(expected)

        score = max(0.0, 1.0 - error)
        success = error < 0.1

        return success, score


def create_causal_benchmark_suite() -> List[Benchmark]:
    """Create all causal reasoning benchmarks."""
    return [
        CounterfactualBenchmark(),
        InterventionBenchmark(),
        CausalDiscoveryBenchmark(),
        NestedCounterfactualBenchmark(),
    ]


__all__ = [
    'CausalGraph',
    'CausalQuery',
    'CounterfactualBenchmark',
    'InterventionBenchmark',
    'CausalDiscoveryBenchmark',
    'NestedCounterfactualBenchmark',
    'create_causal_benchmark_suite',
]
