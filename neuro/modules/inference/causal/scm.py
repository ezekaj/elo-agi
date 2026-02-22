"""
Structural Causal Models: Pearl's SCM framework.

Implements structural equation models for causal reasoning.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
from enum import Enum
import numpy as np


class VariableType(Enum):
    """Types of causal variables."""

    EXOGENOUS = "exogenous"  # External/noise variables (U)
    ENDOGENOUS = "endogenous"  # Determined by equations (V)


@dataclass
class CausalVariable:
    """A variable in a structural causal model."""

    name: str
    var_type: VariableType
    domain: Optional[List[Any]] = None  # Discrete values, or None for continuous
    description: Optional[str] = None

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, CausalVariable):
            return self.name == other.name
        return False


@dataclass
class StructuralEquation:
    """
    A structural equation: V_i = f_i(PA_i, U_i)

    Represents how a variable is determined by its parents
    and noise term.
    """

    variable: str
    parents: List[str]
    noise_var: Optional[str] = None  # Associated exogenous variable

    # The structural function
    # Takes dict of parent values and noise value
    # Returns the variable's value
    function: Optional[Callable[[Dict[str, Any], Any], Any]] = None

    # For discrete variables: probability table
    prob_table: Optional[Dict[Tuple, Dict[Any, float]]] = None

    # For linear models: coefficients
    coefficients: Optional[Dict[str, float]] = None
    intercept: float = 0.0

    def evaluate(
        self,
        parent_values: Dict[str, Any],
        noise: Any = None,
    ) -> Any:
        """Evaluate the structural equation."""
        if self.function is not None:
            return self.function(parent_values, noise)

        if self.coefficients is not None:
            # Linear equation: V = intercept + sum(coef * parent) + noise
            result = self.intercept
            for parent, coef in self.coefficients.items():
                if parent in parent_values:
                    result += coef * parent_values[parent]
            if noise is not None:
                result += noise
            return result

        if self.prob_table is not None:
            # Sample from probability table
            key = tuple(parent_values.get(p) for p in self.parents)
            dist = self.prob_table.get(key, {})
            if dist:
                values = list(dist.keys())
                probs = [dist[v] for v in values]
                return np.random.choice(values, p=probs)

        return None


class StructuralCausalModel:
    """
    Pearl's Structural Causal Model.

    Components:
    - U: Exogenous variables (external noise)
    - V: Endogenous variables (determined by model)
    - F: Structural equations V_i = f_i(PA_i, U_i)

    Supports:
    - Observational queries P(Y)
    - Interventional queries P(Y | do(X=x))
    - Counterfactual queries P(Y_x | X=x', Y=y')
    """

    def __init__(self, name: str = "scm"):
        self.name = name

        # Variables
        self._exogenous: Dict[str, CausalVariable] = {}
        self._endogenous: Dict[str, CausalVariable] = {}

        # Structural equations
        self._equations: Dict[str, StructuralEquation] = {}

        # Graph structure
        self._parents: Dict[str, List[str]] = {}
        self._children: Dict[str, List[str]] = {}

        # Exogenous distributions
        self._noise_distributions: Dict[str, Callable[[], Any]] = {}

    def add_exogenous(
        self,
        name: str,
        distribution: Optional[Callable[[], Any]] = None,
        domain: Optional[List[Any]] = None,
    ) -> CausalVariable:
        """Add an exogenous (noise) variable."""
        var = CausalVariable(name, VariableType.EXOGENOUS, domain)
        self._exogenous[name] = var

        # Default to standard normal
        if distribution is None:

            def distribution():
                return np.random.normal(0, 1)

        self._noise_distributions[name] = distribution

        return var

    def add_endogenous(
        self,
        name: str,
        domain: Optional[List[Any]] = None,
    ) -> CausalVariable:
        """Add an endogenous (internal) variable."""
        var = CausalVariable(name, VariableType.ENDOGENOUS, domain)
        self._endogenous[name] = var
        self._parents[name] = []
        self._children[name] = []
        return var

    def add_equation(
        self,
        variable: str,
        parents: List[str],
        equation: StructuralEquation,
    ) -> None:
        """Add a structural equation."""
        # Ensure variable exists
        if variable not in self._endogenous:
            self.add_endogenous(variable)

        equation.variable = variable
        equation.parents = parents
        self._equations[variable] = equation
        self._parents[variable] = parents

        # Update children
        for parent in parents:
            if parent not in self._children:
                self._children[parent] = []
            self._children[parent].append(variable)

    def add_linear_equation(
        self,
        variable: str,
        parents: List[str],
        coefficients: Dict[str, float],
        intercept: float = 0.0,
        noise_var: Optional[str] = None,
    ) -> StructuralEquation:
        """Add a linear structural equation."""
        if noise_var is None:
            noise_var = f"U_{variable}"
            self.add_exogenous(noise_var)

        eq = StructuralEquation(
            variable=variable,
            parents=parents,
            noise_var=noise_var,
            coefficients=coefficients,
            intercept=intercept,
        )
        self.add_equation(variable, parents, eq)
        return eq

    def add_discrete_equation(
        self,
        variable: str,
        parents: List[str],
        prob_table: Dict[Tuple, Dict[Any, float]],
    ) -> StructuralEquation:
        """Add a discrete structural equation with probability table."""
        eq = StructuralEquation(
            variable=variable,
            parents=parents,
            prob_table=prob_table,
        )
        self.add_equation(variable, parents, eq)
        return eq

    def get_parents(self, variable: str) -> List[str]:
        """Get causal parents of a variable."""
        return self._parents.get(variable, [])

    def get_children(self, variable: str) -> List[str]:
        """Get causal children of a variable."""
        return self._children.get(variable, [])

    def get_ancestors(self, variable: str) -> Set[str]:
        """Get all causal ancestors."""
        ancestors = set()
        queue = list(self._parents.get(variable, []))

        while queue:
            current = queue.pop(0)
            if current not in ancestors and current in self._endogenous:
                ancestors.add(current)
                queue.extend(self._parents.get(current, []))

        return ancestors

    def get_descendants(self, variable: str) -> Set[str]:
        """Get all causal descendants."""
        descendants = set()
        queue = list(self._children.get(variable, []))

        while queue:
            current = queue.pop(0)
            if current not in descendants:
                descendants.add(current)
                queue.extend(self._children.get(current, []))

        return descendants

    def sample_noise(self) -> Dict[str, Any]:
        """Sample all exogenous variables."""
        noise = {}
        for name, dist in self._noise_distributions.items():
            noise[name] = dist()
        return noise

    def compute_endogenous(
        self,
        noise: Dict[str, Any],
        interventions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Compute endogenous variables from noise.

        Uses topological order to evaluate equations.
        """
        interventions = interventions or {}
        values = {}

        # Get topological order
        order = self._topological_order()

        for var in order:
            if var in interventions:
                # Intervention overrides structural equation
                values[var] = interventions[var]
            else:
                eq = self._equations.get(var)
                if eq:
                    parent_values = {p: values.get(p) for p in eq.parents}
                    noise_val = noise.get(eq.noise_var) if eq.noise_var else None
                    values[var] = eq.evaluate(parent_values, noise_val)
                else:
                    values[var] = None

        return values

    def _topological_order(self) -> List[str]:
        """Get endogenous variables in topological order."""
        in_degree = {v: len(self._parents.get(v, [])) for v in self._endogenous}
        queue = [v for v, d in in_degree.items() if d == 0]
        order = []

        while queue:
            node = queue.pop(0)
            order.append(node)

            for child in self._children.get(node, []):
                if child in in_degree:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)

        return order

    def sample(
        self,
        n_samples: int = 1,
        interventions: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Sample from the (possibly intervened) model.

        Args:
            n_samples: Number of samples
            interventions: Dict of variable -> value for do(X=x)
        """
        samples = []

        for _ in range(n_samples):
            noise = self.sample_noise()
            values = self.compute_endogenous(noise, interventions)
            samples.append(values)

        return samples

    def observe(
        self,
        variable: str,
        n_samples: int = 1000,
    ) -> List[Any]:
        """Get observational samples of a variable."""
        samples = self.sample(n_samples)
        return [s.get(variable) for s in samples]

    def intervene(
        self,
        interventions: Dict[str, Any],
        query_var: str,
        n_samples: int = 1000,
    ) -> List[Any]:
        """
        Perform intervention do(X=x) and observe Y.

        This is P(Y | do(X=x)).
        """
        samples = self.sample(n_samples, interventions)
        return [s.get(query_var) for s in samples]

    def is_ancestor(self, var1: str, var2: str) -> bool:
        """Check if var1 is an ancestor of var2."""
        return var1 in self.get_ancestors(var2)

    def causal_effect(
        self,
        treatment: str,
        outcome: str,
        treatment_values: Tuple[Any, Any] = (0, 1),
        n_samples: int = 1000,
    ) -> float:
        """
        Estimate average causal effect of treatment on outcome.

        ACE = E[Y | do(X=1)] - E[Y | do(X=0)]
        """
        x0, x1 = treatment_values

        # Sample under intervention
        y0_samples = self.intervene({treatment: x0}, outcome, n_samples)
        y1_samples = self.intervene({treatment: x1}, outcome, n_samples)

        # Filter None values
        y0_samples = [y for y in y0_samples if y is not None]
        y1_samples = [y for y in y1_samples if y is not None]

        if not y0_samples or not y1_samples:
            return 0.0

        e_y0 = np.mean(y0_samples)
        e_y1 = np.mean(y1_samples)

        return float(e_y1 - e_y0)

    def get_causal_graph(self) -> Dict[str, List[str]]:
        """Get the causal graph as adjacency list."""
        return {v: list(self._children.get(v, [])) for v in self._endogenous}

    def statistics(self) -> Dict[str, Any]:
        """Get model statistics."""
        return {
            "n_exogenous": len(self._exogenous),
            "n_endogenous": len(self._endogenous),
            "n_equations": len(self._equations),
            "name": self.name,
        }
