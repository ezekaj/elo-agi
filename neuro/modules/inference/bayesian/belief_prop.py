"""
Belief Propagation: Message passing inference on factor graphs.

Implements loopy belief propagation and sum-product algorithm.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
import numpy as np
from collections import defaultdict


@dataclass
class Factor:
    """A factor in a factor graph."""

    name: str
    variables: List[str]
    # Maps variable assignments to potential values
    # Assignment is a tuple of (var_name, value) pairs
    potentials: Dict[Tuple, float] = field(default_factory=dict)

    def get_potential(self, assignment: Dict[str, Any]) -> float:
        """Get potential for an assignment."""
        key = tuple(sorted(assignment.items()))
        return self.potentials.get(key, 0.0)

    def set_potential(self, assignment: Dict[str, Any], value: float) -> None:
        """Set potential for an assignment."""
        key = tuple(sorted(assignment.items()))
        self.potentials[key] = value

    def marginalize(self, var: str, var_values: List[Any]) -> "Factor":
        """Marginalize out a variable by summing."""
        new_vars = [v for v in self.variables if v != var]

        if var not in self.variables:
            return Factor(self.name, new_vars, dict(self.potentials))

        new_potentials = {}

        for key, potential in self.potentials.items():
            assignment = dict(key)
            if var in assignment:
                # Create key without this variable
                new_assignment = {k: v for k, v in assignment.items() if k != var}
                new_key = tuple(sorted(new_assignment.items()))

                new_potentials[new_key] = new_potentials.get(new_key, 0) + potential

        return Factor(f"{self.name}_marg_{var}", new_vars, new_potentials)

    def multiply(self, other: "Factor") -> "Factor":
        """Multiply two factors."""
        all_vars = list(set(self.variables + other.variables))
        new_name = f"{self.name}_x_{other.name}"
        new_potentials = {}

        # Get all possible assignments
        for key1, pot1 in self.potentials.items():
            assignment1 = dict(key1)

            for key2, pot2 in other.potentials.items():
                assignment2 = dict(key2)

                # Check compatibility
                compatible = True
                for var in set(assignment1.keys()) & set(assignment2.keys()):
                    if assignment1[var] != assignment2[var]:
                        compatible = False
                        break

                if compatible:
                    combined = {**assignment1, **assignment2}
                    combined_key = tuple(sorted(combined.items()))
                    new_potentials[combined_key] = pot1 * pot2

        return Factor(new_name, all_vars, new_potentials)


@dataclass
class Message:
    """A message in belief propagation."""

    source: str  # Factor or variable name
    target: str  # Factor or variable name
    values: Dict[Any, float]  # Value -> probability/potential

    def normalize(self) -> "Message":
        """Return normalized message."""
        total = sum(self.values.values())
        if total > 0:
            return Message(self.source, self.target, {k: v / total for k, v in self.values.items()})
        return self

    def multiply(self, other: "Message") -> "Message":
        """Element-wise multiplication of messages."""
        result = {}
        for val in set(self.values.keys()) | set(other.values.keys()):
            result[val] = self.values.get(val, 1.0) * other.values.get(val, 1.0)
        return Message(self.source, self.target, result)


class FactorGraph:
    """
    Factor graph for belief propagation.

    A bipartite graph with variable nodes and factor nodes.
    """

    def __init__(self):
        # Variables and their possible values
        self._variables: Dict[str, List[Any]] = {}

        # Factors
        self._factors: Dict[str, Factor] = {}

        # Edges: variable -> factors, factor -> variables
        self._var_to_factors: Dict[str, List[str]] = defaultdict(list)
        self._factor_to_vars: Dict[str, List[str]] = defaultdict(list)

    def add_variable(self, name: str, values: List[Any]) -> None:
        """Add a variable to the graph."""
        self._variables[name] = values

    def add_factor(self, factor: Factor) -> None:
        """Add a factor to the graph."""
        self._factors[factor.name] = factor
        self._factor_to_vars[factor.name] = factor.variables

        for var in factor.variables:
            self._var_to_factors[var].append(factor.name)

    def get_variable_values(self, var: str) -> List[Any]:
        """Get possible values for a variable."""
        return self._variables.get(var, [])

    def get_neighboring_factors(self, var: str) -> List[str]:
        """Get factors connected to a variable."""
        return self._var_to_factors.get(var, [])

    def get_neighboring_variables(self, factor: str) -> List[str]:
        """Get variables in a factor."""
        return self._factor_to_vars.get(factor, [])

    def get_factor(self, name: str) -> Optional[Factor]:
        """Get a factor by name."""
        return self._factors.get(name)

    def is_tree(self) -> bool:
        """Check if the factor graph is a tree (no loops)."""
        # For a tree: |E| = |V| - 1 where V = variables + factors
        n_nodes = len(self._variables) + len(self._factors)
        n_edges = sum(len(f.variables) for f in self._factors.values())

        if n_edges != n_nodes - 1:
            return False

        # Also check connectivity via BFS
        if not self._variables:
            return True

        start = next(iter(self._variables))
        visited = set()
        queue = [("var", start)]

        while queue:
            node_type, node = queue.pop(0)
            if (node_type, node) in visited:
                continue
            visited.add((node_type, node))

            if node_type == "var":
                for factor in self._var_to_factors.get(node, []):
                    queue.append(("factor", factor))
            else:
                for var in self._factor_to_vars.get(node, []):
                    queue.append(("var", var))

        expected = len(self._variables) + len(self._factors)
        return len(visited) == expected


class BeliefPropagation:
    """
    Belief Propagation algorithm on factor graphs.

    Supports:
    - Exact inference on trees
    - Loopy BP for graphs with cycles
    - Max-product for MAP inference
    """

    def __init__(
        self,
        factor_graph: FactorGraph,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        damping: float = 0.0,
    ):
        self.graph = factor_graph
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.damping = damping

        # Messages: (source, target) -> Message
        self._messages: Dict[Tuple[str, str], Message] = {}

        # Beliefs
        self._beliefs: Dict[str, Dict[Any, float]] = {}

    def initialize_messages(self) -> None:
        """Initialize all messages to uniform."""
        self._messages.clear()

        # Variable to factor messages
        for var, values in self.graph._variables.items():
            for factor in self.graph._var_to_factors.get(var, []):
                uniform = {v: 1.0 / len(values) for v in values}
                self._messages[(var, factor)] = Message(var, factor, uniform)

        # Factor to variable messages
        for factor_name, factor in self.graph._factors.items():
            for var in factor.variables:
                values = self.graph._variables.get(var, [])
                uniform = {v: 1.0 / len(values) for v in values}
                self._messages[(factor_name, var)] = Message(factor_name, var, uniform)

    def _compute_var_to_factor_message(
        self,
        var: str,
        target_factor: str,
    ) -> Message:
        """Compute message from variable to factor."""
        values = self.graph._variables.get(var, [])

        # Product of all incoming messages except from target
        result = {v: 1.0 for v in values}

        for factor in self.graph._var_to_factors.get(var, []):
            if factor != target_factor:
                msg = self._messages.get((factor, var))
                if msg:
                    for v in values:
                        result[v] *= msg.values.get(v, 1.0)

        return Message(var, target_factor, result).normalize()

    def _compute_factor_to_var_message(
        self,
        factor_name: str,
        target_var: str,
    ) -> Message:
        """Compute message from factor to variable."""
        factor = self.graph.get_factor(factor_name)
        if factor is None:
            return Message(factor_name, target_var, {})

        target_values = self.graph._variables.get(target_var, [])
        other_vars = [v for v in factor.variables if v != target_var]

        # Sum over all assignments to other variables
        result = {v: 0.0 for v in target_values}

        for key, potential in factor.potentials.items():
            assignment = dict(key)

            if target_var not in assignment:
                continue

            target_val = assignment[target_var]

            # Multiply by incoming messages
            msg_product = potential
            for var in other_vars:
                if var in assignment:
                    msg = self._messages.get((var, factor_name))
                    if msg:
                        msg_product *= msg.values.get(assignment[var], 1.0)

            result[target_val] += msg_product

        return Message(factor_name, target_var, result).normalize()

    def run(self, schedule: str = "parallel") -> int:
        """
        Run belief propagation until convergence.

        Returns number of iterations.
        """
        self.initialize_messages()

        for iteration in range(self.max_iterations):
            max_diff = 0.0

            if schedule == "parallel":
                # Compute all new messages
                new_messages = {}

                # Variable to factor
                for var in self.graph._variables:
                    for factor in self.graph._var_to_factors.get(var, []):
                        new_msg = self._compute_var_to_factor_message(var, factor)
                        new_messages[(var, factor)] = new_msg

                # Factor to variable
                for factor_name in self.graph._factors:
                    for var in self.graph._factor_to_vars.get(factor_name, []):
                        new_msg = self._compute_factor_to_var_message(factor_name, var)
                        new_messages[(factor_name, var)] = new_msg

                # Update messages with damping
                for key, new_msg in new_messages.items():
                    old_msg = self._messages.get(key)
                    if old_msg and self.damping > 0:
                        damped = {}
                        for v in set(new_msg.values.keys()) | set(old_msg.values.keys()):
                            new_val = new_msg.values.get(v, 0)
                            old_val = old_msg.values.get(v, 0)
                            damped[v] = self.damping * old_val + (1 - self.damping) * new_val
                        new_msg = Message(new_msg.source, new_msg.target, damped)

                    # Compute difference
                    if old_msg:
                        for v in set(new_msg.values.keys()) | set(old_msg.values.keys()):
                            diff = abs(new_msg.values.get(v, 0) - old_msg.values.get(v, 0))
                            max_diff = max(max_diff, diff)

                    self._messages[key] = new_msg

            # Check convergence
            if max_diff < self.tolerance:
                self._compute_beliefs()
                return iteration + 1

        self._compute_beliefs()
        return self.max_iterations

    def _compute_beliefs(self) -> None:
        """Compute beliefs from converged messages."""
        self._beliefs.clear()

        for var, values in self.graph._variables.items():
            belief = {v: 1.0 for v in values}

            for factor in self.graph._var_to_factors.get(var, []):
                msg = self._messages.get((factor, var))
                if msg:
                    for v in values:
                        belief[v] *= msg.values.get(v, 1.0)

            # Normalize
            total = sum(belief.values())
            if total > 0:
                belief = {v: p / total for v, p in belief.items()}

            self._beliefs[var] = belief

    def get_belief(self, var: str) -> Dict[Any, float]:
        """Get the marginal belief for a variable."""
        return self._beliefs.get(var, {})

    def get_map_assignment(self) -> Dict[str, Any]:
        """Get the MAP (most likely) assignment."""
        assignment = {}
        for var, belief in self._beliefs.items():
            if belief:
                assignment[var] = max(belief.keys(), key=lambda v: belief[v])
        return assignment

    def set_evidence(self, evidence: Dict[str, Any]) -> None:
        """Set evidence by modifying factors."""
        # Create evidence factors
        for var, value in evidence.items():
            if var in self.graph._variables:
                values = self.graph._variables[var]
                potentials = {}
                for v in values:
                    assignment = {var: v}
                    key = tuple(sorted(assignment.items()))
                    potentials[key] = 1.0 if v == value else 0.0

                evidence_factor = Factor(f"evidence_{var}", [var], potentials)
                self.graph.add_factor(evidence_factor)
