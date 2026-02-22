"""
Bayesian Networks: DAG-based probabilistic models.

Implements discrete and continuous Bayesian networks
with exact and approximate inference.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set, Union
import numpy as np
from collections import deque


@dataclass
class CPT:
    """Conditional Probability Table."""

    variable: str
    parents: List[str]
    probabilities: Dict[Tuple, Dict[str, float]]  # (parent_values) -> {value: prob}

    def get_probability(
        self,
        value: str,
        parent_values: Optional[Dict[str, str]] = None,
    ) -> float:
        """Get probability of value given parent values."""
        if not self.parents:
            # No parents - marginal probability
            return self.probabilities.get((), {}).get(value, 0.0)

        if parent_values is None:
            parent_values = {}

        # Build parent value tuple in correct order
        key = tuple(parent_values.get(p, None) for p in self.parents)

        if key in self.probabilities:
            return self.probabilities[key].get(value, 0.0)

        return 0.0

    def get_distribution(
        self,
        parent_values: Optional[Dict[str, str]] = None,
    ) -> Dict[str, float]:
        """Get full distribution given parent values."""
        if not self.parents:
            return self.probabilities.get((), {})

        if parent_values is None:
            parent_values = {}

        key = tuple(parent_values.get(p, None) for p in self.parents)
        return self.probabilities.get(key, {})


@dataclass
class DiscreteNode:
    """A discrete random variable node in a Bayesian network."""

    name: str
    values: List[str]  # Possible values
    parents: List[str] = field(default_factory=list)
    cpt: Optional[CPT] = None

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, DiscreteNode):
            return self.name == other.name
        return False


@dataclass
class ContinuousNode:
    """A continuous random variable node (Gaussian)."""

    name: str
    parents: List[str] = field(default_factory=list)
    mean: float = 0.0
    variance: float = 1.0
    # Linear Gaussian: mean = base_mean + sum(weight_i * parent_i)
    weights: Dict[str, float] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.name)

    def get_conditional_mean(self, parent_values: Dict[str, float]) -> float:
        """Get conditional mean given parent values."""
        result = self.mean
        for parent, weight in self.weights.items():
            if parent in parent_values:
                result += weight * parent_values[parent]
        return result

    def sample(self, parent_values: Optional[Dict[str, float]] = None) -> float:
        """Sample from conditional distribution."""
        mean = self.get_conditional_mean(parent_values or {})
        return np.random.normal(mean, np.sqrt(self.variance))


@dataclass
class NetworkQuery:
    """A query to the Bayesian network."""

    query_variables: List[str]  # Variables to query
    evidence: Dict[str, str] = field(default_factory=dict)  # Observed values
    method: str = "variable_elimination"  # Inference method


class BayesianNetwork:
    """
    Bayesian Network for probabilistic inference.

    Supports:
    - Discrete and continuous nodes
    - Exact inference (variable elimination)
    - Approximate inference (sampling)
    - D-separation queries
    """

    def __init__(self, name: str = "bn"):
        self.name = name

        # Node storage
        self._nodes: Dict[str, Union[DiscreteNode, ContinuousNode]] = {}

        # Graph structure
        self._parents: Dict[str, List[str]] = {}
        self._children: Dict[str, List[str]] = {}

        # Topological order cache
        self._topo_order: Optional[List[str]] = None

    def add_node(
        self,
        node: Union[DiscreteNode, ContinuousNode],
    ) -> None:
        """Add a node to the network."""
        self._nodes[node.name] = node
        self._parents[node.name] = list(node.parents)

        # Initialize children list
        if node.name not in self._children:
            self._children[node.name] = []

        # Update children of parents
        for parent in node.parents:
            if parent not in self._children:
                self._children[parent] = []
            self._children[parent].append(node.name)

        # Invalidate topological order
        self._topo_order = None

    def add_discrete_node(
        self,
        name: str,
        values: List[str],
        parents: Optional[List[str]] = None,
        cpt: Optional[Dict[Tuple, Dict[str, float]]] = None,
    ) -> DiscreteNode:
        """Convenience method to add a discrete node."""
        parents = parents or []

        cpt_obj = None
        if cpt is not None:
            cpt_obj = CPT(variable=name, parents=parents, probabilities=cpt)

        node = DiscreteNode(name=name, values=values, parents=parents, cpt=cpt_obj)
        self.add_node(node)
        return node

    def add_continuous_node(
        self,
        name: str,
        parents: Optional[List[str]] = None,
        mean: float = 0.0,
        variance: float = 1.0,
        weights: Optional[Dict[str, float]] = None,
    ) -> ContinuousNode:
        """Convenience method to add a continuous node."""
        node = ContinuousNode(
            name=name,
            parents=parents or [],
            mean=mean,
            variance=variance,
            weights=weights or {},
        )
        self.add_node(node)
        return node

    def get_node(self, name: str) -> Optional[Union[DiscreteNode, ContinuousNode]]:
        """Get a node by name."""
        return self._nodes.get(name)

    def has_node(self, name: str) -> bool:
        """Check if node exists."""
        return name in self._nodes

    def get_parents(self, name: str) -> List[str]:
        """Get parents of a node."""
        return self._parents.get(name, [])

    def get_children(self, name: str) -> List[str]:
        """Get children of a node."""
        return self._children.get(name, [])

    def get_topological_order(self) -> List[str]:
        """Get nodes in topological order."""
        if self._topo_order is not None:
            return self._topo_order

        # Kahn's algorithm
        in_degree = {n: len(self._parents.get(n, [])) for n in self._nodes}
        queue = deque([n for n, d in in_degree.items() if d == 0])
        order = []

        while queue:
            node = queue.popleft()
            order.append(node)

            for child in self._children.get(node, []):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(order) != len(self._nodes):
            raise ValueError("Graph contains a cycle")

        self._topo_order = order
        return order

    def is_d_separated(
        self,
        x: str,
        y: str,
        z: Set[str],
    ) -> bool:
        """
        Check if X and Y are d-separated given Z.

        Uses the Bayes Ball algorithm.
        """
        # Start from X, try to reach Y
        # Track (node, direction) where direction is "up" or "down"
        visited = set()
        queue = [(x, "up"), (x, "down")]

        while queue:
            node, direction = queue.pop(0)

            if (node, direction) in visited:
                continue
            visited.add((node, direction))

            if node == y:
                return False  # Found a path, not d-separated

            node_obj = self._nodes.get(node)
            if node_obj is None:
                continue

            in_z = node in z

            if direction == "up":
                # Came from child
                if not in_z:
                    # Can go to parents
                    for parent in self._parents.get(node, []):
                        queue.append((parent, "up"))
                    # Can go to other children
                    for child in self._children.get(node, []):
                        queue.append((child, "down"))
            else:
                # Came from parent (direction == "down")
                if not in_z:
                    # Can go to children
                    for child in self._children.get(node, []):
                        queue.append((child, "down"))
                if in_z:
                    # Can go to parents (v-structure becomes active)
                    for parent in self._parents.get(node, []):
                        queue.append((parent, "up"))

        return True  # No path found, d-separated

    def query(
        self,
        query_vars: List[str],
        evidence: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Query the network using variable elimination.

        Returns marginal distributions for query variables.
        """
        evidence = evidence or {}

        # For discrete nodes only
        discrete_nodes = {
            n: node for n, node in self._nodes.items() if isinstance(node, DiscreteNode)
        }

        result = {}

        for query_var in query_vars:
            if query_var not in discrete_nodes:
                continue

            # Variable elimination
            distribution = self._variable_elimination(query_var, evidence, discrete_nodes)
            result[query_var] = distribution

        return result

    def _variable_elimination(
        self,
        query_var: str,
        evidence: Dict[str, str],
        discrete_nodes: Dict[str, DiscreteNode],
    ) -> Dict[str, float]:
        """Perform variable elimination for a single query variable."""
        # Get elimination order (reverse topological, excluding query and evidence)
        topo_order = self.get_topological_order()

        elim_order = [
            n
            for n in reversed(topo_order)
            if n != query_var and n not in evidence and n in discrete_nodes
        ]

        # Initialize factors from CPTs
        factors = []
        for name, node in discrete_nodes.items():
            if node.cpt:
                factors.append(self._cpt_to_factor(node.cpt, evidence))

        # Eliminate variables
        for var in elim_order:
            # Find factors involving this variable
            relevant = [f for f in factors if var in f["vars"]]
            other = [f for f in factors if var not in f["vars"]]

            if relevant:
                # Multiply all relevant factors
                product = relevant[0]
                for f in relevant[1:]:
                    product = self._multiply_factors(product, f)

                # Sum out the variable
                marginalized = self._marginalize_factor(product, var)
                factors = other + [marginalized]
            else:
                factors = other

        # Multiply remaining factors
        if not factors:
            # Uniform distribution
            node = discrete_nodes[query_var]
            return {v: 1.0 / len(node.values) for v in node.values}

        result = factors[0]
        for f in factors[1:]:
            result = self._multiply_factors(result, f)

        # Normalize
        total = sum(result["probs"].values())
        if total > 0:
            return {k: v / total for k, v in result["probs"].items()}

        # Uniform if all zero
        node = discrete_nodes[query_var]
        return {v: 1.0 / len(node.values) for v in node.values}

    def _cpt_to_factor(
        self,
        cpt: CPT,
        evidence: Dict[str, str],
    ) -> Dict[str, Any]:
        """Convert CPT to factor, applying evidence."""
        vars_list = [cpt.variable] + list(cpt.parents)

        # Filter out evidence variables
        free_vars = [v for v in vars_list if v not in evidence]

        probs = {}

        # Enumerate all assignments
        var_values = {}
        for var in vars_list:
            node = self._nodes.get(var)
            if isinstance(node, DiscreteNode):
                var_values[var] = node.values
            else:
                var_values[var] = ["0"]  # Placeholder for continuous

        # Generate all combinations
        from itertools import product as iter_product

        value_lists = [var_values[v] for v in vars_list]
        for assignment in iter_product(*value_lists):
            var_assignment = dict(zip(vars_list, assignment))

            # Check if consistent with evidence
            consistent = all(
                var_assignment.get(e) == v for e, v in evidence.items() if e in var_assignment
            )

            if consistent:
                # Get probability from CPT
                parent_vals = {p: var_assignment[p] for p in cpt.parents}
                prob = cpt.get_probability(var_assignment[cpt.variable], parent_vals)

                # Key is just the free variable values
                key = tuple(var_assignment[v] for v in free_vars)
                probs[key] = probs.get(key, 0) + prob

        return {"vars": free_vars, "probs": probs}

    def _multiply_factors(
        self,
        f1: Dict[str, Any],
        f2: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Multiply two factors."""
        vars1 = f1["vars"]
        vars2 = f2["vars"]

        # Combined variables
        all_vars = list(vars1)
        for v in vars2:
            if v not in all_vars:
                all_vars.append(v)

        probs = {}

        # Get value lists for all variables
        var_values = {}
        for var in all_vars:
            node = self._nodes.get(var)
            if isinstance(node, DiscreteNode):
                var_values[var] = node.values
            else:
                var_values[var] = ["0"]

        from itertools import product as iter_product

        value_lists = [var_values[v] for v in all_vars]
        for assignment in iter_product(*value_lists):
            var_assignment = dict(zip(all_vars, assignment))

            # Get keys for each factor
            key1 = tuple(var_assignment[v] for v in vars1)
            key2 = tuple(var_assignment[v] for v in vars2)

            p1 = f1["probs"].get(key1, 0)
            p2 = f2["probs"].get(key2, 0)

            result_key = assignment
            probs[result_key] = p1 * p2

        return {"vars": all_vars, "probs": probs}

    def _marginalize_factor(
        self,
        factor: Dict[str, Any],
        var: str,
    ) -> Dict[str, Any]:
        """Sum out a variable from a factor."""
        vars_list = factor["vars"]

        if var not in vars_list:
            return factor

        var_idx = vars_list.index(var)
        new_vars = [v for v in vars_list if v != var]

        probs = {}

        for key, prob in factor["probs"].items():
            # Remove the marginalized variable from key
            new_key = tuple(k for i, k in enumerate(key) if i != var_idx)
            probs[new_key] = probs.get(new_key, 0) + prob

        return {"vars": new_vars, "probs": probs}

    def sample(
        self,
        n_samples: int = 1,
        evidence: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Sample from the network using forward sampling.

        With evidence, uses rejection sampling.
        """
        evidence = evidence or {}
        samples = []

        topo_order = self.get_topological_order()

        attempts = 0
        max_attempts = n_samples * 1000

        while len(samples) < n_samples and attempts < max_attempts:
            attempts += 1
            sample = {}
            valid = True

            for var in topo_order:
                node = self._nodes.get(var)

                if var in evidence:
                    sample[var] = evidence[var]
                    continue

                if isinstance(node, DiscreteNode):
                    if node.cpt:
                        parent_vals = {p: sample[p] for p in node.parents}
                        dist = node.cpt.get_distribution(parent_vals)

                        if not dist:
                            valid = False
                            break

                        values = list(dist.keys())
                        probs = [dist[v] for v in values]

                        total = sum(probs)
                        if total > 0:
                            probs = [p / total for p in probs]
                            sample[var] = np.random.choice(values, p=probs)
                        else:
                            sample[var] = np.random.choice(values)
                    else:
                        sample[var] = np.random.choice(node.values)

                elif isinstance(node, ContinuousNode):
                    parent_vals = {p: float(sample.get(p, 0)) for p in node.parents}
                    sample[var] = node.sample(parent_vals)

            if valid:
                samples.append(sample)

        return samples

    def get_markov_blanket(self, node: str) -> Set[str]:
        """Get the Markov blanket of a node."""
        blanket = set()

        # Parents
        blanket.update(self._parents.get(node, []))

        # Children
        children = self._children.get(node, [])
        blanket.update(children)

        # Parents of children (co-parents)
        for child in children:
            blanket.update(self._parents.get(child, []))

        # Remove the node itself
        blanket.discard(node)

        return blanket

    def statistics(self) -> Dict[str, Any]:
        """Get network statistics."""
        n_discrete = sum(1 for n in self._nodes.values() if isinstance(n, DiscreteNode))
        n_continuous = sum(1 for n in self._nodes.values() if isinstance(n, ContinuousNode))
        n_edges = sum(len(children) for children in self._children.values())

        return {
            "n_nodes": len(self._nodes),
            "n_discrete": n_discrete,
            "n_continuous": n_continuous,
            "n_edges": n_edges,
            "name": self.name,
        }
