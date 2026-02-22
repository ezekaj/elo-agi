"""
Bayesian Network Learning: Structure and parameter learning.

Implements score-based and constraint-based structure learning.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
from collections import Counter
import math


class BayesianScore(Enum):
    """Scoring functions for structure learning."""

    BIC = "bic"  # Bayesian Information Criterion
    AIC = "aic"  # Akaike Information Criterion
    BDeu = "bdeu"  # Bayesian Dirichlet equivalent uniform
    K2 = "k2"  # K2 score
    LOG_LIKELIHOOD = "ll"  # Log likelihood


@dataclass
class LearnedCPT:
    """Learned conditional probability table."""

    variable: str
    parents: List[str]
    counts: Dict[Tuple, Counter]  # parent_values -> Counter of child values
    probabilities: Dict[Tuple, Dict[str, float]]
    pseudo_count: float = 1.0  # Laplace smoothing

    def get_probability(
        self,
        value: str,
        parent_values: Optional[Dict[str, str]] = None,
    ) -> float:
        """Get probability of value given parent values."""
        if not self.parents:
            return self.probabilities.get((), {}).get(value, 0.0)

        if parent_values is None:
            return 0.0

        key = tuple(parent_values.get(p) for p in self.parents)
        return self.probabilities.get(key, {}).get(value, 0.0)


class ParameterLearner:
    """
    Learn CPT parameters from data.

    Uses maximum likelihood estimation with optional
    Laplace smoothing (Bayesian estimation).
    """

    def __init__(
        self,
        pseudo_count: float = 1.0,
    ):
        self.pseudo_count = pseudo_count

    def learn_cpt(
        self,
        data: List[Dict[str, str]],
        variable: str,
        parents: List[str],
        possible_values: Optional[Dict[str, List[str]]] = None,
    ) -> LearnedCPT:
        """
        Learn CPT from data.

        Args:
            data: List of observations (variable -> value)
            variable: The target variable
            parents: List of parent variables
            possible_values: Dict of variable -> possible values
        """
        # Infer possible values if not provided
        if possible_values is None:
            possible_values = {}
            all_vars = [variable] + parents
            for var in all_vars:
                possible_values[var] = list(set(d.get(var) for d in data if d.get(var) is not None))

        var_values = possible_values.get(variable, [])

        # Count occurrences
        counts: Dict[Tuple, Counter] = {}

        for obs in data:
            if variable not in obs:
                continue

            value = obs[variable]

            # Get parent values
            if parents:
                parent_vals = tuple(obs.get(p) for p in parents)
                if None in parent_vals:
                    continue
            else:
                parent_vals = ()

            if parent_vals not in counts:
                counts[parent_vals] = Counter()

            counts[parent_vals][value] += 1

        # Convert to probabilities with smoothing
        probabilities = {}

        # Generate all parent combinations
        if parents:
            from itertools import product

            parent_value_lists = [possible_values.get(p, []) for p in parents]
            parent_combos = list(product(*parent_value_lists))
        else:
            parent_combos = [()]

        for parent_vals in parent_combos:
            counter = counts.get(parent_vals, Counter())
            total = sum(counter.values()) + self.pseudo_count * len(var_values)

            probs = {}
            for v in var_values:
                count = counter.get(v, 0) + self.pseudo_count
                probs[v] = count / total if total > 0 else 1.0 / len(var_values)

            probabilities[parent_vals] = probs

        return LearnedCPT(
            variable=variable,
            parents=parents,
            counts=counts,
            probabilities=probabilities,
            pseudo_count=self.pseudo_count,
        )

    def learn_all_cpts(
        self,
        data: List[Dict[str, str]],
        structure: Dict[str, List[str]],  # variable -> parents
        possible_values: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, LearnedCPT]:
        """Learn CPTs for all variables given structure."""
        cpts = {}
        for variable, parents in structure.items():
            cpts[variable] = self.learn_cpt(data, variable, parents, possible_values)
        return cpts


class StructureLearner:
    """
    Learn Bayesian network structure from data.

    Implements:
    - Score-based search (greedy hill climbing)
    - Constraint-based (PC algorithm approximation)
    """

    def __init__(
        self,
        score_type: BayesianScore = BayesianScore.BIC,
        max_parents: int = 3,
        alpha: float = 0.05,  # For independence tests
    ):
        self.score_type = score_type
        self.max_parents = max_parents
        self.alpha = alpha
        self._param_learner = ParameterLearner()

    def learn_structure(
        self,
        data: List[Dict[str, str]],
        variables: Optional[List[str]] = None,
        method: str = "greedy",
    ) -> Dict[str, List[str]]:
        """
        Learn network structure from data.

        Returns dict mapping variables to their parents.
        """
        if variables is None:
            # Infer from data
            variables = list(set(k for d in data for k in d.keys()))

        if method == "greedy":
            return self._greedy_search(data, variables)
        elif method == "pc":
            return self._pc_algorithm(data, variables)
        else:
            return {v: [] for v in variables}

    def _greedy_search(
        self,
        data: List[Dict[str, str]],
        variables: List[str],
    ) -> Dict[str, List[str]]:
        """Greedy hill climbing search."""
        # Start with empty graph
        structure = {v: [] for v in variables}

        # Get possible values
        possible_values = {}
        for var in variables:
            possible_values[var] = list(set(d.get(var) for d in data if d.get(var) is not None))

        current_score = self._score_structure(data, structure, possible_values)

        improved = True
        while improved:
            improved = False
            best_score = current_score
            best_change = None

            # Try adding edges
            for child in variables:
                for parent in variables:
                    if parent == child:
                        continue
                    if parent in structure[child]:
                        continue
                    if len(structure[child]) >= self.max_parents:
                        continue

                    # Check if adding creates cycle
                    new_structure = {k: list(v) for k, v in structure.items()}
                    new_structure[child].append(parent)

                    if not self._has_cycle(new_structure, variables):
                        score = self._score_structure(data, new_structure, possible_values)
                        if score > best_score:
                            best_score = score
                            best_change = ("add", child, parent)

            # Try removing edges
            for child in variables:
                for parent in structure[child]:
                    new_structure = {k: list(v) for k, v in structure.items()}
                    new_structure[child].remove(parent)

                    score = self._score_structure(data, new_structure, possible_values)
                    if score > best_score:
                        best_score = score
                        best_change = ("remove", child, parent)

            # Apply best change
            if best_change:
                action, child, parent = best_change
                if action == "add":
                    structure[child].append(parent)
                else:
                    structure[child].remove(parent)
                current_score = best_score
                improved = True

        return structure

    def _pc_algorithm(
        self,
        data: List[Dict[str, str]],
        variables: List[str],
    ) -> Dict[str, List[str]]:
        """
        PC algorithm (simplified constraint-based).

        Uses conditional independence tests.
        """
        # Start with complete undirected graph
        adjacencies = {v: set(variables) - {v} for v in variables}

        # Phase 1: Remove edges based on independence
        for d in range(len(variables)):
            for x in variables:
                neighbors = list(adjacencies[x])
                for y in neighbors:
                    if y not in adjacencies[x]:
                        continue

                    # Find conditioning set of size d
                    other_neighbors = [n for n in neighbors if n != y]

                    if len(other_neighbors) >= d:
                        from itertools import combinations

                        for z in combinations(other_neighbors, d):
                            if self._conditional_independent(data, x, y, set(z)):
                                adjacencies[x].discard(y)
                                adjacencies[y].discard(x)
                                break

        # Phase 2: Orient edges (simplified)
        structure = {v: [] for v in variables}

        # Use topological hints from data
        for x in variables:
            for y in adjacencies[x]:
                if y not in structure[x] and x not in structure[y]:
                    # Simple heuristic: earlier in list is parent
                    if variables.index(x) < variables.index(y):
                        structure[y].append(x)
                    else:
                        structure[x].append(y)

        return structure

    def _conditional_independent(
        self,
        data: List[Dict[str, str]],
        x: str,
        y: str,
        z: Set[str],
    ) -> bool:
        """Test if X is conditionally independent of Y given Z."""
        # Use chi-squared test approximation

        # Count joint occurrences
        counts_xyz = Counter()
        counts_xz = Counter()
        counts_yz = Counter()
        counts_z = Counter()

        for obs in data:
            if x not in obs or y not in obs:
                continue
            if any(zv not in obs for zv in z):
                continue

            z_vals = tuple(obs[zv] for zv in sorted(z))
            x_val = obs[x]
            y_val = obs[y]

            counts_xyz[(x_val, y_val, z_vals)] += 1
            counts_xz[(x_val, z_vals)] += 1
            counts_yz[(y_val, z_vals)] += 1
            counts_z[z_vals] += 1

        if not counts_z:
            return True  # No data, assume independent

        # Compute chi-squared statistic
        chi_sq = 0.0
        len(data)

        for (x_val, y_val, z_vals), observed in counts_xyz.items():
            n_xz = counts_xz.get((x_val, z_vals), 0)
            n_yz = counts_yz.get((y_val, z_vals), 0)
            n_z = counts_z.get(z_vals, 0)

            if n_z > 0:
                expected = (n_xz * n_yz) / n_z
                if expected > 0:
                    chi_sq += (observed - expected) ** 2 / expected

        # Compare to threshold (approximation)
        # For alpha=0.05, chi-sq critical value depends on df
        # Use simple threshold
        threshold = 3.84  # df=1, alpha=0.05

        return chi_sq < threshold

    def _has_cycle(
        self,
        structure: Dict[str, List[str]],
        variables: List[str],
    ) -> bool:
        """Check if structure has a directed cycle."""
        # DFS-based cycle detection
        visited = set()
        rec_stack = set()

        def dfs(node):
            visited.add(node)
            rec_stack.add(node)

            # Find children (nodes where this is parent)
            for child, parents in structure.items():
                if node in parents:
                    if child not in visited:
                        if dfs(child):
                            return True
                    elif child in rec_stack:
                        return True

            rec_stack.remove(node)
            return False

        for v in variables:
            if v not in visited:
                if dfs(v):
                    return True

        return False

    def _score_structure(
        self,
        data: List[Dict[str, str]],
        structure: Dict[str, List[str]],
        possible_values: Dict[str, List[str]],
    ) -> float:
        """Score a structure using the selected scoring function."""
        if self.score_type == BayesianScore.BIC:
            return self._bic_score(data, structure, possible_values)
        elif self.score_type == BayesianScore.AIC:
            return self._aic_score(data, structure, possible_values)
        elif self.score_type == BayesianScore.LOG_LIKELIHOOD:
            return self._log_likelihood(data, structure, possible_values)
        else:
            return self._bic_score(data, structure, possible_values)

    def _log_likelihood(
        self,
        data: List[Dict[str, str]],
        structure: Dict[str, List[str]],
        possible_values: Dict[str, List[str]],
    ) -> float:
        """Compute log likelihood of data given structure."""
        cpts = self._param_learner.learn_all_cpts(data, structure, possible_values)

        ll = 0.0
        for obs in data:
            for var, parents in structure.items():
                if var not in obs:
                    continue

                cpt = cpts.get(var)
                if cpt:
                    parent_vals = {p: obs.get(p) for p in parents}
                    prob = cpt.get_probability(obs[var], parent_vals)
                    if prob > 0:
                        ll += math.log(prob)

        return ll

    def _bic_score(
        self,
        data: List[Dict[str, str]],
        structure: Dict[str, List[str]],
        possible_values: Dict[str, List[str]],
    ) -> float:
        """Bayesian Information Criterion score."""
        ll = self._log_likelihood(data, structure, possible_values)

        # Count parameters
        n_params = 0
        for var, parents in structure.items():
            var_card = len(possible_values.get(var, [1]))
            parent_card = 1
            for p in parents:
                parent_card *= len(possible_values.get(p, [1]))
            n_params += parent_card * (var_card - 1)

        n = len(data)
        bic = ll - 0.5 * n_params * math.log(n) if n > 0 else ll

        return bic

    def _aic_score(
        self,
        data: List[Dict[str, str]],
        structure: Dict[str, List[str]],
        possible_values: Dict[str, List[str]],
    ) -> float:
        """Akaike Information Criterion score."""
        ll = self._log_likelihood(data, structure, possible_values)

        # Count parameters
        n_params = 0
        for var, parents in structure.items():
            var_card = len(possible_values.get(var, [1]))
            parent_card = 1
            for p in parents:
                parent_card *= len(possible_values.get(p, [1]))
            n_params += parent_card * (var_card - 1)

        aic = ll - n_params

        return aic
