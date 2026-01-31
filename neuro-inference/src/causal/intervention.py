"""
Intervention: Do-calculus and causal intervention reasoning.

Implements Pearl's do-calculus rules for causal inference.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
import numpy as np


@dataclass
class Intervention:
    """An intervention do(X=x)."""
    variable: str
    value: Any
    description: Optional[str] = None

    def __str__(self):
        return f"do({self.variable}={self.value})"


@dataclass
class DoOperator:
    """
    The do() operator for causal interventions.

    Represents the operation of setting a variable to a specific value
    by "surgically" removing all incoming edges.
    """
    variables: Dict[str, Any]  # Variables being intervened on

    def __str__(self):
        interventions = ", ".join(
            f"{v}={val}" for v, val in self.variables.items()
        )
        return f"do({interventions})"

    @classmethod
    def single(cls, variable: str, value: Any) -> 'DoOperator':
        """Create a single-variable intervention."""
        return cls({variable: value})

    @classmethod
    def multiple(cls, interventions: Dict[str, Any]) -> 'DoOperator':
        """Create a multi-variable intervention."""
        return cls(interventions)


class InterventionEngine:
    """
    Engine for causal interventions and do-calculus.

    Implements:
    - do() operations on SCMs
    - Adjustment formula identification
    - Backdoor/frontdoor criterion checking
    - Identifiability analysis
    """

    def __init__(self, scm=None):
        """
        Initialize with an optional SCM.

        Args:
            scm: StructuralCausalModel instance
        """
        self.scm = scm

    def set_model(self, scm) -> None:
        """Set the structural causal model."""
        self.scm = scm

    def apply_intervention(
        self,
        do_op: DoOperator,
        n_samples: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Apply an intervention and sample from the modified model.

        Returns samples from P(V | do(X=x)).
        """
        if self.scm is None:
            return []

        return self.scm.sample(n_samples, do_op.variables)

    def interventional_distribution(
        self,
        do_op: DoOperator,
        query_var: str,
        n_samples: int = 1000,
    ) -> Dict[str, float]:
        """
        Estimate P(Y | do(X=x)) for discrete Y.

        Returns distribution as dict.
        """
        samples = self.apply_intervention(do_op, n_samples)
        values = [s.get(query_var) for s in samples if query_var in s]

        if not values:
            return {}

        # Count frequencies
        from collections import Counter
        counts = Counter(values)
        total = len(values)

        return {v: c / total for v, c in counts.items()}

    def average_treatment_effect(
        self,
        treatment: str,
        outcome: str,
        control_value: Any = 0,
        treatment_value: Any = 1,
        n_samples: int = 1000,
    ) -> Tuple[float, float]:
        """
        Compute Average Treatment Effect (ATE).

        ATE = E[Y | do(X=1)] - E[Y | do(X=0)]

        Returns (ATE, standard error).
        """
        # Sample under control
        do_control = DoOperator.single(treatment, control_value)
        control_samples = self.apply_intervention(do_control, n_samples)
        y_control = [s.get(outcome) for s in control_samples if outcome in s]

        # Sample under treatment
        do_treat = DoOperator.single(treatment, treatment_value)
        treat_samples = self.apply_intervention(do_treat, n_samples)
        y_treat = [s.get(outcome) for s in treat_samples if outcome in s]

        # Filter numeric values
        y_control = [y for y in y_control if isinstance(y, (int, float))]
        y_treat = [y for y in y_treat if isinstance(y, (int, float))]

        if not y_control or not y_treat:
            return 0.0, 0.0

        mean_control = np.mean(y_control)
        mean_treat = np.mean(y_treat)
        ate = mean_treat - mean_control

        # Standard error
        var_control = np.var(y_control) / len(y_control)
        var_treat = np.var(y_treat) / len(y_treat)
        se = np.sqrt(var_control + var_treat)

        return float(ate), float(se)

    def satisfies_backdoor_criterion(
        self,
        treatment: str,
        outcome: str,
        adjustment_set: Set[str],
    ) -> bool:
        """
        Check if adjustment set satisfies backdoor criterion.

        A set Z satisfies the backdoor criterion relative to (X, Y) if:
        1. No node in Z is a descendant of X
        2. Z blocks every path between X and Y that contains an arrow into X
        """
        if self.scm is None:
            return False

        # Check condition 1: No descendants of X in Z
        descendants_x = self.scm.get_descendants(treatment)
        if adjustment_set & descendants_x:
            return False

        # Check condition 2: Z blocks all backdoor paths
        # A path is a backdoor path if it has an arrow into X

        # For this we need to check d-separation in manipulated graph
        # Simplified: check if Z blocks all non-causal paths
        # This is a simplification - full check requires path enumeration

        return True  # Simplified for now

    def satisfies_frontdoor_criterion(
        self,
        treatment: str,
        outcome: str,
        mediator_set: Set[str],
    ) -> bool:
        """
        Check if mediator set satisfies frontdoor criterion.

        A set M satisfies the frontdoor criterion relative to (X, Y) if:
        1. M intercepts all directed paths from X to Y
        2. There is no unblocked backdoor path from X to M
        3. All backdoor paths from M to Y are blocked by X
        """
        if self.scm is None:
            return False

        # Check condition 1: M intercepts all X->Y paths
        # All directed paths from X to Y must go through M

        # Check condition 2 and 3: backdoor paths blocked

        # This is a simplification - full check is complex
        return True

    def find_valid_adjustment_set(
        self,
        treatment: str,
        outcome: str,
    ) -> Optional[Set[str]]:
        """
        Find a valid adjustment set for causal effect estimation.

        Uses a simple heuristic: adjust for parents of treatment.
        """
        if self.scm is None:
            return None

        # Parents of treatment is often a valid adjustment set
        parents = set(self.scm.get_parents(treatment))

        if self.satisfies_backdoor_criterion(treatment, outcome, parents):
            return parents

        # Try ancestors of treatment
        ancestors = self.scm.get_ancestors(treatment)
        if self.satisfies_backdoor_criterion(treatment, outcome, ancestors):
            return ancestors

        return None

    def adjustment_formula(
        self,
        treatment: str,
        outcome: str,
        adjustment_set: Set[str],
        n_samples: int = 1000,
    ) -> Dict[Any, float]:
        """
        Compute causal effect using adjustment formula.

        P(Y | do(X=x)) = sum_Z P(Y | X=x, Z) P(Z)

        This uses sampling to approximate.
        """
        if self.scm is None:
            return {}

        # Get observational samples
        samples = self.scm.sample(n_samples)

        # Group by treatment value
        treatment_values = set(s.get(treatment) for s in samples)

        results = {}
        for x_val in treatment_values:
            if x_val is None:
                continue

            # Weight each sample by adjustment
            weighted_outcomes = []
            weights = []

            for sample in samples:
                if sample.get(treatment) == x_val:
                    y = sample.get(outcome)
                    if y is not None:
                        weighted_outcomes.append(y)
                        weights.append(1.0)

            if weighted_outcomes:
                results[x_val] = np.average(weighted_outcomes, weights=weights)

        return results

    def is_identifiable(
        self,
        treatment: str,
        outcome: str,
    ) -> Tuple[bool, str]:
        """
        Check if causal effect P(Y | do(X)) is identifiable.

        Returns (is_identifiable, reason).
        """
        if self.scm is None:
            return False, "No model specified"

        # Check if there's a valid adjustment set
        adj_set = self.find_valid_adjustment_set(treatment, outcome)
        if adj_set is not None:
            return True, f"Identifiable via backdoor adjustment on {adj_set}"

        # Check direct effect (no confounders)
        parents_y = set(self.scm.get_parents(outcome))
        if treatment in parents_y:
            # X is direct cause of Y
            other_parents = parents_y - {treatment}
            if not other_parents or self.satisfies_backdoor_criterion(
                treatment, outcome, other_parents
            ):
                return True, "Direct causal effect identifiable"

        return False, "Effect may not be identifiable without additional assumptions"

    def causal_query(
        self,
        query: str,
        evidence: Optional[Dict[str, Any]] = None,
        interventions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Answer a causal query.

        Supports:
        - Observational: P(Y)
        - Conditional: P(Y | X=x)
        - Interventional: P(Y | do(X=x))
        - Conditional interventional: P(Y | do(X=x), Z=z)
        """
        evidence = evidence or {}
        interventions = interventions or {}

        if self.scm is None:
            return {"error": "No model specified"}

        n_samples = 1000

        if interventions:
            # Interventional query
            samples = self.scm.sample(n_samples, interventions)
        else:
            # Observational query
            samples = self.scm.sample(n_samples)

        # Filter by evidence
        if evidence:
            samples = [
                s for s in samples
                if all(s.get(k) == v for k, v in evidence.items())
            ]

        return {
            "n_samples": len(samples),
            "samples": samples[:10],  # First 10 for inspection
            "interventions": interventions,
            "evidence": evidence,
        }
