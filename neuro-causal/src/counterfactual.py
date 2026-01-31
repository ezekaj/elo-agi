"""
Nested and Contrastive Counterfactuals.

Implements advanced counterfactual reasoning:
- Nested counterfactuals (counterfactuals within counterfactuals)
- Contrastive explanations (Why X and not Y?)
- Probability of necessity and sufficiency
- Counterfactual stability analysis
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
import numpy as np

from .differentiable_scm import DifferentiableSCM


class ExplanationType(Enum):
    """Types of causal explanations."""
    NECESSARY = "necessary"      # X was necessary for Y
    SUFFICIENT = "sufficient"    # X was sufficient for Y
    NECESSARY_SUFFICIENT = "necessary_and_sufficient"
    CONTRIBUTORY = "contributory"  # X contributed to Y


@dataclass
class CounterfactualQuery:
    """A counterfactual query specification."""
    evidence: Dict[str, float]  # What we observed
    intervention: Dict[str, float]  # What if we had done this?
    outcome_var: str  # Variable of interest
    description: Optional[str] = None


@dataclass
class CounterfactualResult:
    """Result of a counterfactual query."""
    query: CounterfactualQuery
    factual_value: float  # Actual outcome
    counterfactual_value: float  # Counterfactual outcome
    inferred_noise: Dict[str, float]
    confidence: float = 1.0
    n_iterations: int = 0


@dataclass
class ContrastiveExplanation:
    """A contrastive explanation: Why X and not Y?"""
    fact: str  # What happened
    foil: str  # What didn't happen (but might have)
    cause: str  # The causal variable
    cause_value: float  # Its actual value
    counterfactual_cause_value: float  # Value needed for foil
    necessity: float  # Probability of necessity
    sufficiency: float  # Probability of sufficiency
    explanation: str  # Natural language explanation


@dataclass
class NestedCounterfactualQuery:
    """
    A nested counterfactual query.

    Example: "Would Y have been y' if X had been x',
    given that if X had been x'' then Y would have been y''?"

    This represents reasoning about counterfactuals within counterfactuals.
    """
    outer_evidence: Dict[str, float]  # Outer world evidence
    outer_intervention: Dict[str, float]  # Primary counterfactual
    inner_evidence: Dict[str, float]  # Inner (nested) evidence
    inner_intervention: Dict[str, float]  # Nested counterfactual
    outcome_var: str
    nesting_depth: int = 2


class NestedCounterfactual:
    """
    Compute nested counterfactuals and causal attributions.

    Supports:
    - Standard counterfactuals
    - Nested/iterated counterfactuals
    - Contrastive explanations
    - Probability of necessity (PN)
    - Probability of sufficiency (PS)
    """

    def __init__(
        self,
        scm: DifferentiableSCM,
        n_monte_carlo: int = 100,
        convergence_threshold: float = 1e-6,
        max_iterations: int = 100,
    ):
        self.scm = scm
        self.n_monte_carlo = n_monte_carlo
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations

        # Statistics
        self._n_queries = 0
        self._n_nested_queries = 0
        self._n_explanations = 0

    def compute(
        self,
        evidence: Dict[str, float],
        intervention: Dict[str, float],
        outcome_var: str,
    ) -> CounterfactualResult:
        """
        Compute a standard counterfactual.

        "What would outcome_var have been if intervention,
        given that we observed evidence?"
        """
        self._n_queries += 1

        query = CounterfactualQuery(
            evidence=evidence,
            intervention=intervention,
            outcome_var=outcome_var,
        )

        # Step 1: Abduction - infer noise
        inferred_noise, n_iter = self._abduct(evidence)

        # Step 2: Get factual outcome
        factual_values = self.scm.forward(noise=inferred_noise)
        factual_value = factual_values.get(outcome_var, 0.0)

        # Step 3: Compute counterfactual
        cf_values = self.scm.forward(noise=inferred_noise, interventions=intervention)
        cf_value = cf_values.get(outcome_var, 0.0)

        return CounterfactualResult(
            query=query,
            factual_value=factual_value,
            counterfactual_value=cf_value,
            inferred_noise=inferred_noise,
            confidence=self._compute_confidence(evidence, factual_values),
            n_iterations=n_iter,
        )

    def compute_nested(
        self,
        primary_intervention: Dict[str, float],
        secondary_intervention: Dict[str, float],
        outcome_var: str,
        evidence: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Compute nested counterfactual.

        "What would Y be under intervention A,
        in a world where intervention B had been applied?"

        This involves reasoning about multiple layers of counterfactuals.
        """
        self._n_nested_queries += 1
        evidence = evidence or {}

        # Layer 1: Compute inner counterfactual world
        # Apply secondary intervention to get the "base" counterfactual world
        inner_noise, _ = self._abduct(evidence) if evidence else ({}, 0)

        if not inner_noise:
            # Sample noise for inner world
            inner_noise = {v: np.random.normal() for v in self.scm._variables}

        inner_values = self.scm.forward(
            noise=inner_noise,
            interventions=secondary_intervention,
        )

        # Layer 2: In the inner world, compute the primary counterfactual
        # Update evidence with inner world values
        combined_evidence = {**evidence, **inner_values}
        outer_noise, _ = self._abduct(combined_evidence)

        # Apply primary intervention
        nested_cf_values = self.scm.forward(
            noise=outer_noise,
            interventions=primary_intervention,
        )

        # Also compute non-nested for comparison
        simple_cf_values = self.scm.forward(
            noise=inner_noise,
            interventions=primary_intervention,
        )

        return {
            "nested_outcome": nested_cf_values.get(outcome_var, 0.0),
            "simple_outcome": simple_cf_values.get(outcome_var, 0.0),
            "inner_world": inner_values,
            "nested_world": nested_cf_values,
            "primary_intervention": primary_intervention,
            "secondary_intervention": secondary_intervention,
            "nesting_effect": (
                nested_cf_values.get(outcome_var, 0.0) -
                simple_cf_values.get(outcome_var, 0.0)
            ),
        }

    def contrastive_explanation(
        self,
        actual_outcome: Dict[str, float],
        foil_outcome: Dict[str, float],
        cause_variable: str,
        outcome_variable: str,
    ) -> ContrastiveExplanation:
        """
        Generate a contrastive explanation.

        "Why did outcome_variable = actual, rather than foil?"

        Identifies what change in cause_variable would have
        led to the foil outcome.
        """
        self._n_explanations += 1

        # Get actual cause value from evidence
        actual_cause_value = actual_outcome.get(cause_variable, 0.0)
        actual_outcome_value = actual_outcome.get(outcome_variable, 0.0)
        foil_outcome_value = foil_outcome.get(outcome_variable, 0.0)

        # Search for counterfactual cause value that produces foil
        cf_cause_value = self._find_counterfactual_cause(
            evidence=actual_outcome,
            cause_var=cause_variable,
            outcome_var=outcome_variable,
            target_outcome=foil_outcome_value,
        )

        # Compute necessity and sufficiency
        pn = self.probability_of_necessity(
            cause=cause_variable,
            effect=outcome_variable,
            cause_value=actual_cause_value,
            effect_value=actual_outcome_value,
            evidence=actual_outcome,
        )

        ps = self.probability_of_sufficiency(
            cause=cause_variable,
            effect=outcome_variable,
            cause_value=actual_cause_value,
            effect_value=actual_outcome_value,
            evidence=actual_outcome,
        )

        # Generate explanation text
        explanation = self._generate_explanation(
            cause_variable, actual_cause_value, cf_cause_value,
            outcome_variable, actual_outcome_value, foil_outcome_value,
            pn, ps,
        )

        return ContrastiveExplanation(
            fact=f"{outcome_variable}={actual_outcome_value:.3f}",
            foil=f"{outcome_variable}={foil_outcome_value:.3f}",
            cause=cause_variable,
            cause_value=actual_cause_value,
            counterfactual_cause_value=cf_cause_value,
            necessity=pn,
            sufficiency=ps,
            explanation=explanation,
        )

    def probability_of_necessity(
        self,
        cause: str,
        effect: str,
        cause_value: float,
        effect_value: float,
        evidence: Optional[Dict[str, float]] = None,
        threshold: float = 0.5,
    ) -> float:
        """
        Compute Probability of Necessity (PN).

        PN = P(Y_x' != y | X=x, Y=y)

        "Given that X=x and Y=y occurred, would Y have been different
        if X had been different?"
        """
        evidence = evidence or {}
        count_necessary = 0

        for _ in range(self.n_monte_carlo):
            # Sample noise consistent with evidence
            noise = self._sample_consistent_noise(evidence)

            # Check factual: does X=x lead to Y=y?
            factual = self.scm.forward(noise=noise)
            if abs(factual.get(cause, 0) - cause_value) > threshold:
                continue
            if abs(factual.get(effect, 0) - effect_value) > threshold:
                continue

            # Counterfactual: what if X != x?
            cf_cause_value = cause_value + 1.0  # Different value
            cf = self.scm.forward(noise=noise, interventions={cause: cf_cause_value})

            # Is Y different?
            if abs(cf.get(effect, 0) - effect_value) > threshold:
                count_necessary += 1

        return count_necessary / max(self.n_monte_carlo, 1)

    def probability_of_sufficiency(
        self,
        cause: str,
        effect: str,
        cause_value: float,
        effect_value: float,
        evidence: Optional[Dict[str, float]] = None,
        threshold: float = 0.5,
    ) -> float:
        """
        Compute Probability of Sufficiency (PS).

        PS = P(Y_x = y | X != x, Y != y)

        "Given that X != x and Y != y, would setting X=x have caused Y=y?"
        """
        evidence = evidence or {}
        count_valid = 0
        count_sufficient = 0

        for _ in range(self.n_monte_carlo):
            # Sample noise
            noise = self._sample_consistent_noise(evidence)

            # Check: X != x and Y != y
            factual = self.scm.forward(noise=noise)
            if abs(factual.get(cause, 0) - cause_value) < threshold:
                continue  # X = x, skip
            if abs(factual.get(effect, 0) - effect_value) < threshold:
                continue  # Y = y, skip

            count_valid += 1

            # Counterfactual: set X = x
            cf = self.scm.forward(noise=noise, interventions={cause: cause_value})

            # Does Y = y?
            if abs(cf.get(effect, 0) - effect_value) < threshold:
                count_sufficient += 1

        if count_valid == 0:
            return 0.0
        return count_sufficient / count_valid

    def probability_of_necessity_and_sufficiency(
        self,
        cause: str,
        effect: str,
        cause_value: float,
        effect_value: float,
        evidence: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Compute Probability of Necessity and Sufficiency (PNS).

        PNS = P(Y_x = y, Y_x' != y)

        Measures whether X=x is both necessary and sufficient for Y=y.
        """
        pn = self.probability_of_necessity(cause, effect, cause_value, effect_value, evidence)
        ps = self.probability_of_sufficiency(cause, effect, cause_value, effect_value, evidence)

        # PNS bounds
        return min(pn, ps)

    def explain(
        self,
        cause: str,
        effect: str,
        evidence: Dict[str, float],
    ) -> ExplanationType:
        """
        Determine the type of causal relationship.

        Returns whether the cause was necessary, sufficient,
        both, or merely contributory.
        """
        cause_value = evidence.get(cause, 0.0)
        effect_value = evidence.get(effect, 0.0)

        pn = self.probability_of_necessity(cause, effect, cause_value, effect_value, evidence)
        ps = self.probability_of_sufficiency(cause, effect, cause_value, effect_value, evidence)

        threshold = 0.7

        if pn > threshold and ps > threshold:
            return ExplanationType.NECESSARY_SUFFICIENT
        elif pn > threshold:
            return ExplanationType.NECESSARY
        elif ps > threshold:
            return ExplanationType.SUFFICIENT
        else:
            return ExplanationType.CONTRIBUTORY

    def counterfactual_stability(
        self,
        evidence: Dict[str, float],
        intervention: Dict[str, float],
        outcome_var: str,
        perturbation_std: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Analyze stability of counterfactual under noise perturbations.

        Returns statistics about how sensitive the counterfactual
        is to changes in the inferred noise values.
        """
        # Get base counterfactual
        base_result = self.compute(evidence, intervention, outcome_var)
        base_noise = base_result.inferred_noise

        cf_values = []
        for _ in range(self.n_monte_carlo):
            # Perturb noise
            perturbed_noise = {
                var: val + np.random.normal(0, perturbation_std)
                for var, val in base_noise.items()
            }

            # Compute counterfactual with perturbed noise
            cf = self.scm.forward(noise=perturbed_noise, interventions=intervention)
            cf_values.append(cf.get(outcome_var, 0.0))

        cf_values = np.array(cf_values)

        return {
            "base_counterfactual": base_result.counterfactual_value,
            "mean": float(np.mean(cf_values)),
            "std": float(np.std(cf_values)),
            "min": float(np.min(cf_values)),
            "max": float(np.max(cf_values)),
            "stability": 1.0 / (1.0 + float(np.std(cf_values))),
        }

    def _abduct(
        self,
        evidence: Dict[str, float],
    ) -> Tuple[Dict[str, float], int]:
        """Infer noise values consistent with evidence."""
        noise = {v: 0.0 for v in self.scm._variables}

        for iteration in range(self.max_iterations):
            values = self.scm.forward(noise=noise)

            max_error = 0.0
            for var, obs in evidence.items():
                if var in values:
                    error = obs - values[var]
                    noise[var] = noise.get(var, 0.0) + 0.5 * error
                    max_error = max(max_error, abs(error))

            if max_error < self.convergence_threshold:
                return noise, iteration + 1

        return noise, self.max_iterations

    def _sample_consistent_noise(
        self,
        evidence: Dict[str, float],
    ) -> Dict[str, float]:
        """Sample noise values roughly consistent with evidence."""
        if not evidence:
            return {v: np.random.normal() for v in self.scm._variables}

        # Start from abducted noise and add small perturbation
        noise, _ = self._abduct(evidence)
        for var in noise:
            noise[var] += np.random.normal(0, 0.1)

        return noise

    def _find_counterfactual_cause(
        self,
        evidence: Dict[str, float],
        cause_var: str,
        outcome_var: str,
        target_outcome: float,
        n_search: int = 50,
    ) -> float:
        """Search for cause value that produces target outcome."""
        noise, _ = self._abduct(evidence)
        actual_cause = evidence.get(cause_var, 0.0)

        best_cause = actual_cause
        best_error = float('inf')

        # Grid search around actual value
        for delta in np.linspace(-5, 5, n_search):
            test_cause = actual_cause + delta
            cf = self.scm.forward(noise=noise, interventions={cause_var: test_cause})
            error = abs(cf.get(outcome_var, 0.0) - target_outcome)

            if error < best_error:
                best_error = error
                best_cause = test_cause

        return best_cause

    def _compute_confidence(
        self,
        evidence: Dict[str, float],
        factual_values: Dict[str, float],
    ) -> float:
        """Compute confidence based on how well abduction matched evidence."""
        if not evidence:
            return 1.0

        errors = []
        for var, obs in evidence.items():
            if var in factual_values:
                errors.append(abs(factual_values[var] - obs))

        if not errors:
            return 1.0

        mean_error = np.mean(errors)
        # Confidence decays with error
        return float(1.0 / (1.0 + mean_error))

    def _generate_explanation(
        self,
        cause_var: str,
        actual_cause: float,
        cf_cause: float,
        outcome_var: str,
        actual_outcome: float,
        foil_outcome: float,
        pn: float,
        ps: float,
    ) -> str:
        """Generate natural language explanation."""
        cause_diff = cf_cause - actual_cause

        if abs(cause_diff) < 0.01:
            return f"{outcome_var} = {actual_outcome:.2f} because {cause_var} is stable."

        direction = "higher" if cause_diff > 0 else "lower"
        necessity_str = "necessary" if pn > 0.5 else "not necessary"
        sufficiency_str = "sufficient" if ps > 0.5 else "not sufficient"

        return (
            f"{outcome_var} = {actual_outcome:.2f} rather than {foil_outcome:.2f} "
            f"because {cause_var} = {actual_cause:.2f}. "
            f"If {cause_var} had been {abs(cause_diff):.2f} {direction} "
            f"({cf_cause:.2f}), then {outcome_var} would have been {foil_outcome:.2f}. "
            f"This cause was {necessity_str} and {sufficiency_str} (PN={pn:.2f}, PS={ps:.2f})."
        )

    def statistics(self) -> Dict[str, Any]:
        """Get counterfactual reasoning statistics."""
        return {
            "n_queries": self._n_queries,
            "n_nested_queries": self._n_nested_queries,
            "n_explanations": self._n_explanations,
            "n_monte_carlo": self.n_monte_carlo,
            "convergence_threshold": self.convergence_threshold,
        }
