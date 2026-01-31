"""
Counterfactual Reasoning: "What if" queries on causal models.

Implements counterfactual inference using the twin network method.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
import numpy as np


@dataclass
class CounterfactualQuery:
    """
    A counterfactual query.

    "What would Y have been if X had been x', given that we observed X=x, Y=y?"

    Notation: P(Y_{x'} | X=x, Y=y)
    """
    # The counterfactual outcome variable
    outcome: str

    # The hypothetical intervention
    intervention_var: str
    intervention_value: Any

    # The factual evidence (what we observed)
    evidence: Dict[str, Any] = field(default_factory=dict)

    # Optional: specific outcome value to compute probability of
    outcome_value: Optional[Any] = None

    def __str__(self):
        evidence_str = ", ".join(f"{k}={v}" for k, v in self.evidence.items())
        return f"P({self.outcome}_{{{self.intervention_var}={self.intervention_value}}} | {evidence_str})"


@dataclass
class PotentialOutcome:
    """
    A potential outcome Y(x) - what Y would be under intervention do(X=x).

    This is the outcome that would have occurred had treatment been x.
    """
    outcome_var: str
    treatment_var: str
    treatment_value: Any
    value: Any

    def __str__(self):
        return f"{self.outcome_var}({self.treatment_var}={self.treatment_value}) = {self.value}"


class CounterfactualReasoner:
    """
    Counterfactual reasoning engine using structural causal models.

    Implements the three-step counterfactual algorithm:
    1. Abduction: Infer noise variables from evidence
    2. Action: Modify model with intervention
    3. Prediction: Compute counterfactual outcome

    Also supports:
    - Probability of necessity
    - Probability of sufficiency
    - Explanation generation
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

    def compute_counterfactual(
        self,
        query: CounterfactualQuery,
        n_samples: int = 1000,
    ) -> Dict[str, Any]:
        """
        Compute a counterfactual query.

        Uses the twin network method:
        1. Abduction: Find noise consistent with evidence
        2. Action: Apply hypothetical intervention
        3. Prediction: Compute outcome under intervention
        """
        if self.scm is None:
            return {"error": "No model specified"}

        # Step 1: Abduction - find noise values consistent with evidence
        consistent_noise = self._abduction(query.evidence, n_samples)

        if not consistent_noise:
            return {
                "error": "No noise values found consistent with evidence",
                "query": str(query),
            }

        # Step 2 & 3: Action and Prediction
        # For each consistent noise sample, compute counterfactual
        intervention = {query.intervention_var: query.intervention_value}
        counterfactual_outcomes = []

        for noise in consistent_noise:
            # Compute with intervention
            values = self.scm.compute_endogenous(noise, intervention)
            outcome = values.get(query.outcome)
            if outcome is not None:
                counterfactual_outcomes.append(outcome)

        if not counterfactual_outcomes:
            return {"error": "Could not compute counterfactual outcomes"}

        # Compute statistics
        result = {
            "query": str(query),
            "n_consistent_samples": len(consistent_noise),
            "outcomes": counterfactual_outcomes,
        }

        # Numeric outcomes
        numeric_outcomes = [o for o in counterfactual_outcomes
                          if isinstance(o, (int, float))]
        if numeric_outcomes:
            result["mean"] = float(np.mean(numeric_outcomes))
            result["std"] = float(np.std(numeric_outcomes))
            result["median"] = float(np.median(numeric_outcomes))

        # Discrete outcomes
        if query.outcome_value is not None:
            # Probability of specific outcome
            matches = sum(1 for o in counterfactual_outcomes
                         if o == query.outcome_value)
            result["probability"] = matches / len(counterfactual_outcomes)
        else:
            # Distribution
            from collections import Counter
            counts = Counter(counterfactual_outcomes)
            result["distribution"] = {
                k: v / len(counterfactual_outcomes)
                for k, v in counts.items()
            }

        return result

    def _abduction(
        self,
        evidence: Dict[str, Any],
        n_samples: int,
    ) -> List[Dict[str, Any]]:
        """
        Infer noise values consistent with observed evidence.

        Uses rejection sampling.
        """
        consistent = []
        max_attempts = n_samples * 100

        for _ in range(max_attempts):
            if len(consistent) >= n_samples:
                break

            # Sample noise
            noise = self.scm.sample_noise()

            # Compute endogenous variables
            values = self.scm.compute_endogenous(noise)

            # Check if consistent with evidence
            if all(values.get(k) == v for k, v in evidence.items()):
                consistent.append(noise)

        return consistent

    def probability_of_necessity(
        self,
        treatment: str,
        outcome: str,
        treatment_present: Any = 1,
        treatment_absent: Any = 0,
        outcome_present: Any = 1,
        n_samples: int = 1000,
    ) -> float:
        """
        Compute Probability of Necessity (PN).

        PN = P(Y_{x=0} = 0 | X=1, Y=1)

        "Given that X and Y occurred, would Y not have occurred
        had X not occurred?"

        This measures how necessary X was for Y.
        """
        # Evidence: X=1, Y=1
        evidence = {treatment: treatment_present, outcome: outcome_present}

        # Counterfactual: What if X=0?
        query = CounterfactualQuery(
            outcome=outcome,
            intervention_var=treatment,
            intervention_value=treatment_absent,
            evidence=evidence,
        )

        result = self.compute_counterfactual(query, n_samples)

        if "distribution" in result:
            # P(Y_0 != Y | X=1, Y=1)
            counterfactual_absent = result["distribution"].get(
                1 - outcome_present if isinstance(outcome_present, int) else None,
                0
            )
            return counterfactual_absent
        elif "outcomes" in result:
            # Probability that outcome differs
            different = sum(
                1 for o in result["outcomes"]
                if o != outcome_present
            )
            return different / len(result["outcomes"])

        return 0.0

    def probability_of_sufficiency(
        self,
        treatment: str,
        outcome: str,
        treatment_present: Any = 1,
        treatment_absent: Any = 0,
        outcome_present: Any = 1,
        n_samples: int = 1000,
    ) -> float:
        """
        Compute Probability of Sufficiency (PS).

        PS = P(Y_{x=1} = 1 | X=0, Y=0)

        "Given that X and Y did not occur, would Y have occurred
        had X occurred?"

        This measures how sufficient X would be for Y.
        """
        # Evidence: X=0, Y=0
        outcome_absent = 1 - outcome_present if isinstance(outcome_present, int) else 0
        evidence = {treatment: treatment_absent, outcome: outcome_absent}

        # Counterfactual: What if X=1?
        query = CounterfactualQuery(
            outcome=outcome,
            intervention_var=treatment,
            intervention_value=treatment_present,
            evidence=evidence,
        )

        result = self.compute_counterfactual(query, n_samples)

        if "distribution" in result:
            return result["distribution"].get(outcome_present, 0)
        elif "outcomes" in result:
            matches = sum(
                1 for o in result["outcomes"]
                if o == outcome_present
            )
            return matches / len(result["outcomes"])

        return 0.0

    def probability_of_necessity_and_sufficiency(
        self,
        treatment: str,
        outcome: str,
        treatment_present: Any = 1,
        treatment_absent: Any = 0,
        outcome_present: Any = 1,
        n_samples: int = 1000,
    ) -> float:
        """
        Compute Probability of Necessity and Sufficiency (PNS).

        PNS = P(Y_{x=1}=1, Y_{x=0}=0)

        "Would Y occur if X, and not occur if not X?"

        This is the strongest measure of causation.
        """
        if self.scm is None:
            return 0.0

        count_pns = 0

        for _ in range(n_samples):
            # Sample a "unit" (noise realization)
            noise = self.scm.sample_noise()

            # Compute Y under X=1
            values_x1 = self.scm.compute_endogenous(
                noise, {treatment: treatment_present}
            )
            y_x1 = values_x1.get(outcome)

            # Compute Y under X=0
            values_x0 = self.scm.compute_endogenous(
                noise, {treatment: treatment_absent}
            )
            y_x0 = values_x0.get(outcome)

            # Check if necessary and sufficient
            outcome_absent = 1 - outcome_present if isinstance(outcome_present, int) else 0
            if y_x1 == outcome_present and y_x0 == outcome_absent:
                count_pns += 1

        return count_pns / n_samples

    def individual_treatment_effect(
        self,
        treatment: str,
        outcome: str,
        individual_evidence: Dict[str, Any],
        treatment_present: Any = 1,
        treatment_absent: Any = 0,
        n_samples: int = 1000,
    ) -> Dict[str, Any]:
        """
        Compute Individual Treatment Effect (ITE).

        ITE = Y_{x=1} - Y_{x=0} for a specific individual

        Uses the individual's evidence to infer their noise and
        compute both potential outcomes.
        """
        # Get noise consistent with individual
        consistent_noise = self._abduction(individual_evidence, n_samples)

        if not consistent_noise:
            return {"error": "No consistent noise found"}

        ites = []
        y1s = []
        y0s = []

        for noise in consistent_noise:
            # Y under treatment
            values_x1 = self.scm.compute_endogenous(
                noise, {treatment: treatment_present}
            )
            y1 = values_x1.get(outcome)

            # Y under control
            values_x0 = self.scm.compute_endogenous(
                noise, {treatment: treatment_absent}
            )
            y0 = values_x0.get(outcome)

            if y1 is not None and y0 is not None:
                if isinstance(y1, (int, float)) and isinstance(y0, (int, float)):
                    ites.append(y1 - y0)
                y1s.append(y1)
                y0s.append(y0)

        result = {
            "individual": individual_evidence,
            "n_samples": len(consistent_noise),
            "y_treatment": y1s,
            "y_control": y0s,
        }

        if ites:
            result["ite_mean"] = float(np.mean(ites))
            result["ite_std"] = float(np.std(ites))

        return result

    def explain_outcome(
        self,
        evidence: Dict[str, Any],
        outcome_var: str,
    ) -> Dict[str, Any]:
        """
        Generate explanation for why outcome occurred.

        Identifies which factors were necessary causes.
        """
        if self.scm is None:
            return {"error": "No model specified"}

        outcome_value = evidence.get(outcome_var)
        if outcome_value is None:
            return {"error": f"Outcome {outcome_var} not in evidence"}

        # Find parents (direct causes) of outcome
        parents = self.scm.get_parents(outcome_var)

        explanations = []

        for parent in parents:
            if parent not in evidence:
                continue

            parent_value = evidence[parent]

            # Compute probability of necessity
            pn = self.probability_of_necessity(
                treatment=parent,
                outcome=outcome_var,
                treatment_present=parent_value,
                treatment_absent=self._get_counterfactual_value(parent, parent_value),
                outcome_present=outcome_value,
                n_samples=500,
            )

            explanations.append({
                "cause": parent,
                "value": parent_value,
                "probability_necessary": pn,
            })

        # Sort by necessity
        explanations.sort(key=lambda x: x["probability_necessary"], reverse=True)

        return {
            "outcome": outcome_var,
            "outcome_value": outcome_value,
            "explanations": explanations,
            "most_likely_cause": explanations[0] if explanations else None,
        }

    def _get_counterfactual_value(
        self,
        variable: str,
        current_value: Any,
    ) -> Any:
        """Get a counterfactual value different from current."""
        # For binary: flip
        if current_value in [0, 1, True, False]:
            return 1 - current_value if isinstance(current_value, int) else not current_value

        # For numeric: return 0 or opposite sign
        if isinstance(current_value, (int, float)):
            return 0 if current_value != 0 else 1

        return None
