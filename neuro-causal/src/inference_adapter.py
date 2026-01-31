"""
Adapter for neuro-inference integration.

Bridges the DifferentiableSCM from neuro-causal with the
StructuralCausalModel from neuro-inference.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
import numpy as np

from .differentiable_scm import DifferentiableSCM, CausalMechanism
from .counterfactual import NestedCounterfactual
from .causal_discovery import CausalDiscovery


@dataclass
class AdapterConfig:
    """Configuration for the inference adapter."""
    learning_rate: float = 0.01
    abduction_iterations: int = 100
    n_effect_samples: int = 100
    gradient_eps: float = 1e-5
    random_seed: Optional[int] = None


class InferenceSCMAdapter:
    """
    Adapter that wraps DifferentiableSCM to provide
    neuro-inference compatible interface.

    This allows existing neuro-inference code to use
    the advanced differentiable SCM capabilities from neuro-causal.
    """

    def __init__(
        self,
        name: str = "adapted_scm",
        config: Optional[AdapterConfig] = None,
    ):
        self.name = name
        self.config = config or AdapterConfig()

        # Create underlying differentiable SCM
        self._dscm = DifferentiableSCM(
            name=name,
            random_seed=self.config.random_seed,
        )
        self._dscm.learning_rate = self.config.learning_rate

        # Cache for inference compatibility
        self._exogenous: Dict[str, Any] = {}
        self._endogenous: Dict[str, Any] = {}
        self._equations: Dict[str, Any] = {}
        self._noise_distributions: Dict[str, Callable] = {}

    def add_exogenous(
        self,
        name: str,
        distribution: Optional[Callable[[], Any]] = None,
        domain: Optional[List[Any]] = None,
    ) -> Any:
        """Add an exogenous variable (neuro-inference compatible)."""
        self._exogenous[name] = {
            "domain": domain,
            "distribution": distribution or (lambda: np.random.normal(0, 1)),
        }
        if distribution:
            self._noise_distributions[name] = distribution
        return {"name": name, "type": "exogenous", "domain": domain}

    def add_endogenous(
        self,
        name: str,
        domain: Optional[List[Any]] = None,
    ) -> Any:
        """Add an endogenous variable (neuro-inference compatible)."""
        self._endogenous[name] = {"domain": domain}
        return {"name": name, "type": "endogenous", "domain": domain}

    def add_equation(
        self,
        variable: str,
        parents: List[str],
        equation: Any,
    ) -> None:
        """Add a structural equation (neuro-inference compatible)."""
        self._equations[variable] = {
            "parents": parents,
            "equation": equation,
        }

        # Create corresponding mechanism in DifferentiableSCM
        if hasattr(equation, 'coefficients') and equation.coefficients:
            # Linear equation
            self._dscm.add_linear_mechanism(
                name=variable,
                parents=parents,
                coefficients=equation.coefficients,
                intercept=getattr(equation, 'intercept', 0.0),
                noise_std=1.0,
            )
        elif hasattr(equation, 'function') and equation.function:
            # Custom function
            mechanism = CausalMechanism(
                variable=variable,
                parents=parents,
                analytical_fn=equation.function,
            )
            self._dscm.add_variable(variable, parents, mechanism)
        else:
            # Generic neural mechanism
            self._dscm.add_variable(variable, parents)

    def add_linear_equation(
        self,
        variable: str,
        parents: List[str],
        coefficients: Dict[str, float],
        intercept: float = 0.0,
        noise_var: Optional[str] = None,
    ) -> Any:
        """Add a linear structural equation (neuro-inference compatible)."""
        self._dscm.add_linear_mechanism(
            name=variable,
            parents=parents,
            coefficients=coefficients,
            intercept=intercept,
            noise_std=1.0,
        )

        self._equations[variable] = {
            "parents": parents,
            "coefficients": coefficients,
            "intercept": intercept,
            "noise_var": noise_var,
        }

        return {"variable": variable, "type": "linear"}

    def get_parents(self, variable: str) -> List[str]:
        """Get causal parents of a variable."""
        return self._dscm.get_parents(variable)

    def get_children(self, variable: str) -> List[str]:
        """Get causal children of a variable."""
        return self._dscm.get_children(variable)

    def get_ancestors(self, variable: str) -> Set[str]:
        """Get all causal ancestors."""
        return self._dscm.get_ancestors(variable)

    def get_descendants(self, variable: str) -> Set[str]:
        """Get all causal descendants."""
        return self._dscm.get_descendants(variable)

    def sample_noise(self) -> Dict[str, Any]:
        """Sample all exogenous variables."""
        noise = {}
        for name in self._dscm._variables:
            mean, std = self._dscm._exogenous.get(name, (0.0, 1.0))
            noise[name] = float(np.random.normal(mean, std))
        return noise

    def compute_endogenous(
        self,
        noise: Dict[str, Any],
        interventions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compute endogenous variables from noise."""
        return self._dscm.forward(noise, interventions)

    def sample(
        self,
        n_samples: int = 1,
        interventions: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Sample from the (possibly intervened) model."""
        return self._dscm.sample(n_samples, interventions)

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
        """Perform intervention do(X=x) and observe Y."""
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
        """Estimate average causal effect of treatment on outcome."""
        x0, x1 = treatment_values
        return self._dscm.causal_effect(treatment, outcome, x1, x0)

    def get_causal_graph(self) -> Dict[str, List[str]]:
        """Get the causal graph as adjacency list."""
        return {
            v: self._dscm.get_children(v)
            for v in self._dscm._variables
        }

    # ========== Enhanced capabilities from neuro-causal ==========

    def causal_gradient(
        self,
        treatment: str,
        outcome: str,
        values: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Compute gradient d(outcome)/d(treatment).

        This is an enhanced capability from DifferentiableSCM
        that provides gradient-based causal effect estimation.
        """
        if values is None:
            values = {}
        return self._dscm.causal_gradient(treatment, outcome, values)

    def counterfactual(
        self,
        evidence: Dict[str, float],
        intervention: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Compute counterfactual query.

        What would Y be if we had done do(X=x),
        given that we observed the evidence?
        """
        return self._dscm.counterfactual(evidence, intervention)

    def is_d_separated(
        self,
        x: str,
        y: str,
        conditioning: Optional[Set[str]] = None,
    ) -> bool:
        """Check if X and Y are d-separated given conditioning set."""
        return self._dscm.is_d_separated(x, y, conditioning)

    def fit(
        self,
        data: List[Dict[str, float]],
        n_epochs: int = 100,
    ) -> Dict[str, float]:
        """
        Fit SCM parameters to observational data.

        Uses gradient-based learning of causal mechanisms.
        """
        return self._dscm.fit(data, n_epochs)

    def intervene_model(
        self,
        interventions: Dict[str, float],
    ) -> 'InferenceSCMAdapter':
        """
        Create an intervened model do(X=x).

        Returns a new adapter wrapping the intervened DSCM.
        """
        intervened_dscm = self._dscm.intervene(interventions)
        adapter = InferenceSCMAdapter(f"{self.name}_intervened", self.config)
        adapter._dscm = intervened_dscm
        return adapter

    def statistics(self) -> Dict[str, Any]:
        """Get model statistics."""
        dscm_stats = self._dscm.statistics()
        return {
            **dscm_stats,
            "adapter_type": "InferenceSCMAdapter",
            "n_exogenous_registered": len(self._exogenous),
            "n_endogenous_registered": len(self._endogenous),
            "n_equations_registered": len(self._equations),
        }


class CausalInferenceEnhanced:
    """
    Enhanced causal inference engine combining neuro-causal
    capabilities with neuro-inference patterns.

    Provides:
    - Structure learning via CausalDiscovery
    - Nested counterfactuals
    - Gradient-based effect estimation
    - Probability of necessity/sufficiency
    """

    def __init__(
        self,
        scm: Optional[InferenceSCMAdapter] = None,
        random_seed: Optional[int] = None,
    ):
        self.scm = scm or InferenceSCMAdapter(random_seed=random_seed)
        self._discovery = CausalDiscovery(random_seed=random_seed)
        self._counterfactual_engine: Optional[NestedCounterfactual] = None

        # Statistics
        self._n_queries = 0
        self._n_discoveries = 0

    def discover_structure(
        self,
        data: List[Dict[str, float]],
        alpha: float = 0.05,
    ) -> Dict[str, List[str]]:
        """
        Discover causal structure from data.

        Uses PC algorithm from CausalDiscovery.
        """
        self._n_discoveries += 1

        # Convert to numpy array
        if not data:
            return {}

        variables = list(data[0].keys())
        n_vars = len(variables)
        n_samples = len(data)

        data_array = np.zeros((n_samples, n_vars))
        for i, sample in enumerate(data):
            for j, var in enumerate(variables):
                data_array[i, j] = sample.get(var, 0.0)

        # Run PC algorithm
        adjacency, confidence = self._discovery.pc_algorithm(data_array, alpha)

        # Convert to parent dictionary
        parents = {var: [] for var in variables}
        for i in range(n_vars):
            for j in range(n_vars):
                if adjacency[i, j] == 1:  # i -> j
                    parents[variables[j]].append(variables[i])

        return parents

    def build_scm_from_structure(
        self,
        parents: Dict[str, List[str]],
        data: Optional[List[Dict[str, float]]] = None,
    ) -> InferenceSCMAdapter:
        """
        Build SCM from discovered structure.

        Optionally fits parameters to data.
        """
        scm = InferenceSCMAdapter(name="discovered_scm")

        # Add variables with parents
        for var, var_parents in parents.items():
            if var_parents:
                # Create coefficients (will be learned if data provided)
                coefficients = {p: 0.5 for p in var_parents}
                scm.add_linear_equation(var, var_parents, coefficients)
            else:
                # Root variable
                scm._dscm.add_variable(var, [])

        # Fit if data provided
        if data:
            scm.fit(data, n_epochs=100)

        self.scm = scm
        return scm

    def probability_of_necessity(
        self,
        cause: str,
        effect: str,
        evidence: Dict[str, float],
        n_samples: int = 1000,
    ) -> float:
        """
        Compute probability of necessity (PN).

        PN = P(Y_x'=0 | X=x, Y=1)
        "Would the effect not have occurred if the cause had not occurred?"
        """
        self._n_queries += 1

        if self._counterfactual_engine is None:
            self._counterfactual_engine = NestedCounterfactual(self.scm._dscm)

        return self._counterfactual_engine.probability_of_necessity(
            cause, effect,
            cause_value=evidence.get(cause, 1.0),
            effect_value=evidence.get(effect, 1.0),
            n_samples=n_samples,
        )

    def probability_of_sufficiency(
        self,
        cause: str,
        effect: str,
        evidence: Dict[str, float],
        n_samples: int = 1000,
    ) -> float:
        """
        Compute probability of sufficiency (PS).

        PS = P(Y_x=1 | X=x', Y=0)
        "Would the effect have occurred if the cause had occurred?"
        """
        self._n_queries += 1

        if self._counterfactual_engine is None:
            self._counterfactual_engine = NestedCounterfactual(self.scm._dscm)

        return self._counterfactual_engine.probability_of_sufficiency(
            cause, effect,
            cause_value=evidence.get(cause, 1.0),
            effect_value=evidence.get(effect, 0.0),
            n_samples=n_samples,
        )

    def contrastive_explanation(
        self,
        actual: Dict[str, float],
        foil: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Generate contrastive explanation.

        Why did Y=y happen instead of Y=y'?
        """
        self._n_queries += 1

        if self._counterfactual_engine is None:
            self._counterfactual_engine = NestedCounterfactual(self.scm._dscm)

        return self._counterfactual_engine.contrastive_explanation(actual, foil)

    def causal_attribution(
        self,
        outcome: str,
        outcome_value: float,
        evidence: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Compute causal attribution for each cause.

        Returns contribution of each variable to the outcome.
        """
        self._n_queries += 1

        attributions = {}
        ancestors = self.scm.get_ancestors(outcome)

        for var in ancestors:
            if var in evidence:
                # Compute counterfactual with this cause removed
                cf = self.scm.counterfactual(
                    evidence,
                    {var: 0.0},  # Remove cause
                )
                # Attribution = outcome - counterfactual outcome
                attributions[var] = outcome_value - cf.get(outcome, 0.0)

        return attributions

    def statistics(self) -> Dict[str, Any]:
        """Get inference engine statistics."""
        return {
            "n_queries": self._n_queries,
            "n_discoveries": self._n_discoveries,
            "scm_stats": self.scm.statistics() if self.scm else None,
        }


__all__ = [
    'AdapterConfig',
    'InferenceSCMAdapter',
    'CausalInferenceEnhanced',
]
