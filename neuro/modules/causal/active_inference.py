"""
Causal Active Inference.

Integrates causal reasoning with the Free Energy Principle:
- Causal generative models for prediction
- Planning via imagined interventions
- Free energy minimization with causal structure
- Expected free energy for action selection
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
import numpy as np

from .differentiable_scm import DifferentiableSCM
from .counterfactual import NestedCounterfactual


class InferenceMode(Enum):
    """Modes of active inference."""
    PERCEPTUAL = "perceptual"    # Update beliefs from observations
    ACTIVE = "active"            # Select actions to minimize EFE
    PLANNING = "planning"        # Multi-step planning via imagination


@dataclass
class CausalBelief:
    """
    Beliefs over causal states and structure.

    Represents uncertainty over:
    - Current state values
    - Causal graph structure
    - Mechanism parameters
    """
    # State beliefs (mean and precision for each variable)
    state_means: Dict[str, float] = field(default_factory=dict)
    state_precisions: Dict[str, float] = field(default_factory=dict)

    # Structure beliefs (probability of each edge)
    edge_beliefs: Dict[Tuple[str, str], float] = field(default_factory=dict)

    # Mechanism beliefs (parameters)
    mechanism_params: Dict[str, np.ndarray] = field(default_factory=dict)

    # Confidence in overall model
    model_confidence: float = 1.0

    def entropy(self) -> float:
        """Compute belief entropy (uncertainty)."""
        entropy = 0.0

        # State entropy
        for var, prec in self.state_precisions.items():
            if prec > 0:
                entropy += 0.5 * np.log(2 * np.pi * np.e / prec)

        # Structure entropy
        for edge, prob in self.edge_beliefs.items():
            if 0 < prob < 1:
                entropy -= prob * np.log(prob + 1e-10)
                entropy -= (1 - prob) * np.log(1 - prob + 1e-10)

        return entropy

    def update_state(self, var: str, observation: float, precision: float) -> None:
        """Update beliefs about a state variable."""
        prior_mean = self.state_means.get(var, 0.0)
        prior_prec = self.state_precisions.get(var, 1.0)

        # Bayesian update (Gaussian)
        post_prec = prior_prec + precision
        post_mean = (prior_prec * prior_mean + precision * observation) / post_prec

        self.state_means[var] = post_mean
        self.state_precisions[var] = post_prec

    def predict(self, var: str) -> Tuple[float, float]:
        """Predict variable value with uncertainty."""
        mean = self.state_means.get(var, 0.0)
        prec = self.state_precisions.get(var, 1.0)
        std = 1.0 / np.sqrt(prec) if prec > 0 else float('inf')
        return mean, std


@dataclass
class ActionOutcome:
    """Predicted outcome of an action."""
    action: Dict[str, float]  # Intervention specification
    predicted_state: Dict[str, float]
    expected_free_energy: float
    epistemic_value: float  # Information gain
    pragmatic_value: float  # Goal achievement
    risk: float  # Uncertainty in outcome


class CausalActiveInference:
    """
    Active inference agent using causal world model.

    Combines:
    - Causal generative model (SCM) for prediction
    - Free energy minimization for belief updating
    - Expected free energy for action selection
    - Counterfactual imagination for planning
    """

    def __init__(
        self,
        scm: DifferentiableSCM,
        goals: Optional[Dict[str, float]] = None,
        precision_policy: float = 4.0,
        precision_observation: float = 10.0,
        planning_horizon: int = 5,
        n_action_samples: int = 20,
    ):
        self.scm = scm
        self.goals = goals or {}
        self.precision_policy = precision_policy
        self.precision_observation = precision_observation
        self.planning_horizon = planning_horizon
        self.n_action_samples = n_action_samples

        # Current beliefs
        self.beliefs = CausalBelief()
        self._initialize_beliefs()

        # Counterfactual reasoner
        self.counterfactual = NestedCounterfactual(scm)

        # History
        self._observations: List[Dict[str, float]] = []
        self._actions: List[Dict[str, float]] = []
        self._free_energies: List[float] = []

        # Statistics
        self._n_inference_steps = 0
        self._n_planning_steps = 0

    def _initialize_beliefs(self) -> None:
        """Initialize beliefs from SCM structure."""
        for var in self.scm._variables:
            self.beliefs.state_means[var] = 0.0
            self.beliefs.state_precisions[var] = 1.0

        # Initialize structure beliefs
        for var in self.scm._variables:
            for parent in self.scm._parents.get(var, []):
                self.beliefs.edge_beliefs[(parent, var)] = 1.0

    def free_energy(
        self,
        observation: Dict[str, float],
        belief: Optional[CausalBelief] = None,
    ) -> float:
        """
        Compute variational free energy.

        F = -log P(o) + KL[Q(s) || P(s|o)]

        Lower free energy = better model fit.
        """
        belief = belief or self.beliefs

        # Prediction error (negative log likelihood)
        prediction_error = 0.0
        for var, obs in observation.items():
            pred_mean, pred_std = belief.predict(var)
            error = (obs - pred_mean) ** 2
            prediction_error += 0.5 * self.precision_observation * error

        # Complexity (KL divergence from prior)
        complexity = 0.0
        for var in self.scm._variables:
            post_mean = belief.state_means.get(var, 0.0)
            post_prec = belief.state_precisions.get(var, 1.0)
            prior_prec = 1.0

            # KL between Gaussians
            kl = 0.5 * (prior_prec / post_prec + post_mean ** 2 * prior_prec - 1 + np.log(post_prec / prior_prec))
            complexity += max(0, kl)

        free_energy = prediction_error + complexity
        return free_energy

    def infer(
        self,
        observation: Dict[str, float],
        n_iterations: int = 10,
    ) -> CausalBelief:
        """
        Perform perceptual inference: update beliefs from observation.

        Uses gradient descent on free energy.
        """
        self._n_inference_steps += 1
        self._observations.append(observation)

        # Update beliefs about observed variables
        for var, obs in observation.items():
            self.beliefs.update_state(var, obs, self.precision_observation)

        # Propagate beliefs through causal structure
        for _ in range(n_iterations):
            self._propagate_beliefs()

        # Record free energy
        fe = self.free_energy(observation)
        self._free_energies.append(fe)

        return self.beliefs

    def _propagate_beliefs(self) -> None:
        """Propagate beliefs through causal graph."""
        for var in self.scm._topological_order():
            parents = self.scm._parents.get(var, [])
            if not parents:
                continue

            # Predict from parents using mechanism
            parent_values = {p: self.beliefs.state_means.get(p, 0.0) for p in parents}
            mechanism = self.scm._mechanisms.get(var)
            if mechanism:
                predicted = mechanism.forward(parent_values, 0.0)

                # Combine with current belief
                current = self.beliefs.state_means.get(var, 0.0)
                self.beliefs.state_means[var] = 0.5 * current + 0.5 * predicted

    def expected_free_energy(
        self,
        action: Dict[str, float],
        belief: Optional[CausalBelief] = None,
        horizon: int = 1,
    ) -> Tuple[float, float, float]:
        """
        Compute expected free energy of an action.

        G = E[F(o', s')] under action policy

        Returns:
            (total_efe, epistemic_value, pragmatic_value)
        """
        belief = belief or self.beliefs

        # Predict outcome using counterfactual
        current_state = {v: belief.state_means.get(v, 0.0) for v in self.scm._variables}
        predicted_state = self.scm.forward(interventions=action)

        # Epistemic value: expected information gain
        # High when predictions are uncertain
        epistemic_value = 0.0
        for var in self.scm._variables:
            prec = belief.state_precisions.get(var, 1.0)
            epistemic_value += 1.0 / (prec + 1e-8)

        # Pragmatic value: expected goal achievement
        pragmatic_value = 0.0
        for goal_var, goal_value in self.goals.items():
            if goal_var in predicted_state:
                error = (predicted_state[goal_var] - goal_value) ** 2
                pragmatic_value -= error

        # Risk: variance in predictions
        risk = sum(1.0 / (belief.state_precisions.get(v, 1.0) + 1e-8) for v in self.scm._variables)

        # EFE = risk - epistemic - pragmatic
        efe = risk - epistemic_value - pragmatic_value

        return efe, epistemic_value, pragmatic_value

    def plan_with_imagination(
        self,
        current_state: Dict[str, float],
        goal_state: Dict[str, float],
        max_steps: int = None,
    ) -> List[Dict[str, float]]:
        """
        Plan action sequence using counterfactual imagination.

        Uses nested counterfactuals to evaluate action sequences.
        """
        self._n_planning_steps += 1
        max_steps = max_steps or self.planning_horizon

        # Set goals temporarily
        old_goals = self.goals.copy()
        self.goals = goal_state

        trajectory = []
        state = current_state.copy()

        for step in range(max_steps):
            # Generate candidate actions
            actions = self._generate_actions(state)

            # Evaluate each action
            best_action = None
            best_efe = float('inf')

            for action in actions:
                # Imagine outcome
                imagined_outcome = self.counterfactual.compute(
                    evidence=state,
                    intervention=action,
                    outcome_var=list(goal_state.keys())[0] if goal_state else list(state.keys())[0],
                )

                # Compute EFE
                efe, _, _ = self.expected_free_energy(action)

                if efe < best_efe:
                    best_efe = efe
                    best_action = action

            if best_action is None:
                break

            trajectory.append(best_action)

            # Update state for next step
            state = self.scm.forward(interventions=best_action)

            # Check if goal reached
            goal_reached = all(
                abs(state.get(var, 0) - val) < 0.1
                for var, val in goal_state.items()
            )
            if goal_reached:
                break

        self.goals = old_goals
        return trajectory

    def _generate_actions(
        self,
        state: Dict[str, float],
    ) -> List[Dict[str, float]]:
        """Generate candidate actions (interventions)."""
        actions = []

        # For each variable, generate interventions
        for var in self.scm._variables:
            current_val = state.get(var, 0.0)

            # Sample intervention values
            for delta in np.linspace(-2, 2, self.n_action_samples // len(self.scm._variables)):
                actions.append({var: current_val + delta})

        return actions

    def infer_interventions(
        self,
        current: Dict[str, float],
        goal: Dict[str, float],
        max_interventions: int = 3,
    ) -> List[Dict[str, float]]:
        """
        Infer what interventions would achieve the goal.

        Uses abduction and counterfactual reasoning.
        """
        interventions = []

        # For each goal variable, find what intervention achieves it
        for goal_var, goal_value in goal.items():
            current_value = current.get(goal_var, 0.0)

            if abs(current_value - goal_value) < 0.1:
                continue  # Already at goal

            # Find causal parents
            parents = self.scm._parents.get(goal_var, [])

            if not parents:
                # Direct intervention needed
                interventions.append({goal_var: goal_value})
            else:
                # Try intervening on parents
                best_parent = None
                best_effect = 0.0

                for parent in parents:
                    effect = self.scm.causal_effect(parent, goal_var)
                    if abs(effect) > best_effect:
                        best_effect = abs(effect)
                        best_parent = parent

                if best_parent and best_effect > 0.1:
                    # Compute required intervention value
                    needed_change = goal_value - current_value
                    intervention_value = current.get(best_parent, 0.0) + needed_change / best_effect
                    interventions.append({best_parent: intervention_value})
                else:
                    interventions.append({goal_var: goal_value})

            if len(interventions) >= max_interventions:
                break

        return interventions

    def select_action(
        self,
        belief: Optional[CausalBelief] = None,
    ) -> Tuple[Dict[str, float], ActionOutcome]:
        """
        Select action that minimizes expected free energy.

        Returns best action and predicted outcome.
        """
        belief = belief or self.beliefs

        # Get current state estimate
        state = {v: belief.state_means.get(v, 0.0) for v in self.scm._variables}

        # Generate candidate actions
        actions = self._generate_actions(state)

        # Evaluate each action
        best_action = None
        best_outcome = None
        best_efe = float('inf')

        for action in actions:
            efe, epistemic, pragmatic = self.expected_free_energy(action, belief)

            if efe < best_efe:
                best_efe = efe
                best_action = action

                # Predict outcome
                predicted = self.scm.forward(interventions=action)
                risk = sum(1.0 / (belief.state_precisions.get(v, 1.0) + 1e-8) for v in self.scm._variables)

                best_outcome = ActionOutcome(
                    action=action,
                    predicted_state=predicted,
                    expected_free_energy=efe,
                    epistemic_value=epistemic,
                    pragmatic_value=pragmatic,
                    risk=risk,
                )

        if best_action:
            self._actions.append(best_action)

        return best_action or {}, best_outcome or ActionOutcome({}, {}, 0, 0, 0, 0)

    def step(
        self,
        observation: Dict[str, float],
    ) -> Tuple[Dict[str, float], ActionOutcome]:
        """
        Complete active inference step:
        1. Perceive (update beliefs from observation)
        2. Act (select action minimizing EFE)

        Returns selected action and predicted outcome.
        """
        # Perception
        self.infer(observation)

        # Action
        action, outcome = self.select_action()

        return action, outcome

    def evaluate_policy(
        self,
        policy: List[Dict[str, float]],
        initial_state: Dict[str, float],
    ) -> float:
        """
        Evaluate a policy (sequence of actions).

        Returns cumulative expected free energy.
        """
        state = initial_state.copy()
        total_efe = 0.0

        # Create temporary belief
        temp_belief = CausalBelief()
        for var, val in state.items():
            temp_belief.state_means[var] = val
            temp_belief.state_precisions[var] = self.beliefs.state_precisions.get(var, 1.0)

        for action in policy:
            # Compute EFE for this action
            efe, _, _ = self.expected_free_energy(action, temp_belief)
            total_efe += efe

            # Update state
            state = self.scm.forward(interventions=action)
            for var, val in state.items():
                temp_belief.state_means[var] = val

        return total_efe

    def model_evidence(self, observations: List[Dict[str, float]]) -> float:
        """
        Estimate model evidence (marginal likelihood).

        Higher is better (model explains data well).
        """
        total_log_evidence = 0.0

        for obs in observations:
            # Approximate log evidence as negative free energy
            fe = self.free_energy(obs)
            total_log_evidence -= fe

        return total_log_evidence

    def statistics(self) -> Dict[str, Any]:
        """Get active inference statistics."""
        return {
            "n_inference_steps": self._n_inference_steps,
            "n_planning_steps": self._n_planning_steps,
            "n_observations": len(self._observations),
            "n_actions": len(self._actions),
            "avg_free_energy": float(np.mean(self._free_energies)) if self._free_energies else 0.0,
            "belief_entropy": self.beliefs.entropy(),
            "n_goals": len(self.goals),
        }
