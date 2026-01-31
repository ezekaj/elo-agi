"""
Counterfactual Engine: "What if" reasoning and causal inference.

The counterfactual engine enables reasoning about alternative histories
and hypothetical scenarios. This is essential for learning from mistakes,
understanding causation, and creative problem-solving.

Based on:
- Counterfactual reasoning in cognitive science
- Causal inference (Pearl, 2009)
- Mental simulation for causal understanding
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np
import time

from .transition_model import TransitionModel, Transition


class CounterfactualType(Enum):
    """Types of counterfactual queries."""
    ACTION = "action"           # What if different action?
    STATE = "state"             # What if different initial state?
    INTERVENTION = "intervention"  # What if we intervene on a variable?
    POLICY = "policy"           # What if different policy?


@dataclass
class CounterfactualParams:
    """Parameters for counterfactual reasoning."""
    n_samples: int = 10           # Samples for uncertainty estimation
    horizon: int = 20             # How far to project counterfactual
    causal_strength_threshold: float = 0.1  # Below this, effect is weak
    significance_threshold: float = 0.05  # Statistical significance


@dataclass
class Counterfactual:
    """A counterfactual query result."""
    query_type: CounterfactualType
    actual_trajectory: List[Transition]
    counterfactual_trajectory: List[Transition]
    intervention_point: int       # When intervention occurred
    actual_outcome: float         # Actual cumulative reward
    counterfactual_outcome: float  # Counterfactual cumulative reward
    effect_size: float            # Difference in outcomes
    confidence: float             # Confidence in the estimate
    causal_attribution: Dict[str, float]  # Attribution to factors
    timestamp: float = field(default_factory=time.time)

    @property
    def is_significant(self) -> bool:
        """Check if the counterfactual effect is significant."""
        return abs(self.effect_size) > 0.1 and self.confidence > 0.5

    @property
    def improved(self) -> bool:
        """Check if counterfactual would have been better."""
        return self.counterfactual_outcome > self.actual_outcome


class CounterfactualEngine:
    """
    Engine for counterfactual reasoning and causal inference.

    The counterfactual engine answers "what if" questions by:
    1. Reconstructing the actual trajectory
    2. Intervening at a specific point
    3. Simulating the alternative trajectory
    4. Comparing outcomes

    Key features:

    1. **Action counterfactuals**: What if I had done X instead of Y?
    2. **State counterfactuals**: What if the world had been different?
    3. **Policy counterfactuals**: What if I had a different strategy?
    4. **Causal attribution**: Why did the outcome occur?

    This enables learning from experience without re-experiencing it.
    """

    def __init__(
        self,
        transition_model: TransitionModel,
        params: Optional[CounterfactualParams] = None,
    ):
        self.transition_model = transition_model
        self.params = params or CounterfactualParams()

        # History
        self._counterfactual_history: List[Counterfactual] = []
        self._causal_model: Dict[str, Dict[str, float]] = {}

    def what_if_action(
        self,
        initial_state: np.ndarray,
        actual_actions: List[np.ndarray],
        alternative_action: np.ndarray,
        intervention_index: int,
    ) -> Counterfactual:
        """
        What if a different action was taken?

        Args:
            initial_state: Starting state
            actual_actions: Sequence of actions taken
            alternative_action: The counterfactual action
            intervention_index: When to substitute action

        Returns:
            Counterfactual analysis
        """
        # Simulate actual trajectory
        actual_trajectory = self._simulate_trajectory(initial_state, actual_actions)

        # Create counterfactual actions
        cf_actions = actual_actions.copy()
        cf_actions[intervention_index] = alternative_action

        # Simulate counterfactual trajectory
        cf_trajectory = self._simulate_trajectory(initial_state, cf_actions)

        # Compute outcomes
        actual_outcome = self._compute_cumulative_reward(actual_trajectory)
        cf_outcome = self._compute_cumulative_reward(cf_trajectory)

        # Compute effect size and confidence
        effect_size = cf_outcome - actual_outcome
        confidence = self._compute_confidence(actual_trajectory, cf_trajectory)

        # Causal attribution
        attribution = self._attribute_causes(
            actual_trajectory, cf_trajectory, intervention_index
        )

        result = Counterfactual(
            query_type=CounterfactualType.ACTION,
            actual_trajectory=actual_trajectory,
            counterfactual_trajectory=cf_trajectory,
            intervention_point=intervention_index,
            actual_outcome=actual_outcome,
            counterfactual_outcome=cf_outcome,
            effect_size=effect_size,
            confidence=confidence,
            causal_attribution=attribution,
        )

        self._counterfactual_history.append(result)
        if len(self._counterfactual_history) > 1000:
            self._counterfactual_history.pop(0)

        return result

    def what_if_state(
        self,
        actual_initial: np.ndarray,
        counterfactual_initial: np.ndarray,
        actions: List[np.ndarray],
    ) -> Counterfactual:
        """
        What if the initial state had been different?

        Useful for understanding how initial conditions affect outcomes.
        """
        actual_trajectory = self._simulate_trajectory(actual_initial, actions)
        cf_trajectory = self._simulate_trajectory(counterfactual_initial, actions)

        actual_outcome = self._compute_cumulative_reward(actual_trajectory)
        cf_outcome = self._compute_cumulative_reward(cf_trajectory)
        effect_size = cf_outcome - actual_outcome
        confidence = self._compute_confidence(actual_trajectory, cf_trajectory)

        attribution = self._attribute_initial_state_effect(
            actual_initial, counterfactual_initial, actual_outcome, cf_outcome
        )

        return Counterfactual(
            query_type=CounterfactualType.STATE,
            actual_trajectory=actual_trajectory,
            counterfactual_trajectory=cf_trajectory,
            intervention_point=0,
            actual_outcome=actual_outcome,
            counterfactual_outcome=cf_outcome,
            effect_size=effect_size,
            confidence=confidence,
            causal_attribution=attribution,
        )

    def what_if_intervention(
        self,
        initial_state: np.ndarray,
        actions: List[np.ndarray],
        intervention_variable: int,
        intervention_value: float,
        intervention_time: int,
    ) -> Counterfactual:
        """
        What if we directly set a state variable?

        Models external interventions (do-operator in causal inference).
        """
        # Simulate actual trajectory
        actual_trajectory = self._simulate_trajectory(initial_state, actions)

        # Simulate up to intervention point
        current_state = initial_state.copy()
        cf_trajectory = []

        for i, action in enumerate(actions):
            if i == intervention_time:
                # Apply intervention
                current_state = current_state.copy()
                if intervention_variable < len(current_state):
                    current_state[intervention_variable] = intervention_value

            transition = self.transition_model.predict(current_state, action)
            cf_trajectory.append(transition)
            current_state = transition.predicted_state

        actual_outcome = self._compute_cumulative_reward(actual_trajectory)
        cf_outcome = self._compute_cumulative_reward(cf_trajectory)
        effect_size = cf_outcome - actual_outcome
        confidence = self._compute_confidence(actual_trajectory, cf_trajectory)

        attribution = {
            f'intervention_var_{intervention_variable}': abs(effect_size)
        }

        return Counterfactual(
            query_type=CounterfactualType.INTERVENTION,
            actual_trajectory=actual_trajectory,
            counterfactual_trajectory=cf_trajectory,
            intervention_point=intervention_time,
            actual_outcome=actual_outcome,
            counterfactual_outcome=cf_outcome,
            effect_size=effect_size,
            confidence=confidence,
            causal_attribution=attribution,
        )

    def _simulate_trajectory(
        self,
        initial_state: np.ndarray,
        actions: List[np.ndarray],
    ) -> List[Transition]:
        """Simulate a trajectory given actions."""
        trajectory = []
        current_state = initial_state.copy()

        for action in actions:
            transition = self.transition_model.predict(current_state, action)
            trajectory.append(transition)
            current_state = transition.predicted_state

        return trajectory

    def _compute_cumulative_reward(
        self,
        trajectory: List[Transition],
        discount: float = 0.99,
    ) -> float:
        """Compute discounted cumulative reward."""
        total = 0.0
        for i, t in enumerate(trajectory):
            total += (discount ** i) * t.predicted_reward
        return total

    def _compute_confidence(
        self,
        actual: List[Transition],
        counterfactual: List[Transition],
    ) -> float:
        """Compute confidence in counterfactual estimate."""
        if not actual or not counterfactual:
            return 0.0

        # Lower confidence if predictions are uncertain
        actual_uncertainty = np.mean([t.uncertainty for t in actual])
        cf_uncertainty = np.mean([t.uncertainty for t in counterfactual])
        avg_uncertainty = (actual_uncertainty + cf_uncertainty) / 2

        # Confidence decreases with uncertainty
        confidence = max(0, 1 - avg_uncertainty)

        # Also consider trajectory length
        length_factor = min(1.0, len(actual) / self.params.horizon)
        confidence *= length_factor

        return float(confidence)

    def _attribute_causes(
        self,
        actual: List[Transition],
        counterfactual: List[Transition],
        intervention_point: int,
    ) -> Dict[str, float]:
        """Attribute causal responsibility to factors."""
        attribution = {}

        if not actual or not counterfactual:
            return attribution

        # Action effect
        if intervention_point < len(actual):
            action_diff = np.linalg.norm(
                actual[intervention_point].action -
                counterfactual[intervention_point].action
            )
            attribution['action_difference'] = float(action_diff)

        # State divergence over time
        divergences = []
        for i in range(min(len(actual), len(counterfactual))):
            div = np.linalg.norm(
                actual[i].predicted_state -
                counterfactual[i].predicted_state
            )
            divergences.append(div)

        if divergences:
            attribution['state_divergence'] = float(np.mean(divergences))
            attribution['max_divergence'] = float(np.max(divergences))
            attribution['divergence_growth'] = float(
                divergences[-1] - divergences[0] if len(divergences) > 1 else 0
            )

        return attribution

    def _attribute_initial_state_effect(
        self,
        actual_initial: np.ndarray,
        cf_initial: np.ndarray,
        actual_outcome: float,
        cf_outcome: float,
    ) -> Dict[str, float]:
        """Attribute effect to initial state difference."""
        attribution = {}

        # Which dimensions changed most?
        diff = np.abs(actual_initial - cf_initial)
        top_dims = np.argsort(diff)[-5:][::-1]  # Top 5 changed

        for dim in top_dims:
            if diff[dim] > 0.01:
                attribution[f'dim_{dim}'] = float(diff[dim])

        attribution['total_state_change'] = float(np.sum(diff))
        attribution['outcome_sensitivity'] = float(
            abs(cf_outcome - actual_outcome) / max(0.01, np.sum(diff))
        )

        return attribution

    def compute_causal_strength(
        self,
        initial_state: np.ndarray,
        actions: List[np.ndarray],
        cause_index: int,
        effect_index: int,
    ) -> float:
        """
        Compute causal strength between two time points.

        How much does varying the action at cause_index
        affect the state at effect_index?
        """
        if cause_index >= effect_index:
            return 0.0

        # Sample multiple alternative actions
        strengths = []
        for _ in range(self.params.n_samples):
            # Random alternative action
            alt_action = np.random.randn(len(actions[cause_index]))

            cf = self.what_if_action(
                initial_state, actions, alt_action, cause_index
            )

            # Compare states at effect_index
            if effect_index < len(cf.actual_trajectory):
                actual_state = cf.actual_trajectory[effect_index].predicted_state
                cf_state = cf.counterfactual_trajectory[effect_index].predicted_state
                strength = np.linalg.norm(actual_state - cf_state)
                strengths.append(strength)

        return float(np.mean(strengths)) if strengths else 0.0

    def learn_causal_model(
        self,
        trajectories: List[Tuple[np.ndarray, List[np.ndarray]]],
    ) -> None:
        """
        Learn a causal model from observed trajectories.

        Discovers which actions tend to cause which state changes.
        """
        for initial_state, actions in trajectories:
            for i in range(len(actions)):
                for j in range(i + 1, min(i + 5, len(actions))):
                    strength = self.compute_causal_strength(
                        initial_state, actions, i, j
                    )
                    key = f"t{i}_to_t{j}"
                    if key not in self._causal_model:
                        self._causal_model[key] = {'strength': 0, 'count': 0}
                    self._causal_model[key]['strength'] += strength
                    self._causal_model[key]['count'] += 1

    def regret_analysis(
        self,
        initial_state: np.ndarray,
        actual_actions: List[np.ndarray],
        candidate_actions: List[List[np.ndarray]],
    ) -> Dict[str, Any]:
        """
        Analyze regret: could we have done better?

        Compares actual outcome to best counterfactual.
        """
        actual_trajectory = self._simulate_trajectory(initial_state, actual_actions)
        actual_outcome = self._compute_cumulative_reward(actual_trajectory)

        best_cf_outcome = actual_outcome
        best_cf_actions = actual_actions

        for cf_actions in candidate_actions:
            cf_trajectory = self._simulate_trajectory(initial_state, cf_actions)
            cf_outcome = self._compute_cumulative_reward(cf_trajectory)

            if cf_outcome > best_cf_outcome:
                best_cf_outcome = cf_outcome
                best_cf_actions = cf_actions

        regret = best_cf_outcome - actual_outcome

        return {
            'actual_outcome': actual_outcome,
            'best_counterfactual_outcome': best_cf_outcome,
            'regret': regret,
            'regret_percentage': regret / max(0.01, abs(actual_outcome)) * 100,
            'best_actions': best_cf_actions,
            'could_have_improved': regret > 0,
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get counterfactual engine statistics."""
        if not self._counterfactual_history:
            return {
                'n_counterfactuals': 0,
                'avg_effect_size': 0.0,
                'avg_confidence': 0.0,
            }

        recent = self._counterfactual_history[-100:]

        return {
            'n_counterfactuals': len(self._counterfactual_history),
            'avg_effect_size': float(np.mean([c.effect_size for c in recent])),
            'avg_confidence': float(np.mean([c.confidence for c in recent])),
            'improvement_rate': sum(1 for c in recent if c.improved) / len(recent),
            'significant_rate': sum(1 for c in recent if c.is_significant) / len(recent),
            'causal_model_size': len(self._causal_model),
        }

    def reset(self) -> None:
        """Reset counterfactual history."""
        self._counterfactual_history = []
