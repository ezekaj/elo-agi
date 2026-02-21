"""
The Dopamine System

Implements the computational functions of dopamine signaling:
1. Prediction Error: Surprise (difference between expected and actual)
2. Incentive Salience: "Wanting" (motivational pull toward rewards)
3. Benefit/Cost Ratio: Potentiates sensitivity to benefits vs. costs

Key insight: Dopamine does NOT signal pleasure. It signals:
- How surprising an outcome was
- How much we "want" something (not how much we "like" it)
- Whether benefits outweigh costs

Motivation can both ENHANCE and IMPAIR decision-making:
- Enhance: Faster goal-directed responses under reward
- Impair: Over-reliance on habitual responses
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum


class DopamineChannel(Enum):
    """Different dopamine signaling channels"""
    PHASIC = "phasic"        # Fast, burst signaling (prediction error)
    TONIC = "tonic"          # Slow, background level (motivation state)
    RAMPING = "ramping"      # Gradual increase toward reward


@dataclass
class DopamineSignal:
    """A dopamine signal with multiple components"""
    prediction_error: float          # RPE: actual - expected
    incentive_salience: float        # "Wanting" level
    benefit_cost_ratio: float        # Net value signal
    channel: DopamineChannel = DopamineChannel.PHASIC
    timestamp: float = 0.0


@dataclass
class RewardPrediction:
    """Prediction about upcoming reward"""
    expected_value: float
    uncertainty: float
    time_to_reward: float
    cue_strength: float


class PredictionErrorComputer:
    """Computes reward prediction errors (RPE).

    RPE = Actual Reward - Expected Reward

    This is the core dopamine signal that drives learning.
    Positive RPE: Better than expected -> increase approach
    Negative RPE: Worse than expected -> decrease approach
    Zero RPE: As expected -> no learning
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        eligibility_decay: float = 0.9
    ):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.eligibility_decay = eligibility_decay

        # Value function (state -> expected value)
        self.state_values: Dict[tuple, float] = {}

        # TD error history
        self.td_errors: deque = deque(maxlen=1000)

        # Eligibility traces for credit assignment
        self.eligibility_traces: Dict[tuple, float] = {}

        # Current prediction
        self.current_prediction = 0.0

    def _discretize_state(self, state: np.ndarray) -> tuple:
        """Discretize state for value function lookup."""
        return tuple(np.round(state, 2))

    def get_value(self, state: np.ndarray) -> float:
        """Get expected value of a state."""
        key = self._discretize_state(state)
        return self.state_values.get(key, 0.0)

    def predict_reward(self, state: np.ndarray, cues: Optional[np.ndarray] = None) -> RewardPrediction:
        """Generate prediction about upcoming reward."""
        expected = self.get_value(state)

        # Uncertainty from variance of past predictions
        if len(self.td_errors) > 10:
            uncertainty = np.std(list(self.td_errors)[-50:])
        else:
            uncertainty = 1.0

        self.current_prediction = expected

        return RewardPrediction(
            expected_value=expected,
            uncertainty=uncertainty,
            time_to_reward=1.0,  # Would be learned
            cue_strength=1.0 if cues is None else float(np.linalg.norm(cues))
        )

    def compute_prediction_error(
        self,
        actual_reward: float,
        current_state: np.ndarray,
        next_state: np.ndarray
    ) -> float:
        """Compute temporal difference prediction error.

        RPE = r + γV(s') - V(s)
        """
        current_value = self.get_value(current_state)
        next_value = self.get_value(next_state)

        # TD error
        td_error = actual_reward + self.discount_factor * next_value - current_value

        self.td_errors.append(td_error)
        return td_error

    def update_values(
        self,
        state: np.ndarray,
        td_error: float
    ) -> None:
        """Update value function using TD error."""
        key = self._discretize_state(state)

        # Update eligibility trace
        self.eligibility_traces[key] = 1.0

        # Update all states with active traces
        for s_key in list(self.eligibility_traces.keys()):
            trace = self.eligibility_traces[s_key]
            self.state_values[s_key] = self.state_values.get(s_key, 0.0) + \
                                        self.learning_rate * td_error * trace

            # Decay trace
            self.eligibility_traces[s_key] *= self.eligibility_decay
            if self.eligibility_traces[s_key] < 0.01:
                del self.eligibility_traces[s_key]

    def get_surprise(self, td_error: float) -> float:
        """Convert TD error to surprise signal (absolute magnitude)."""
        return abs(td_error)

    def get_valence(self, td_error: float) -> float:
        """Get valence (positive/negative) of prediction error."""
        return np.sign(td_error)


class IncentiveSalience:
    """Computes incentive salience - the "wanting" signal.

    Incentive salience is distinct from "liking" (hedonic pleasure).
    It represents the motivational pull toward reward-associated stimuli.

    Key properties:
    - Can dissociate from liking (wanting without liking, liking without wanting)
    - Modulated by internal state (hunger increases food wanting)
    - Attached to cues that predict reward
    - Can become pathologically strong (addiction)
    """

    def __init__(
        self,
        cue_dim: int,
        learning_rate: float = 0.1,
        decay_rate: float = 0.01
    ):
        self.cue_dim = cue_dim
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate

        # Salience associated with each cue pattern
        self.cue_salience: Dict[tuple, float] = {}

        # Current motivational state modulator
        self.state_modulator = 1.0

        # History of cue-reward associations
        self.association_history: deque = deque(maxlen=500)

    def _discretize_cue(self, cue: np.ndarray) -> tuple:
        """Discretize cue for lookup."""
        return tuple(np.round(cue, 1))

    def compute_salience(
        self,
        cue: np.ndarray,
        internal_state: Optional[Dict[str, float]] = None
    ) -> float:
        """Compute incentive salience of a cue.

        Args:
            cue: The stimulus cue
            internal_state: Current internal needs (hunger, thirst, etc.)

        Returns:
            Incentive salience ("wanting" level)
        """
        key = self._discretize_cue(cue)
        base_salience = self.cue_salience.get(key, 0.0)

        # Modulate by internal state
        if internal_state is not None:
            self.state_modulator = self._compute_state_modulation(internal_state)

        salience = base_salience * self.state_modulator
        return np.clip(salience, 0.0, 10.0)

    def _compute_state_modulation(self, internal_state: Dict[str, float]) -> float:
        """Compute how internal state modulates wanting."""
        modulation = 1.0

        # Deprivation increases wanting
        for need, level in internal_state.items():
            if level < 0.5:  # Below satiation
                deprivation = 1.0 - level
                modulation += 0.5 * deprivation

        return np.clip(modulation, 0.5, 3.0)

    def update_salience(
        self,
        cue: np.ndarray,
        reward: float,
        prediction_error: float
    ) -> None:
        """Update salience based on cue-reward pairing."""
        key = self._discretize_cue(cue)

        # Salience increases when cue predicts reward
        current = self.cue_salience.get(key, 0.0)

        # Learning driven by prediction error
        update = self.learning_rate * prediction_error
        self.cue_salience[key] = current + update

        # Keep bounded
        self.cue_salience[key] = np.clip(self.cue_salience[key], 0.0, 10.0)

        self.association_history.append((key, reward, prediction_error))

    def decay_salience(self) -> None:
        """Apply decay to all salience values."""
        for key in list(self.cue_salience.keys()):
            self.cue_salience[key] *= (1 - self.decay_rate)
            if self.cue_salience[key] < 0.01:
                del self.cue_salience[key]

    def get_most_wanted(self, cues: List[np.ndarray]) -> Tuple[int, float]:
        """Find which cue is most wanted."""
        if not cues:
            return -1, 0.0

        saliences = [self.compute_salience(c) for c in cues]
        max_idx = int(np.argmax(saliences))
        return max_idx, saliences[max_idx]


class BenefitCostEvaluator:
    """Evaluates benefit/cost ratio for actions.

    Dopamine potentiates sensitivity to benefits vs. costs.
    High dopamine: Benefits weighted more heavily
    Low dopamine: Costs weighted more heavily

    This explains:
    - Increased effort under high motivation
    - Decreased effort in depression (low dopamine)
    - Risk preferences varying with dopamine state
    """

    def __init__(
        self,
        default_dopamine_level: float = 0.5,
        benefit_sensitivity: float = 1.0,
        cost_sensitivity: float = 1.0
    ):
        self.dopamine_level = default_dopamine_level
        self.benefit_sensitivity = benefit_sensitivity
        self.cost_sensitivity = cost_sensitivity

        # Track decision history
        self.decisions: deque = deque(maxlen=200)

    def set_dopamine_level(self, level: float) -> None:
        """Set current tonic dopamine level."""
        self.dopamine_level = np.clip(level, 0.0, 1.0)

    def evaluate(
        self,
        benefits: List[float],
        costs: List[float],
        probabilities: Optional[List[float]] = None
    ) -> float:
        """Evaluate net value given benefits and costs.

        Args:
            benefits: List of potential benefits
            costs: List of potential costs
            probabilities: Probability of each outcome

        Returns:
            Net value (benefit - cost, modulated by dopamine)
        """
        if probabilities is None:
            probabilities = [1.0 / len(benefits)] * len(benefits)

        # Dopamine modulates benefit vs cost weighting
        # High DA: benefits weighted more
        # Low DA: costs weighted more
        benefit_weight = self.benefit_sensitivity * (0.5 + self.dopamine_level)
        cost_weight = self.cost_sensitivity * (1.5 - self.dopamine_level)

        expected_benefit = sum(b * p for b, p in zip(benefits, probabilities))
        expected_cost = sum(c * p for c, p in zip(costs, probabilities))

        net_value = benefit_weight * expected_benefit - cost_weight * expected_cost

        return net_value

    def compute_effort_willingness(
        self,
        reward_magnitude: float,
        effort_required: float
    ) -> float:
        """Compute willingness to exert effort for reward.

        Dopamine determines how much effort we're willing to expend.
        """
        # Base value calculation
        value_per_effort = reward_magnitude / (effort_required + 0.1)

        # Dopamine modulates effort tolerance
        # High DA: More willing to exert effort
        effort_tolerance = 0.5 + self.dopamine_level

        willingness = value_per_effort * effort_tolerance

        return np.clip(willingness, 0.0, 1.0)

    def record_decision(
        self,
        chosen_value: float,
        effort: float,
        outcome: float
    ) -> None:
        """Record a decision for learning."""
        self.decisions.append({
            'value': chosen_value,
            'effort': effort,
            'outcome': outcome,
            'dopamine': self.dopamine_level
        })


class DopamineSystem:
    """Complete dopamine system integrating all components.

    Combines:
    - Prediction error computation
    - Incentive salience
    - Benefit/cost evaluation

    Outputs dopamine signals that modulate:
    - Learning (via prediction error)
    - Motivation (via incentive salience)
    - Decision-making (via benefit/cost ratio)
    """

    def __init__(
        self,
        state_dim: int,
        cue_dim: int,
        learning_rate: float = 0.1
    ):
        self.state_dim = state_dim
        self.cue_dim = cue_dim

        # Components
        self.prediction_error = PredictionErrorComputer(learning_rate=learning_rate)
        self.incentive_salience = IncentiveSalience(cue_dim=cue_dim)
        self.benefit_cost = BenefitCostEvaluator()

        # Tonic dopamine level (baseline motivation)
        self.tonic_level = 0.5

        # Phasic signal history
        self.phasic_history: deque = deque(maxlen=500)

        # Current internal state (needs)
        self.internal_state: Dict[str, float] = {
            'energy': 0.7,
            'social': 0.6,
            'novelty': 0.5,
        }

    def process_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        cue: Optional[np.ndarray] = None
    ) -> DopamineSignal:
        """Process a state transition and generate dopamine signal.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            cue: Optional cue associated with reward

        Returns:
            Complete dopamine signal
        """
        # Compute prediction error
        rpe = self.prediction_error.compute_prediction_error(reward, state, next_state)
        self.prediction_error.update_values(state, rpe)

        # Update incentive salience if cue present
        if cue is not None:
            self.incentive_salience.update_salience(cue, reward, rpe)
            salience = self.incentive_salience.compute_salience(cue, self.internal_state)
        else:
            salience = 0.0

        # Compute benefit/cost for this outcome
        # Benefits: reward + future value
        # Costs: effort (approximated by action magnitude)
        future_value = self.prediction_error.get_value(next_state)
        benefit = reward + 0.9 * future_value
        cost = float(np.linalg.norm(action))

        bc_ratio = self.benefit_cost.evaluate([benefit], [cost])

        # Update tonic level based on recent phasic activity
        self._update_tonic_level(rpe)

        # Create signal
        signal = DopamineSignal(
            prediction_error=rpe,
            incentive_salience=salience,
            benefit_cost_ratio=bc_ratio,
            channel=DopamineChannel.PHASIC if abs(rpe) > 0.1 else DopamineChannel.TONIC
        )

        self.phasic_history.append(signal)

        return signal

    def _update_tonic_level(self, recent_rpe: float) -> None:
        """Update baseline dopamine level."""
        # Tonic level tracks average reward state
        if len(self.phasic_history) > 10:
            recent_rpes = [s.prediction_error for s in list(self.phasic_history)[-20:]]
            mean_rpe = np.mean(recent_rpes)

            # Positive average RPE increases tonic level
            self.tonic_level += 0.01 * mean_rpe
            self.tonic_level = np.clip(self.tonic_level, 0.1, 0.9)

            # Update benefit/cost evaluator
            self.benefit_cost.set_dopamine_level(self.tonic_level)

    def get_motivation_level(self) -> float:
        """Get current overall motivation level."""
        # Combine tonic level with recent salience
        if self.phasic_history:
            recent_salience = np.mean([
                s.incentive_salience for s in list(self.phasic_history)[-10:]
            ])
        else:
            recent_salience = 0.0

        return 0.7 * self.tonic_level + 0.3 * min(recent_salience, 1.0)

    def get_exploration_bonus(self) -> float:
        """Get bonus for exploration based on dopamine state.

        Dopamine is the "neuromodulator of exploration".
        """
        # Higher tonic = more exploration
        exploration_drive = self.tonic_level

        # Also boost exploration when predictions uncertain
        if len(self.phasic_history) > 10:
            rpe_variance = np.var([s.prediction_error for s in list(self.phasic_history)[-20:]])
            uncertainty_bonus = min(rpe_variance, 0.5)
            exploration_drive += 0.5 * uncertainty_bonus

        return exploration_drive

    def update_internal_state(self, state_name: str, value: float) -> None:
        """Update an internal need state."""
        self.internal_state[state_name] = np.clip(value, 0.0, 1.0)

    def get_state_summary(self) -> Dict[str, float]:
        """Get summary of dopamine system state."""
        return {
            'tonic_level': self.tonic_level,
            'motivation': self.get_motivation_level(),
            'exploration_bonus': self.get_exploration_bonus(),
            'recent_rpe_mean': np.mean([
                s.prediction_error for s in list(self.phasic_history)[-20:]
            ]) if self.phasic_history else 0.0,
            'recent_salience_mean': np.mean([
                s.incentive_salience for s in list(self.phasic_history)[-20:]
            ]) if self.phasic_history else 0.0,
        }

    def reset(self) -> None:
        """Reset the dopamine system."""
        self.prediction_error = PredictionErrorComputer()
        self.incentive_salience = IncentiveSalience(cue_dim=self.cue_dim)
        self.benefit_cost = BenefitCostEvaluator()
        self.tonic_level = 0.5
        self.phasic_history.clear()
