"""
Value Computation: VMPFC/OFC value signals.

Based on research showing VMPFC/OFC computes subjective value that integrates:
- Objective magnitude
- Emotional coloring
- Social context (fairness, reciprocity)
- Temporal factors (delay discounting)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np


@dataclass
class ValueSignal:
    """
    Computed value of an option or outcome.

    VMPFC/OFC represents value as a common currency for decision-making.
    """
    magnitude: float           # Raw value magnitude
    certainty: float           # Probability/confidence (0-1)
    temporal_distance: float   # Delay in time units (0 = immediate)
    emotional_weight: float    # Emotional modulation (-1 to 1)
    social_weight: float       # Social context modulation (-1 to 1)
    source: str = "unknown"

    @property
    def expected_value(self) -> float:
        """Basic expected value (magnitude * certainty)."""
        return self.magnitude * self.certainty

    @property
    def subjective_value(self) -> float:
        """
        Full subjective value including all modulations.

        This is what VMPFC actually represents.
        """
        # Start with expected value
        ev = self.expected_value

        # Apply delay discounting
        if self.temporal_distance > 0:
            discount = 1 / (1 + 0.1 * self.temporal_distance)
            ev *= discount

        # Emotional modulation (can increase or decrease value)
        ev *= (1 + self.emotional_weight * 0.5)

        # Social modulation
        ev *= (1 + self.social_weight * 0.3)

        return ev


class OFCValueComputer:
    """
    Orbitofrontal Cortex value computation.

    Functions:
    - Compute expected value of outcomes
    - Delay discounting (future value reduction)
    - Compare options on common scale
    - Learn value from experience
    """

    def __init__(
        self,
        discount_rate: float = 0.1,
        learning_rate: float = 0.1
    ):
        self.discount_rate = discount_rate  # k in hyperbolic discounting
        self.learning_rate = learning_rate

        # Learned values
        self.option_values: Dict[str, float] = {}

        # Experience history
        self.experience_history: Dict[str, List[float]] = {}

    def compute_expected_value(
        self,
        outcome: float,
        probability: float
    ) -> float:
        """
        Compute expected value.

        EV = outcome * probability
        """
        return outcome * probability

    def delay_discount(self, value: float, delay: float) -> float:
        """
        Apply delay discounting.

        Uses hyperbolic discounting: V = A / (1 + k*D)
        Where A = amount, k = discount rate, D = delay
        """
        if delay <= 0:
            return value

        discounted = value / (1 + self.discount_rate * delay)
        return discounted

    def compare_options(
        self,
        options: List[Tuple[str, ValueSignal]]
    ) -> List[Tuple[str, float]]:
        """
        Compare multiple options on common value scale.

        Returns options ranked by subjective value.
        """
        ranked = [
            (name, signal.subjective_value)
            for name, signal in options
        ]

        # Sort by value (highest first)
        ranked.sort(key=lambda x: x[1], reverse=True)

        return ranked

    def update_value(self, option: str, outcome: float):
        """
        Learn value from experience.

        Uses simple running average with learning rate.
        """
        if option not in self.experience_history:
            self.experience_history[option] = []

        self.experience_history[option].append(outcome)

        if option in self.option_values:
            # Update existing value
            current = self.option_values[option]
            error = outcome - current
            self.option_values[option] = current + self.learning_rate * error
        else:
            # Initialize value
            self.option_values[option] = outcome

    def get_learned_value(self, option: str) -> Optional[float]:
        """Get learned value for an option."""
        return self.option_values.get(option)

    def compute_value_signal(
        self,
        option: str,
        magnitude: float,
        probability: float = 1.0,
        delay: float = 0.0
    ) -> ValueSignal:
        """
        Compute full value signal for an option.
        """
        # Get learned emotional association if exists
        learned = self.option_values.get(option, magnitude)
        emotional = (learned - magnitude) / (abs(magnitude) + 0.01)  # Relative difference

        return ValueSignal(
            magnitude=magnitude,
            certainty=probability,
            temporal_distance=delay,
            emotional_weight=np.clip(emotional, -1, 1),
            social_weight=0.0,  # Set separately
            source="ofc"
        )


class VMPFCIntegrator:
    """
    Integrates value with emotion and social context.

    VMPFC doesn't just compute objective value - it integrates:
    - Emotional coloring from amygdala
    - Social context (fairness, reciprocity, reputation)
    - Gut feelings from accumulated experience
    """

    def __init__(self):
        self.ofc = OFCValueComputer()

        # Social value weights
        self.fairness_weight = 0.3
        self.reciprocity_weight = 0.2
        self.reputation_weight = 0.2

        # Integration intact flag (for lesion simulation)
        self.intact = True

    def integrate(
        self,
        value: ValueSignal,
        emotion: float,
        social_context: Dict[str, Any]
    ) -> float:
        """
        Full VMPFC integration of value, emotion, and social context.

        Args:
            value: Base value signal from OFC
            emotion: Emotional valence (-1 to 1)
            social_context: Dict with fairness, reciprocity, etc.

        Returns:
            Integrated subjective value
        """
        if not self.intact:
            # Without VMPFC, just return raw expected value
            return value.expected_value

        # Start with subjective value
        base_value = value.subjective_value

        # Emotional modulation
        emotional_modulation = 1 + emotion * 0.4
        base_value *= emotional_modulation

        # Social modulations
        fairness = social_context.get('fairness', 0.0)  # -1 to 1
        reciprocity = social_context.get('reciprocity', 0.0)  # -1 to 1
        reputation = social_context.get('reputation_impact', 0.0)  # -1 to 1

        social_modulation = (
            1 +
            fairness * self.fairness_weight +
            reciprocity * self.reciprocity_weight +
            reputation * self.reputation_weight
        )
        base_value *= social_modulation

        return base_value

    def evaluate_fairness(self, my_share: float, their_share: float) -> float:
        """
        Evaluate fairness of an allocation.

        Humans show inequity aversion - prefer fair splits.
        """
        total = my_share + their_share
        if total == 0:
            return 0.0

        my_proportion = my_share / total
        fair_proportion = 0.5

        # Deviation from fair
        unfairness = abs(my_proportion - fair_proportion)

        # Both getting less than fair AND more than fair feel unfair
        # (though getting less feels worse)
        if my_proportion < fair_proportion:
            return -unfairness * 1.5  # Disadvantageous inequity
        else:
            return -unfairness * 0.5  # Advantageous inequity (still negative)

    def evaluate_reciprocity(
        self,
        my_past_actions: List[float],
        their_past_actions: List[float]
    ) -> float:
        """
        Evaluate reciprocity (tit-for-tat expectations).

        Returns positive if they've been reciprocal, negative if not.
        """
        if not my_past_actions or not their_past_actions:
            return 0.0

        my_avg = np.mean(my_past_actions)
        their_avg = np.mean(their_past_actions)

        # Reciprocity = their behavior matches mine
        difference = their_avg - my_avg

        # Positive if they match or exceed, negative if they defect
        return np.clip(difference, -1, 1)

    def ultimatum_response(
        self,
        offer: float,
        total: float
    ) -> Tuple[bool, float]:
        """
        Respond to ultimatum game offer.

        Demonstrates fairness preferences over pure utility maximization.
        """
        if total == 0:
            return False, 0.0

        proportion = offer / total
        fairness_eval = self.evaluate_fairness(offer, total - offer)

        # Subjective value of accepting
        accept_value = offer + fairness_eval * offer

        # Would a purely rational agent accept any positive offer?
        # Humans often reject unfair offers even at cost to themselves
        if accept_value > 0:
            accept = True
        else:
            accept = False

        return accept, accept_value

    def compute_regret(
        self,
        chosen_outcome: float,
        foregone_outcome: float
    ) -> float:
        """
        Compute anticipated/experienced regret.

        Regret = foregone - obtained (if positive)
        """
        regret = foregone_outcome - chosen_outcome
        return max(0, regret)

    def compute_relief(
        self,
        obtained_outcome: float,
        avoided_outcome: float
    ) -> float:
        """
        Compute relief from avoiding bad outcome.

        Relief = obtained - avoided (if negative avoided)
        """
        if avoided_outcome < 0:
            return obtained_outcome - avoided_outcome
        return 0.0

    def lesion(self):
        """Simulate VMPFC damage."""
        self.intact = False

    def restore(self):
        """Restore VMPFC function."""
        self.intact = True
