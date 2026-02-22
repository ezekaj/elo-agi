"""Hot vs Cold Executive Functions

Hot EF: Orbital/medial PFC - emotionally-laden decisions, reward sensitivity
Cold EF: Lateral PFC - abstract reasoning, logical processing
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple


@dataclass
class HotColdParams:
    """Parameters for hot/cold EF systems"""

    n_units: int = 50
    emotion_weight: float = 0.5
    reward_sensitivity: float = 0.7
    risk_aversion: float = 0.5
    delay_discounting_rate: float = 0.1
    cognitive_load_effect: float = 0.3


class HotExecutiveFunction:
    """Hot executive function - orbital/medial PFC

    Handles emotionally-laden decisions, reward processing, social contexts
    """

    def __init__(self, params: Optional[HotColdParams] = None):
        self.params = params or HotColdParams()

        # OFC/vmPFC activation
        self.ofc_activation = np.zeros(self.params.n_units)
        self.vmpfc_activation = np.zeros(self.params.n_units)

        # Emotional state
        self.emotional_arousal = 0.5
        self.valence = 0.0  # -1 to 1 (negative to positive)

        # Reward history
        self.reward_history = []
        self.expected_reward = 0.5

    def process_emotional_stimulus(
        self, stimulus: np.ndarray, emotional_intensity: float = 0.5, valence: float = 0.0
    ) -> dict:
        """Process emotionally-laden stimulus

        Args:
            stimulus: Input stimulus
            emotional_intensity: How emotionally intense (0-1)
            valence: Emotional valence (-1 negative to +1 positive)
        """
        if len(stimulus) != self.params.n_units:
            stimulus = np.resize(stimulus, self.params.n_units)

        # Update emotional state
        self.emotional_arousal = 0.7 * self.emotional_arousal + 0.3 * emotional_intensity
        self.valence = 0.7 * self.valence + 0.3 * valence

        # OFC processes reward/punishment associations
        self.ofc_activation = stimulus * (1 + emotional_intensity * self.params.emotion_weight)
        self.ofc_activation = np.clip(self.ofc_activation, 0, 1)

        # vmPFC integrates with emotion
        self.vmpfc_activation = self.ofc_activation * (1 + valence * 0.3)
        self.vmpfc_activation = np.clip(self.vmpfc_activation, 0, 1)

        return {
            "ofc_activity": np.mean(self.ofc_activation),
            "vmpfc_activity": np.mean(self.vmpfc_activation),
            "emotional_arousal": self.emotional_arousal,
            "valence": self.valence,
        }

    def evaluate_reward(self, reward_magnitude: float, delay: float = 0.0) -> float:
        """Evaluate reward with temporal discounting

        Args:
            reward_magnitude: Size of reward
            delay: Time until reward received

        Returns:
            Subjective value of reward
        """
        # Hyperbolic discounting
        k = self.params.delay_discounting_rate
        discounted_value = reward_magnitude / (1 + k * delay)

        # Modulate by emotional state
        if self.valence > 0:
            discounted_value *= 1 + self.valence * 0.2
        else:
            discounted_value *= 1 + self.valence * 0.1  # Less impact of negative

        return discounted_value

    def make_risky_decision(
        self, safe_option: float, risky_option: Tuple[float, float, float]
    ) -> dict:
        """Make decision between safe and risky options

        Args:
            safe_option: Guaranteed reward
            risky_option: (high_reward, low_reward, prob_high)

        Returns:
            Decision and processing details
        """
        high_reward, low_reward, prob_high = risky_option

        # Expected value of risky option
        expected_risky = high_reward * prob_high + low_reward * (1 - prob_high)

        # Subjective value modified by risk aversion and emotional state
        # Higher arousal -> more risk-seeking (or risk-averse depending on valence)
        risk_modifier = 1 - self.params.risk_aversion
        if self.emotional_arousal > 0.7:
            if self.valence > 0:
                risk_modifier += 0.2  # Positive arousal -> more risk-seeking
            else:
                risk_modifier -= 0.2  # Negative arousal -> more risk-averse

        subjective_risky = expected_risky * risk_modifier

        # Choice
        choose_risky = subjective_risky > safe_option

        return {
            "choice": "risky" if choose_risky else "safe",
            "safe_value": safe_option,
            "risky_expected": expected_risky,
            "risky_subjective": subjective_risky,
            "risk_modifier": risk_modifier,
            "emotional_arousal": self.emotional_arousal,
        }

    def update_reward_learning(self, received_reward: float):
        """Update reward expectations"""
        self.reward_history.append(received_reward)
        # Running average
        self.expected_reward = 0.9 * self.expected_reward + 0.1 * received_reward

    def get_state(self) -> dict:
        """Get hot EF state"""
        return {
            "emotional_arousal": self.emotional_arousal,
            "valence": self.valence,
            "expected_reward": self.expected_reward,
            "ofc_mean": np.mean(self.ofc_activation),
            "vmpfc_mean": np.mean(self.vmpfc_activation),
        }


class ColdExecutiveFunction:
    """Cold executive function - lateral PFC

    Handles abstract reasoning, logical decisions, non-emotional contexts
    """

    def __init__(self, params: Optional[HotColdParams] = None):
        self.params = params or HotColdParams()

        # Lateral PFC activation
        self.dlpfc_activation = np.zeros(self.params.n_units)
        self.vlpfc_activation = np.zeros(self.params.n_units)

        # Cognitive load
        self.cognitive_load = 0.0

        # Rule representations
        self.active_rules = []
        self.rule_strength = {}

    def process_abstract_stimulus(self, stimulus: np.ndarray, rules: List[str] = None) -> dict:
        """Process stimulus according to abstract rules

        Args:
            stimulus: Input stimulus
            rules: List of rule names to apply
        """
        if len(stimulus) != self.params.n_units:
            stimulus = np.resize(stimulus, self.params.n_units)

        rules = rules or []

        # DLPFC processes rules
        rule_signal = np.zeros(self.params.n_units)
        for rule in rules:
            if rule not in self.rule_strength:
                self.rule_strength[rule] = 0.5
            rule_signal += np.random.rand(self.params.n_units) * self.rule_strength[rule]

        self.dlpfc_activation = np.clip(stimulus + rule_signal * 0.5, 0, 1)

        # VLPFC handles interference
        self.vlpfc_activation = self.dlpfc_activation * (1 - self.cognitive_load * 0.3)
        self.vlpfc_activation = np.clip(self.vlpfc_activation, 0, 1)

        self.active_rules = rules

        return {
            "dlpfc_activity": np.mean(self.dlpfc_activation),
            "vlpfc_activity": np.mean(self.vlpfc_activation),
            "cognitive_load": self.cognitive_load,
            "active_rules": rules,
        }

    def reason_logically(self, premises: List[np.ndarray], conclusion: np.ndarray) -> dict:
        """Evaluate logical conclusion from premises

        Args:
            premises: List of premise representations
            conclusion: Proposed conclusion representation

        Returns:
            Reasoning results
        """
        # Cognitive load increases with number of premises
        self.cognitive_load = min(1.0, len(premises) * 0.2)

        # Combine premises
        combined = np.zeros(self.params.n_units)
        for p in premises:
            if len(p) != self.params.n_units:
                p = np.resize(p, self.params.n_units)
            combined += p / len(premises)

        if len(conclusion) != self.params.n_units:
            conclusion = np.resize(conclusion, self.params.n_units)

        # Check similarity of conclusion to combined premises
        similarity = np.dot(combined, conclusion) / (
            np.linalg.norm(combined) * np.linalg.norm(conclusion) + 1e-8
        )

        # Accuracy decreases with cognitive load
        effective_threshold = 0.5 * (1 - self.cognitive_load * self.params.cognitive_load_effect)

        valid = similarity > effective_threshold

        return {
            "valid": valid,
            "confidence": similarity,
            "cognitive_load": self.cognitive_load,
            "threshold": effective_threshold,
        }

    def solve_problem(self, problem_complexity: float) -> dict:
        """Attempt to solve an abstract problem

        Args:
            problem_complexity: 0-1 difficulty rating

        Returns:
            Solution attempt results
        """
        # Update cognitive load
        self.cognitive_load = min(1.0, problem_complexity)

        # Success probability depends on available resources
        available_resources = 1 - self.cognitive_load * 0.5
        success_prob = available_resources * (1 - problem_complexity * 0.5)

        # Add noise
        success_prob = np.clip(success_prob + np.random.randn() * 0.1, 0, 1)

        solved = np.random.rand() < success_prob

        # Update rule strengths based on outcome
        for rule in self.active_rules:
            if solved:
                self.rule_strength[rule] = min(1.0, self.rule_strength.get(rule, 0.5) + 0.1)
            else:
                self.rule_strength[rule] = max(0.1, self.rule_strength.get(rule, 0.5) - 0.05)

        return {
            "solved": solved,
            "success_probability": success_prob,
            "cognitive_load": self.cognitive_load,
            "resources_available": available_resources,
        }

    def get_state(self) -> dict:
        """Get cold EF state"""
        return {
            "cognitive_load": self.cognitive_load,
            "active_rules": self.active_rules.copy(),
            "rule_strengths": self.rule_strength.copy(),
            "dlpfc_mean": np.mean(self.dlpfc_activation),
            "vlpfc_mean": np.mean(self.vlpfc_activation),
        }


class EmotionalRegulator:
    """Integrates hot and cold EF for emotional regulation

    Models how cognitive control regulates emotional responses
    """

    def __init__(self, params: Optional[HotColdParams] = None):
        self.params = params or HotColdParams()

        self.hot_ef = HotExecutiveFunction(params)
        self.cold_ef = ColdExecutiveFunction(params)

        # Regulation strategy
        self.regulation_strategy = "reappraisal"  # or "suppression"
        self.regulation_effort = 0.0

    def regulate_emotion(self, emotional_intensity: float, strategy: str = "reappraisal") -> dict:
        """Apply cognitive regulation to emotion

        Args:
            emotional_intensity: Initial emotional intensity
            strategy: 'reappraisal' (cognitive reframing) or 'suppression' (response inhibition)

        Returns:
            Regulation results
        """
        self.regulation_strategy = strategy

        if strategy == "reappraisal":
            # Reappraisal uses DLPFC to reframe meaning
            # More effective, less costly
            self.cold_ef.cognitive_load += 0.2
            regulation_success = min(0.8, self.cold_ef.dlpfc_activation.mean() + 0.3)
            regulated_intensity = emotional_intensity * (1 - regulation_success)
            cognitive_cost = 0.2

        else:  # suppression
            # Suppression uses more effort, less effective long-term
            self.regulation_effort += 0.3
            regulation_success = min(0.6, self.regulation_effort)
            regulated_intensity = emotional_intensity * (1 - regulation_success * 0.5)
            cognitive_cost = 0.4

        # Update hot EF
        self.hot_ef.emotional_arousal = regulated_intensity

        return {
            "strategy": strategy,
            "initial_intensity": emotional_intensity,
            "regulated_intensity": regulated_intensity,
            "regulation_success": regulation_success,
            "cognitive_cost": cognitive_cost,
            "cognitive_load": self.cold_ef.cognitive_load,
        }

    def make_hybrid_decision(
        self, stimulus: np.ndarray, emotional_content: float, logical_content: float
    ) -> dict:
        """Make decision using both hot and cold systems

        Args:
            stimulus: Input stimulus
            emotional_content: How emotional the decision (0-1)
            logical_content: How logical the decision (0-1)

        Returns:
            Decision results
        """
        # Hot processing
        hot_result = self.hot_ef.process_emotional_stimulus(
            stimulus, emotional_intensity=emotional_content, valence=0.0
        )

        # Cold processing
        cold_result = self.cold_ef.process_abstract_stimulus(stimulus)

        # Weight contributions
        hot_weight = emotional_content / (emotional_content + logical_content + 1e-8)
        cold_weight = logical_content / (emotional_content + logical_content + 1e-8)

        # Combined decision signal
        combined_activation = (
            hot_result["ofc_activity"] * hot_weight + cold_result["dlpfc_activity"] * cold_weight
        )

        return {
            "hot_contribution": hot_result["ofc_activity"] * hot_weight,
            "cold_contribution": cold_result["dlpfc_activity"] * cold_weight,
            "combined_activation": combined_activation,
            "hot_weight": hot_weight,
            "cold_weight": cold_weight,
            "emotional_arousal": self.hot_ef.emotional_arousal,
            "cognitive_load": self.cold_ef.cognitive_load,
        }

    def get_state(self) -> dict:
        """Get integrated state"""
        return {
            "hot": self.hot_ef.get_state(),
            "cold": self.cold_ef.get_state(),
            "regulation_strategy": self.regulation_strategy,
            "regulation_effort": self.regulation_effort,
        }
