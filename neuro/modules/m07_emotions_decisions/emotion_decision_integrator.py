"""
Emotion Decision Integrator: Full emotional decision-making system.

Integrates all components:
- EmotionCircuit (VMPFC, Amygdala, ACC, Insula)
- DualRouteProcessor (fast/slow emotion pathways)
- MotivationalSystem (drives)
- OutcomeEvaluator (emotional reactions)
- MoralDilemmaProcessor (moral reasoning)
- VMPFCIntegrator (value computation)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np

from .emotion_circuit import EmotionCircuit, EmotionalEvaluation
from .dual_emotion_routes import DualRouteProcessor, EmotionRouteResponse, ResponseType
from .motivational_states import MotivationalSystem, IncentiveSalience
from .emotional_states import OutcomeEvaluator, EmotionalDynamics, Outcome
from .moral_reasoning import MoralDilemmaProcessor, MoralScenario, MoralDecision
from .value_computation import VMPFCIntegrator, OFCValueComputer, ValueSignal


class SituationType(Enum):
    """Types of situations requiring decision."""

    THREAT = "threat"
    REWARD = "reward"
    MORAL_DILEMMA = "moral_dilemma"
    SOCIAL = "social"
    NEUTRAL = "neutral"


@dataclass
class Situation:
    """
    A situation requiring emotional evaluation and decision.
    """

    stimulus: np.ndarray
    situation_type: SituationType
    context: Dict[str, Any] = field(default_factory=dict)
    options: List[str] = field(default_factory=list)
    moral_scenario: Optional[MoralScenario] = None

    def __post_init__(self):
        if not self.options:
            self.options = ["act", "dont_act"]


@dataclass
class Decision:
    """
    Result of emotional decision-making process.
    """

    action: str
    confidence: float
    value: float
    emotional_state: EmotionalEvaluation
    fast_response: Optional[EmotionRouteResponse]
    slow_response: Optional[EmotionRouteResponse]
    moral_decision: Optional[MoralDecision]
    motivation: Dict[str, Any]
    processing_time_ms: float
    reasoning: str


class EmotionDecisionSystem:
    """
    Complete emotional decision-making system.

    Integrates all components for realistic emotion-driven decisions.
    """

    def __init__(self):
        # Core components
        self.emotion_circuit = EmotionCircuit()
        self.dual_routes = DualRouteProcessor()
        self.motivational = MotivationalSystem()
        self.incentive = IncentiveSalience()
        self.outcome_evaluator = OutcomeEvaluator()
        self.emotional_dynamics = EmotionalDynamics()
        self.moral_processor = MoralDilemmaProcessor()
        self.value_integrator = VMPFCIntegrator()
        self.ofc = OFCValueComputer()

        # State
        self.current_emotional_state: Optional[EmotionalEvaluation] = None
        self.decision_history: List[Decision] = []

    def process_situation(self, situation: Situation) -> Decision:
        """
        Process a situation through the full emotional decision system.
        """

        # 1. Fast emotional evaluation (dual routes)
        fast_response, slow_response = self.dual_routes.process(
            situation.stimulus, situation.context
        )

        # 2. Full emotion circuit evaluation
        emotional_state = self.emotion_circuit.process(situation.stimulus, situation.context)
        self.current_emotional_state = emotional_state

        # 2b. Update emotional dynamics for mood tracking
        from .emotional_states import EmotionalResponse, EmotionCategory

        emotion_response = EmotionalResponse(
            valence=emotional_state.valence,
            arousal=emotional_state.arousal,
            emotion_type=EmotionCategory(emotional_state.emotion_type.value)
            if hasattr(emotional_state.emotion_type, "value")
            else EmotionCategory.NEUTRAL,
            intensity=emotional_state.arousal,
        )
        self.emotional_dynamics.set_emotion(emotion_response)

        # 3. Compute value
        base_value = self._compute_situation_value(situation, emotional_state)

        # 4. Check for moral dimension
        moral_decision = None
        if situation.situation_type == SituationType.MORAL_DILEMMA:
            if situation.moral_scenario:
                moral_decision = self.moral_processor.process_dilemma(situation.moral_scenario)
                base_value = self._incorporate_moral(base_value, moral_decision)

        # 5. Compute motivation
        motivation = self._compute_motivation(situation, emotional_state)

        # 6. Make decision
        action, confidence, reasoning = self._decide(
            situation,
            base_value,
            motivation,
            emotional_state,
            fast_response,
            slow_response,
            moral_decision,
        )

        # Processing time based on route
        if slow_response:
            processing_time = slow_response.latency_ms
        else:
            processing_time = fast_response.latency_ms if fast_response else 50.0

        decision = Decision(
            action=action,
            confidence=confidence,
            value=base_value,
            emotional_state=emotional_state,
            fast_response=fast_response,
            slow_response=slow_response,
            moral_decision=moral_decision,
            motivation=motivation,
            processing_time_ms=processing_time,
            reasoning=reasoning,
        )

        self.decision_history.append(decision)
        return decision

    def _compute_situation_value(
        self, situation: Situation, emotional_state: EmotionalEvaluation
    ) -> float:
        """Compute overall value of acting in situation."""
        # Base value from stimulus
        stimulus_value = np.mean(situation.stimulus)

        # Create value signal
        value_signal = ValueSignal(
            magnitude=stimulus_value,
            certainty=emotional_state.confidence,
            temporal_distance=0.0,
            emotional_weight=emotional_state.valence,
            social_weight=situation.context.get("social_relevance", 0.0),
            source="situation",
        )

        # Integrate with emotion and social context
        social_context = {
            "fairness": situation.context.get("fairness", 0.0),
            "reciprocity": situation.context.get("reciprocity", 0.0),
            "reputation_impact": situation.context.get("reputation", 0.0),
        }

        integrated_value = self.value_integrator.integrate(
            value_signal, emotional_state.valence, social_context
        )

        return integrated_value

    def _incorporate_moral(self, base_value: float, moral: MoralDecision) -> float:
        """Incorporate moral evaluation into value computation."""
        # Moral considerations can override pure value
        moral_weight = 0.5

        if moral.action_taken:
            # Action is permissible - combine with base value
            moral_value = moral.confidence * 0.5
        else:
            # Action not permissible - strong negative value
            moral_value = -moral.confidence * 0.8

        return base_value * (1 - moral_weight) + moral_value * moral_weight

    def _compute_motivation(
        self, situation: Situation, emotional_state: EmotionalEvaluation
    ) -> Dict[str, Any]:
        """Compute motivational state for situation."""
        # Approach vs avoid
        if emotional_state.threat_level > 0.5:
            avoid_strength = self.motivational.avoidance_tendency(emotional_state.threat_level)
            approach_strength = 0.0
        elif emotional_state.reward_level > 0.3:
            approach_strength = self.motivational.approach_tendency(emotional_state.reward_level)
            avoid_strength = 0.0
        else:
            approach_strength = emotional_state.reward_level * 0.5
            avoid_strength = emotional_state.threat_level * 0.5

        # Resolve conflict
        resolution = self.motivational.resolve_conflict(approach_strength, avoid_strength)

        return resolution

    def _decide(
        self,
        situation: Situation,
        value: float,
        motivation: Dict[str, Any],
        emotional_state: EmotionalEvaluation,
        fast_response: Optional[EmotionRouteResponse],
        slow_response: Optional[EmotionRouteResponse],
        moral_decision: Optional[MoralDecision],
    ) -> Tuple[str, float, str]:
        """Make final decision based on all inputs."""
        reasoning_parts = []

        # Start with motivation
        base_action = motivation["decision"]
        base_confidence = motivation["confidence"]
        reasoning_parts.append(f"Motivation: {base_action} (conf={base_confidence:.2f})")

        # THREAT situation type override - high threat = avoidance
        if situation.situation_type == SituationType.THREAT:
            if emotional_state.threat_level > 0.3 or emotional_state.valence < -0.2:
                reasoning_parts.append(
                    f"Threat situation (level={emotional_state.threat_level:.2f})"
                )
                return "avoid", 0.9, "; ".join(reasoning_parts)

        # Fast emotional override for threats
        if fast_response and fast_response.response_type == ResponseType.THREAT:
            if fast_response.intensity > 0.6:
                # Strong threat - immediate avoidance
                reasoning_parts.append("Fast threat response triggered")
                return "avoid", 0.9, "; ".join(reasoning_parts)

        # Moral override
        if moral_decision:
            if not moral_decision.action_taken and moral_decision.confidence > 0.7:
                reasoning_parts.append(f"Moral: {moral_decision.reasoning}")
                return "dont_act", moral_decision.confidence, "; ".join(reasoning_parts)

        # Value-based decision
        if value > 0.3:
            action = "act"
            confidence = min(0.9, value + 0.3)
            reasoning_parts.append(f"Value positive ({value:.2f})")
        elif value < -0.3:
            action = "dont_act"
            confidence = min(0.9, abs(value) + 0.3)
            reasoning_parts.append(f"Value negative ({value:.2f})")
        else:
            # Ambiguous - use motivation
            action = base_action if base_action != "freeze" else "dont_act"
            confidence = base_confidence
            reasoning_parts.append("Ambiguous value, using motivation")

        return action, confidence, "; ".join(reasoning_parts)

    def learn_from_outcome(self, decision: Decision, outcome: Outcome):
        """Update system based on experienced outcome."""
        # Generate emotional response to outcome
        emotional_response = self.outcome_evaluator.evaluate(outcome)

        # Update emotional dynamics
        self.emotional_dynamics.step(outcome, self.outcome_evaluator)

        # Update emotion circuit learning
        self.emotion_circuit.learn_from_outcome(
            stimulus=np.array([decision.value]),
            action=decision.action,
            outcome=outcome.magnitude,
            valence=emotional_response.valence,
            arousal=emotional_response.arousal,
        )

        # Update motivation based on outcome
        if outcome.magnitude > 0:
            self.motivational.record_reward(decision.action, outcome.magnitude)
        else:
            self.motivational.record_punishment(decision.action, abs(outcome.magnitude))

        # Update value learning
        self.ofc.update_value(decision.action, outcome.magnitude)

    def simulate_lesion(self, region: str):
        """
        Disable specific component to study effects.

        Regions: 'vmpfc', 'amygdala', 'acc', 'insula'
        """
        if region == "vmpfc":
            self.emotion_circuit.lesion_vmpfc()
            self.value_integrator.lesion()
            self.moral_processor.vmpfc_intact = False
        elif region == "amygdala":
            # Reduce amygdala sensitivity
            self.emotion_circuit.amygdala.threat_threshold = 0.9
            self.emotion_circuit.amygdala.reward_threshold = 0.9
        elif region == "acc":
            # Reduce error detection
            self.emotion_circuit.acc.error_threshold = 0.9
        elif region == "insula":
            # Reset body state influence
            self.emotion_circuit.insula.current_body_state.heart_rate = 0.5
            self.emotion_circuit.insula.current_body_state.muscle_tension = 0.0

    def restore_all(self):
        """Restore all components to normal function."""
        self.emotion_circuit.restore_vmpfc()
        self.value_integrator.restore()
        self.moral_processor.vmpfc_intact = True
        self.emotion_circuit.amygdala.threat_threshold = 0.5
        self.emotion_circuit.amygdala.reward_threshold = 0.5
        self.emotion_circuit.acc.error_threshold = 0.3

    def get_emotional_state(self) -> Optional[EmotionalEvaluation]:
        """Get current emotional state."""
        return self.current_emotional_state

    def get_mood(self) -> Dict[str, float]:
        """Get current mood (aggregate of recent emotions)."""
        return self.emotional_dynamics.get_mood()

    def set_stress_level(self, level: float):
        """Set stress level (affects fast/slow balance)."""
        self.dual_routes.set_stress_level(level)


# Convenience functions for creating test situations


def create_threat_situation(intensity: float = 0.7) -> Situation:
    """Create a threat situation for testing."""
    return Situation(
        stimulus=np.array([intensity, intensity * 0.8, intensity * 0.9, 0.1]),
        situation_type=SituationType.THREAT,
        context={"known_danger": True},
        options=["flee", "freeze", "fight"],
    )


def create_reward_situation(value: float = 0.6) -> Situation:
    """Create a reward situation for testing."""
    return Situation(
        stimulus=np.array([value, value * 0.9, value * 0.5, 0.8]),
        situation_type=SituationType.REWARD,
        context={"safe_environment": True},
        options=["approach", "ignore"],
    )


def create_moral_situation(scenario: MoralScenario) -> Situation:
    """Create a moral dilemma situation for testing."""
    return Situation(
        stimulus=np.array([0.5, 0.5, 0.5, 0.5]),
        situation_type=SituationType.MORAL_DILEMMA,
        context={"moral_weight": 1.0},
        options=["act", "dont_act"],
        moral_scenario=scenario,
    )
