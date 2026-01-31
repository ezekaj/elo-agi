"""
Emotion Circuit: Brain emotion network implementation.

Based on research from: https://link.springer.com/article/10.1007/s00429-023-02644-9

Brain Emotion Circuit:
                  ┌─────────────────┐
                  │   VMPFC/OFC     │
                  │ (value signals) │
                  └────────┬────────┘
                           │
      ┌────────────────────┼────────────────────┐
      ▼                    ▼                    ▼
┌───────────────┐  ┌───────────────┐    ┌───────────────┐
│   Amygdala    │  │     ACC       │    │   Insula      │
│ (fast emotion)│  │ (action learn)│    │ (body states) │
└───────────────┘  └───────────────┘    └───────────────┘
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np


class EmotionType(Enum):
    """Basic emotion categories."""
    FEAR = "fear"
    ANGER = "anger"
    JOY = "joy"
    SADNESS = "sadness"
    DISGUST = "disgust"
    SURPRISE = "surprise"
    NEUTRAL = "neutral"


@dataclass
class EmotionalEvaluation:
    """Result of emotional processing."""
    valence: float  # -1 (negative) to 1 (positive)
    arousal: float  # 0 (calm) to 1 (excited)
    emotion_type: EmotionType
    threat_level: float = 0.0
    reward_level: float = 0.0
    confidence: float = 1.0
    source: str = "unknown"


@dataclass
class BodyState:
    """Interoceptive body state representation."""
    heart_rate: float = 0.5  # 0-1 normalized
    skin_conductance: float = 0.0
    muscle_tension: float = 0.0
    respiration_rate: float = 0.5
    temperature: float = 0.5


@dataclass
class EmotionalMemory:
    """Memory tagged with emotional valence."""
    content: np.ndarray
    valence: float
    arousal: float
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)


class VMPFC:
    """
    Ventromedial Prefrontal Cortex - value signals and gut feelings.

    Functions:
    - Computes subjective value of stimuli
    - Integrates emotion with cognition for decisions
    - Generates intuitive "gut feelings"
    """

    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.value_associations: Dict[int, float] = {}
        self.context_modulation: float = 1.0
        self.intact: bool = True

    def compute_value(self, stimulus: np.ndarray) -> float:
        """Compute subjective value of a stimulus."""
        if not self.intact:
            return 0.0

        stimulus_hash = hash(stimulus.tobytes())
        base_value = self.value_associations.get(stimulus_hash, 0.0)

        # Value computation includes learned associations
        # and intrinsic stimulus properties
        intrinsic = np.mean(stimulus) * 0.1
        return (base_value + intrinsic) * self.context_modulation

    def integrate_emotion_cognition(
        self,
        emotion: EmotionalEvaluation,
        context: Dict[str, Any]
    ) -> float:
        """Combine emotional and cognitive information for decision-making."""
        if not self.intact:
            # Without VMPFC, emotional input is reduced
            return emotion.valence * 0.2

        # Emotional valence affects decision value
        emotional_weight = emotion.arousal * emotion.valence

        # Context modulates the integration
        context_factor = context.get('importance', 1.0)
        social_factor = context.get('social_relevance', 1.0)

        return emotional_weight * context_factor * social_factor

    def generate_gut_feeling(self, situation: Dict[str, Any]) -> Tuple[float, float]:
        """
        Generate intuitive evaluation of a situation.

        Returns: (valence, confidence) tuple
        """
        if not self.intact:
            return (0.0, 0.0)  # No gut feelings without VMPFC

        # Aggregate past experiences with similar situations
        familiarity = situation.get('familiarity', 0.5)
        past_outcomes = situation.get('past_outcomes', [])

        if past_outcomes:
            avg_outcome = np.mean(past_outcomes)
            confidence = min(1.0, len(past_outcomes) / 10)
        else:
            avg_outcome = 0.0
            confidence = 0.2

        return (avg_outcome * familiarity, confidence)

    def update_value(self, stimulus: np.ndarray, outcome: float):
        """Learn value association from experience."""
        stimulus_hash = hash(stimulus.tobytes())
        current = self.value_associations.get(stimulus_hash, 0.0)
        self.value_associations[stimulus_hash] = (
            current + self.learning_rate * (outcome - current)
        )

    def lesion(self):
        """Simulate VMPFC damage."""
        self.intact = False

    def restore(self):
        """Restore VMPFC function."""
        self.intact = True


class Amygdala:
    """
    Amygdala - fast emotional processing.

    Functions:
    - Rapid threat detection (<12ms pathway)
    - Rapid reward detection
    - Emotional memory storage
    - Fear conditioning
    """

    def __init__(
        self,
        threat_threshold: float = 0.5,
        reward_threshold: float = 0.5,
        learning_rate: float = 0.2
    ):
        self.threat_threshold = threat_threshold
        self.reward_threshold = reward_threshold
        self.learning_rate = learning_rate

        # Learned threat/reward associations
        self.threat_associations: Dict[int, float] = {}
        self.reward_associations: Dict[int, float] = {}

        # Emotional memories
        self.emotional_memories: List[EmotionalMemory] = []

        # Conditioned responses
        self.conditioned_responses: Dict[int, Tuple[float, float]] = {}

    def detect_threat(self, stimulus: np.ndarray) -> float:
        """
        Rapid threat evaluation.

        This is the fast pathway (~12ms in biological systems).
        Low resolution but high speed - "better safe than sorry".
        """
        stimulus_hash = hash(stimulus.tobytes())

        # Check learned associations
        learned_threat = self.threat_associations.get(stimulus_hash, 0.0)

        # Check conditioned responses
        conditioned = self.conditioned_responses.get(stimulus_hash, (0.0, 0.0))
        conditioned_threat = conditioned[0]

        # Innate threat features (simplified)
        # High contrast, sudden movement, looming patterns
        innate_threat = self._compute_innate_threat(stimulus)

        # Combine all threat signals
        total_threat = max(learned_threat, conditioned_threat, innate_threat)

        return total_threat

    def detect_reward(self, stimulus: np.ndarray) -> float:
        """Rapid reward evaluation."""
        stimulus_hash = hash(stimulus.tobytes())

        learned_reward = self.reward_associations.get(stimulus_hash, 0.0)
        conditioned = self.conditioned_responses.get(stimulus_hash, (0.0, 0.0))
        conditioned_reward = conditioned[1]

        return max(learned_reward, conditioned_reward)

    def _compute_innate_threat(self, stimulus: np.ndarray) -> float:
        """Compute innate threat level from stimulus features."""
        # Simplified: high variance and extreme values indicate threat
        variance = np.var(stimulus)
        extremity = np.max(np.abs(stimulus))
        return min(1.0, (variance + extremity) / 2)

    def store_emotional_memory(
        self,
        content: np.ndarray,
        valence: float,
        arousal: float,
        context: Dict[str, Any] = None
    ):
        """Tag memory with emotional valence for storage."""
        memory = EmotionalMemory(
            content=content.copy(),
            valence=valence,
            arousal=arousal,
            timestamp=len(self.emotional_memories),
            context=context or {}
        )
        self.emotional_memories.append(memory)

        # High arousal events strengthen storage
        if arousal > 0.7:
            self._strengthen_associations(content, valence)

    def _strengthen_associations(self, content: np.ndarray, valence: float):
        """Strengthen threat/reward associations for high arousal events."""
        content_hash = hash(content.tobytes())
        if valence < 0:
            current = self.threat_associations.get(content_hash, 0.0)
            self.threat_associations[content_hash] = min(1.0, current + 0.2)
        else:
            current = self.reward_associations.get(content_hash, 0.0)
            self.reward_associations[content_hash] = min(1.0, current + 0.2)

    def fear_conditioning(
        self,
        conditioned_stimulus: np.ndarray,
        unconditioned_stimulus_threat: float
    ):
        """
        Classical fear conditioning.

        CS (neutral) paired with US (threat) → CS becomes threatening.
        """
        cs_hash = hash(conditioned_stimulus.tobytes())
        current = self.conditioned_responses.get(cs_hash, (0.0, 0.0))

        new_threat = current[0] + self.learning_rate * (
            unconditioned_stimulus_threat - current[0]
        )
        self.conditioned_responses[cs_hash] = (new_threat, current[1])

    def extinction(self, stimulus: np.ndarray, rate: float = 0.1):
        """Gradual reduction of conditioned response."""
        stimulus_hash = hash(stimulus.tobytes())
        if stimulus_hash in self.conditioned_responses:
            current = self.conditioned_responses[stimulus_hash]
            new_threat = current[0] * (1 - rate)
            new_reward = current[1] * (1 - rate)
            self.conditioned_responses[stimulus_hash] = (new_threat, new_reward)

    def process(self, stimulus: np.ndarray) -> EmotionalEvaluation:
        """Full amygdala processing of a stimulus."""
        threat = self.detect_threat(stimulus)
        reward = self.detect_reward(stimulus)

        # Determine dominant emotion
        if threat > reward and threat > self.threat_threshold:
            emotion_type = EmotionType.FEAR
            valence = -threat
            arousal = threat
        elif reward > self.reward_threshold:
            emotion_type = EmotionType.JOY
            valence = reward
            arousal = reward * 0.8
        else:
            emotion_type = EmotionType.NEUTRAL
            valence = reward - threat
            arousal = max(threat, reward)

        return EmotionalEvaluation(
            valence=valence,
            arousal=arousal,
            emotion_type=emotion_type,
            threat_level=threat,
            reward_level=reward,
            source="amygdala"
        )


class ACC:
    """
    Anterior Cingulate Cortex - action-emotion learning.

    Functions:
    - Monitor action outcomes
    - Detect prediction errors
    - Adjust motivation based on errors
    """

    def __init__(self, error_threshold: float = 0.3, learning_rate: float = 0.15):
        self.error_threshold = error_threshold
        self.learning_rate = learning_rate

        # Action-outcome associations
        self.action_outcomes: Dict[str, List[float]] = {}

        # Expected values
        self.expected_values: Dict[str, float] = {}

        # Motivation adjustments
        self.motivation_modulation: float = 1.0

    def monitor_outcomes(self, action: str, result: float):
        """Track action-outcome associations."""
        if action not in self.action_outcomes:
            self.action_outcomes[action] = []

        self.action_outcomes[action].append(result)

        # Update expected value
        if action in self.expected_values:
            expected = self.expected_values[action]
            error = result - expected
            self.expected_values[action] = expected + self.learning_rate * error
        else:
            self.expected_values[action] = result

    def detect_error(self, expected: float, actual: float) -> float:
        """
        Compute prediction error for emotions.

        Positive error = better than expected
        Negative error = worse than expected
        """
        error = actual - expected
        return error

    def adjust_motivation(self, error: float) -> float:
        """Update motivation drive based on prediction error."""
        if abs(error) > self.error_threshold:
            # Large errors trigger motivation adjustment
            if error > 0:
                # Better than expected - increase motivation
                self.motivation_modulation = min(2.0, self.motivation_modulation + 0.1)
            else:
                # Worse than expected - may decrease motivation
                self.motivation_modulation = max(0.5, self.motivation_modulation - 0.1)

        return self.motivation_modulation

    def get_expected_value(self, action: str) -> float:
        """Get expected outcome for an action."""
        return self.expected_values.get(action, 0.0)

    def conflict_detection(self, responses: List[Tuple[str, float]]) -> float:
        """
        Detect conflict between multiple response options.

        Returns conflict level (0-1).
        """
        if len(responses) < 2:
            return 0.0

        values = [r[1] for r in responses]

        # Conflict when multiple options have similar high values
        sorted_values = sorted(values, reverse=True)
        if len(sorted_values) >= 2:
            top_diff = sorted_values[0] - sorted_values[1]
            conflict = max(0.0, 1.0 - top_diff * 2)
        else:
            conflict = 0.0

        return conflict


class Insula:
    """
    Insula - interoceptive awareness and body states.

    Functions:
    - Read body state
    - Map body states to emotions
    - Generate disgust response
    - Empathy signaling
    """

    def __init__(self):
        self.current_body_state = BodyState()

        # Body-emotion mappings (simplified)
        self.body_emotion_map = {
            'high_heart_high_tension': EmotionType.FEAR,
            'high_heart_low_tension': EmotionType.JOY,
            'low_heart_high_tension': EmotionType.ANGER,
            'low_heart_low_tension': EmotionType.SADNESS,
        }

    def read_body_state(self) -> BodyState:
        """Get current physiological state."""
        return self.current_body_state

    def update_body_state(
        self,
        heart_rate: Optional[float] = None,
        skin_conductance: Optional[float] = None,
        muscle_tension: Optional[float] = None,
        respiration_rate: Optional[float] = None,
        temperature: Optional[float] = None
    ):
        """Update body state readings."""
        if heart_rate is not None:
            self.current_body_state.heart_rate = np.clip(heart_rate, 0, 1)
        if skin_conductance is not None:
            self.current_body_state.skin_conductance = np.clip(skin_conductance, 0, 1)
        if muscle_tension is not None:
            self.current_body_state.muscle_tension = np.clip(muscle_tension, 0, 1)
        if respiration_rate is not None:
            self.current_body_state.respiration_rate = np.clip(respiration_rate, 0, 1)
        if temperature is not None:
            self.current_body_state.temperature = np.clip(temperature, 0, 1)

    def map_to_emotion(self, body_state: Optional[BodyState] = None) -> EmotionalEvaluation:
        """Map body state to emotion (interoception → feeling)."""
        state = body_state or self.current_body_state

        # Simplified mapping based on heart rate and muscle tension
        hr_level = 'high' if state.heart_rate > 0.6 else 'low'
        tension_level = 'high' if state.muscle_tension > 0.5 else 'low'

        key = f'{hr_level}_heart_{tension_level}_tension'
        emotion_type = self.body_emotion_map.get(key, EmotionType.NEUTRAL)

        # Compute arousal from overall activation
        arousal = (state.heart_rate + state.skin_conductance +
                   state.muscle_tension + state.respiration_rate) / 4

        # Valence derived from emotion type
        valence_map = {
            EmotionType.FEAR: -0.8,
            EmotionType.ANGER: -0.6,
            EmotionType.SADNESS: -0.5,
            EmotionType.JOY: 0.8,
            EmotionType.NEUTRAL: 0.0,
        }
        valence = valence_map.get(emotion_type, 0.0)

        return EmotionalEvaluation(
            valence=valence,
            arousal=arousal,
            emotion_type=emotion_type,
            source="insula"
        )

    def disgust_response(self, stimulus: np.ndarray) -> float:
        """
        Generate visceral rejection response.

        Returns disgust level (0-1).
        """
        # Simplified: certain patterns trigger disgust
        # In reality, learned and innate disgust triggers
        pattern_irregularity = np.std(stimulus)

        # Disgust often accompanied by nausea-like body response
        if pattern_irregularity > 0.5:
            self.update_body_state(
                skin_conductance=0.7,
                muscle_tension=0.6
            )
            return min(1.0, pattern_irregularity)
        return 0.0

    def empathy_signal(self, other_body_state: BodyState) -> EmotionalEvaluation:
        """
        Mirror another's body state for empathy.

        Simulates how observing others' states affects our own.
        """
        # Partially adopt observed body state
        mirror_strength = 0.3

        mirrored = BodyState(
            heart_rate=self.current_body_state.heart_rate * (1 - mirror_strength) +
                       other_body_state.heart_rate * mirror_strength,
            skin_conductance=self.current_body_state.skin_conductance * (1 - mirror_strength) +
                            other_body_state.skin_conductance * mirror_strength,
            muscle_tension=self.current_body_state.muscle_tension * (1 - mirror_strength) +
                          other_body_state.muscle_tension * mirror_strength,
            respiration_rate=self.current_body_state.respiration_rate * (1 - mirror_strength) +
                            other_body_state.respiration_rate * mirror_strength,
            temperature=self.current_body_state.temperature
        )

        return self.map_to_emotion(mirrored)


class EmotionCircuit:
    """
    Integrated emotion processing network.

    Combines VMPFC, Amygdala, ACC, and Insula into a unified system.
    """

    def __init__(self):
        self.vmpfc = VMPFC()
        self.amygdala = Amygdala()
        self.acc = ACC()
        self.insula = Insula()

    def process(self, stimulus: np.ndarray, context: Dict[str, Any] = None) -> EmotionalEvaluation:
        """
        Full emotional evaluation of a stimulus.

        Integrates all components of the emotion circuit.
        """
        context = context or {}

        # Amygdala: fast threat/reward
        amygdala_eval = self.amygdala.process(stimulus)

        # Insula: body state awareness
        body_eval = self.insula.map_to_emotion()

        # VMPFC: value computation and integration
        value = self.vmpfc.compute_value(stimulus)
        gut_feeling, gut_confidence = self.vmpfc.generate_gut_feeling(context)

        # ACC: check for conflicts
        conflict = self.acc.conflict_detection([
            ('amygdala', amygdala_eval.valence),
            ('body', body_eval.valence),
            ('value', value)
        ])

        # Integrate evaluations
        # Amygdala dominates when threat is high
        if amygdala_eval.threat_level > 0.7:
            weight_amygdala = 0.7
            weight_body = 0.2
            weight_value = 0.1
        else:
            weight_amygdala = 0.4
            weight_body = 0.3
            weight_value = 0.3

        integrated_valence = (
            amygdala_eval.valence * weight_amygdala +
            body_eval.valence * weight_body +
            value * weight_value
        )

        integrated_arousal = max(amygdala_eval.arousal, body_eval.arousal)

        # Determine final emotion type (amygdala dominates if high arousal)
        if amygdala_eval.arousal > 0.6:
            emotion_type = amygdala_eval.emotion_type
        elif body_eval.arousal > 0.5:
            emotion_type = body_eval.emotion_type
        else:
            emotion_type = EmotionType.NEUTRAL

        return EmotionalEvaluation(
            valence=integrated_valence,
            arousal=integrated_arousal,
            emotion_type=emotion_type,
            threat_level=amygdala_eval.threat_level,
            reward_level=amygdala_eval.reward_level,
            confidence=1.0 - conflict * 0.5,
            source="emotion_circuit"
        )

    def learn_from_outcome(
        self,
        stimulus: np.ndarray,
        action: str,
        outcome: float,
        valence: float,
        arousal: float
    ):
        """Update all components based on experienced outcome."""
        # VMPFC learns value
        self.vmpfc.update_value(stimulus, outcome)

        # ACC monitors outcome
        self.acc.monitor_outcomes(action, outcome)

        # Amygdala stores emotional memory
        self.amygdala.store_emotional_memory(
            content=stimulus,
            valence=valence,
            arousal=arousal
        )

    def lesion_vmpfc(self):
        """Simulate VMPFC damage (as in lesion studies)."""
        self.vmpfc.lesion()

    def restore_vmpfc(self):
        """Restore VMPFC function."""
        self.vmpfc.restore()
