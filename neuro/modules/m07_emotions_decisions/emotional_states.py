"""
Emotional States: Reactions to outcomes.

Based on research distinguishing emotional from motivational states:
- Emotional: Reactions when rewards ARE or ARE NOT received (AFTER outcome)
- Motivational: Goal-directed actions to obtain rewards/avoid punishments (BEFORE outcome)
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from enum import Enum
import numpy as np


class EmotionCategory(Enum):
    """Categories of emotional responses."""
    JOY = "joy"
    SATISFACTION = "satisfaction"
    DISAPPOINTMENT = "disappointment"
    FRUSTRATION = "frustration"
    FEAR = "fear"
    ANGER = "anger"
    RELIEF = "relief"
    SURPRISE = "surprise"
    SADNESS = "sadness"
    NEUTRAL = "neutral"


class OutcomeType(Enum):
    """Types of outcomes that generate emotions."""
    REWARD_RECEIVED = "reward_received"
    REWARD_OMITTED = "reward_omitted"
    PUNISHMENT_RECEIVED = "punishment_received"
    PUNISHMENT_AVOIDED = "punishment_avoided"
    UNEXPECTED_REWARD = "unexpected_reward"
    UNEXPECTED_PUNISHMENT = "unexpected_punishment"


@dataclass
class Outcome:
    """Represents an outcome event."""
    outcome_type: OutcomeType
    magnitude: float  # 0-1
    expected: bool
    context: Dict[str, Any] = None

    def __post_init__(self):
        if self.context is None:
            self.context = {}


@dataclass
class EmotionalResponse:
    """
    Emotional reaction to an outcome.

    Characterized by:
    - Valence: positive (pleasant) to negative (unpleasant)
    - Arousal: calm to excited
    - Specific emotion category
    """
    valence: float  # -1 to 1
    arousal: float  # 0 to 1
    emotion_type: EmotionCategory
    intensity: float = 1.0  # Overall intensity
    duration: float = 1.0   # Expected duration
    trigger: Optional[Outcome] = None

    def decay(self, rate: float = 0.1) -> 'EmotionalResponse':
        """Return decayed version of this emotion."""
        return EmotionalResponse(
            valence=self.valence * (1 - rate),
            arousal=self.arousal * (1 - rate),
            emotion_type=self.emotion_type,
            intensity=self.intensity * (1 - rate),
            duration=self.duration,
            trigger=self.trigger
        )


class OutcomeEvaluator:
    """
    Generates emotions from outcomes.

    Maps different outcome types to appropriate emotional responses:
    - Reward received → joy/satisfaction
    - Reward omitted → disappointment/frustration
    - Punishment received → fear/pain/anger
    - Punishment avoided → relief
    """

    def __init__(self, sensitivity: float = 1.0):
        self.sensitivity = sensitivity  # Emotional reactivity
        self.expectation_weight = 0.5   # How much expectations matter
        self.history: List[EmotionalResponse] = []

    def evaluate(self, outcome: Outcome) -> EmotionalResponse:
        """Generate emotional response to an outcome."""
        if outcome.outcome_type == OutcomeType.REWARD_RECEIVED:
            return self._reward_received(outcome)
        elif outcome.outcome_type == OutcomeType.REWARD_OMITTED:
            return self._reward_omitted(outcome)
        elif outcome.outcome_type == OutcomeType.PUNISHMENT_RECEIVED:
            return self._punishment_received(outcome)
        elif outcome.outcome_type == OutcomeType.PUNISHMENT_AVOIDED:
            return self._punishment_avoided(outcome)
        elif outcome.outcome_type == OutcomeType.UNEXPECTED_REWARD:
            return self._unexpected_reward(outcome)
        elif outcome.outcome_type == OutcomeType.UNEXPECTED_PUNISHMENT:
            return self._unexpected_punishment(outcome)
        else:
            return EmotionalResponse(
                valence=0.0,
                arousal=0.0,
                emotion_type=EmotionCategory.NEUTRAL,
                trigger=outcome
            )

    def _reward_received(self, outcome: Outcome) -> EmotionalResponse:
        """Generate joy/satisfaction from receiving reward."""
        base_valence = outcome.magnitude * self.sensitivity

        if outcome.expected:
            # Expected reward: satisfaction (calmer)
            emotion = EmotionCategory.SATISFACTION
            arousal = 0.3 + outcome.magnitude * 0.3
        else:
            # Unexpected reward: joy/surprise (more arousing)
            emotion = EmotionCategory.JOY
            arousal = 0.5 + outcome.magnitude * 0.4

        response = EmotionalResponse(
            valence=base_valence,
            arousal=arousal,
            emotion_type=emotion,
            intensity=outcome.magnitude * self.sensitivity,
            trigger=outcome
        )
        self.history.append(response)
        return response

    def _reward_omitted(self, outcome: Outcome) -> EmotionalResponse:
        """Generate disappointment/frustration from missing expected reward."""
        # Omission of expected reward is negative
        valence = -outcome.magnitude * self.sensitivity * self.expectation_weight

        if outcome.magnitude > 0.7:
            # Large expected reward missed: frustration (more active)
            emotion = EmotionCategory.FRUSTRATION
            arousal = 0.6
        else:
            # Smaller missed reward: disappointment (more passive)
            emotion = EmotionCategory.DISAPPOINTMENT
            arousal = 0.4

        response = EmotionalResponse(
            valence=valence,
            arousal=arousal,
            emotion_type=emotion,
            intensity=outcome.magnitude * self.sensitivity,
            trigger=outcome
        )
        self.history.append(response)
        return response

    def _punishment_received(self, outcome: Outcome) -> EmotionalResponse:
        """Generate fear/anger from receiving punishment."""
        valence = -outcome.magnitude * self.sensitivity

        controllability = outcome.context.get('controllable', True)

        if controllability:
            # Controllable punishment: anger (approach motivation)
            emotion = EmotionCategory.ANGER
            arousal = 0.7 + outcome.magnitude * 0.2
        else:
            # Uncontrollable punishment: fear/sadness
            if outcome.expected:
                emotion = EmotionCategory.SADNESS
                arousal = 0.3
            else:
                emotion = EmotionCategory.FEAR
                arousal = 0.8 + outcome.magnitude * 0.2

        response = EmotionalResponse(
            valence=valence,
            arousal=arousal,
            emotion_type=emotion,
            intensity=outcome.magnitude * self.sensitivity,
            trigger=outcome
        )
        self.history.append(response)
        return response

    def _punishment_avoided(self, outcome: Outcome) -> EmotionalResponse:
        """Generate relief from avoiding expected punishment."""
        # Avoided punishment is positive
        valence = outcome.magnitude * self.sensitivity * 0.8

        response = EmotionalResponse(
            valence=valence,
            arousal=0.4,  # Relief is moderately arousing
            emotion_type=EmotionCategory.RELIEF,
            intensity=outcome.magnitude * self.sensitivity,
            trigger=outcome
        )
        self.history.append(response)
        return response

    def _unexpected_reward(self, outcome: Outcome) -> EmotionalResponse:
        """Generate surprise + joy from unexpected reward."""
        valence = outcome.magnitude * self.sensitivity * 1.2  # Bonus for surprise

        response = EmotionalResponse(
            valence=valence,
            arousal=0.7 + outcome.magnitude * 0.3,  # Surprise is arousing
            emotion_type=EmotionCategory.SURPRISE,
            intensity=outcome.magnitude * self.sensitivity,
            trigger=outcome
        )
        self.history.append(response)
        return response

    def _unexpected_punishment(self, outcome: Outcome) -> EmotionalResponse:
        """Generate shock/fear from unexpected punishment."""
        valence = -outcome.magnitude * self.sensitivity * 1.5  # Worse when unexpected

        response = EmotionalResponse(
            valence=valence,
            arousal=0.9,  # Very arousing
            emotion_type=EmotionCategory.FEAR,
            intensity=outcome.magnitude * self.sensitivity,
            trigger=outcome
        )
        self.history.append(response)
        return response


class EmotionalDynamics:
    """
    Models how emotions evolve over time.

    Key behaviors:
    - Emotions decay toward baseline
    - Active regulation can speed decay
    - Emotional inertia (current emotion affects next)
    """

    def __init__(
        self,
        baseline_valence: float = 0.0,
        baseline_arousal: float = 0.3,
        decay_rate: float = 0.1
    ):
        self.baseline = EmotionalResponse(
            valence=baseline_valence,
            arousal=baseline_arousal,
            emotion_type=EmotionCategory.NEUTRAL
        )
        self.current_emotion = self.baseline
        self.decay_rate = decay_rate
        self.time = 0

        # Emotion history for inertia
        self.emotion_history: List[EmotionalResponse] = []

    def set_emotion(self, emotion: EmotionalResponse):
        """Set current emotional state."""
        self.emotion_history.append(self.current_emotion)
        self.current_emotion = emotion

    def step(self, new_outcome: Optional[Outcome] = None, evaluator: Optional[OutcomeEvaluator] = None):
        """
        Advance time by one step.

        If new outcome provided, generate new emotion.
        Otherwise, decay current emotion toward baseline.
        """
        self.time += 1

        if new_outcome and evaluator:
            new_emotion = evaluator.evaluate(new_outcome)
            # Blend with current emotion (emotional inertia)
            self.current_emotion = self._blend_emotions(
                self.current_emotion,
                new_emotion,
                weight_new=0.7
            )
        else:
            # Decay toward baseline
            self.current_emotion = self._decay_toward_baseline()

        self.emotion_history.append(self.current_emotion)

    def _decay_toward_baseline(self) -> EmotionalResponse:
        """Decay current emotion toward baseline."""
        decay = self.decay_rate

        new_valence = self.current_emotion.valence * (1 - decay) + \
                      self.baseline.valence * decay
        new_arousal = self.current_emotion.arousal * (1 - decay) + \
                      self.baseline.arousal * decay
        new_intensity = self.current_emotion.intensity * (1 - decay)

        # Determine emotion type based on remaining intensity
        if new_intensity < 0.1:
            emotion_type = EmotionCategory.NEUTRAL
        else:
            emotion_type = self.current_emotion.emotion_type

        return EmotionalResponse(
            valence=new_valence,
            arousal=new_arousal,
            emotion_type=emotion_type,
            intensity=new_intensity,
            duration=self.current_emotion.duration
        )

    def _blend_emotions(
        self,
        current: EmotionalResponse,
        new: EmotionalResponse,
        weight_new: float = 0.5
    ) -> EmotionalResponse:
        """Blend two emotions (emotional inertia)."""
        weight_current = 1 - weight_new

        blended_valence = current.valence * weight_current + new.valence * weight_new
        blended_arousal = current.arousal * weight_current + new.arousal * weight_new
        blended_intensity = current.intensity * weight_current + new.intensity * weight_new

        # New emotion determines type if stronger
        if new.intensity * weight_new > current.intensity * weight_current:
            emotion_type = new.emotion_type
        else:
            emotion_type = current.emotion_type

        return EmotionalResponse(
            valence=blended_valence,
            arousal=blended_arousal,
            emotion_type=emotion_type,
            intensity=blended_intensity,
            trigger=new.trigger
        )

    def emotion_regulation(self, strategy: str) -> EmotionalResponse:
        """
        Apply emotion regulation strategy.

        Strategies:
        - 'reappraisal': Change interpretation (reduces intensity)
        - 'suppression': Suppress expression (reduces arousal, not valence)
        - 'distraction': Shift attention (accelerates decay)
        - 'acceptance': Accept emotion (slight arousal reduction)
        """
        current = self.current_emotion

        if strategy == 'reappraisal':
            # Cognitive reappraisal: reduce intensity toward baseline
            regulated = EmotionalResponse(
                valence=current.valence * 0.5,
                arousal=current.arousal * 0.6,
                emotion_type=current.emotion_type,
                intensity=current.intensity * 0.5
            )
        elif strategy == 'suppression':
            # Suppression: reduce arousal but valence lingers
            regulated = EmotionalResponse(
                valence=current.valence * 0.9,
                arousal=current.arousal * 0.5,
                emotion_type=current.emotion_type,
                intensity=current.intensity * 0.8
            )
        elif strategy == 'distraction':
            # Distraction: accelerate decay
            regulated = self._decay_toward_baseline()
            regulated = regulated.decay(0.2)  # Extra decay
        elif strategy == 'acceptance':
            # Acceptance: slight calming, allows natural processing
            regulated = EmotionalResponse(
                valence=current.valence,
                arousal=current.arousal * 0.85,
                emotion_type=current.emotion_type,
                intensity=current.intensity * 0.95
            )
        else:
            regulated = current

        self.current_emotion = regulated
        return regulated

    def get_mood(self) -> Dict[str, float]:
        """
        Get current mood (aggregate of recent emotions).

        Mood is slower-changing than emotion.
        """
        if not self.emotion_history:
            return {'valence': 0.0, 'arousal': self.baseline.arousal}

        # Weight recent emotions more
        weights = np.exp(-np.arange(len(self.emotion_history)) * 0.1)[::-1]
        weights = weights / weights.sum()

        avg_valence = sum(
            e.valence * w for e, w in zip(self.emotion_history, weights)
        )
        avg_arousal = sum(
            e.arousal * w for e, w in zip(self.emotion_history, weights)
        )

        return {
            'valence': avg_valence,
            'arousal': avg_arousal,
            'history_length': len(self.emotion_history)
        }
