"""
Time Perception Modulation

Factors that affect subjective time perception:
- Emotion: High arousal → time slows (more events processed)
- Attention: More attention → longer perceived duration
- Dopamine: Dopamine increase → time speeds up (faster clock)
- Age: Aging → subjective time acceleration
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum


class EmotionalState(Enum):
    """Emotional states affecting time perception"""

    NEUTRAL = "neutral"
    FEAR = "fear"  # High arousal, time slows
    EXCITEMENT = "excitement"  # High arousal, time slows
    BOREDOM = "boredom"  # Low arousal, time drags
    FLOW = "flow"  # Absorbed, time flies
    ANXIETY = "anxiety"  # High arousal, time distortion


@dataclass
class ModulationEffect:
    """Effect of a modulation factor on time perception"""

    factor_name: str
    distortion_ratio: float  # >1 = time feels longer, <1 = time feels shorter
    confidence_modifier: float  # How it affects confidence
    description: str


class EmotionalModulator:
    """Modulates time perception based on emotional state.

    High arousal states (fear, excitement) slow down subjective time
    because more information is processed per unit time.
    """

    def __init__(self):
        # Arousal-time mapping: high arousal = slower perceived time
        self.arousal_effects = {
            EmotionalState.NEUTRAL: (0.5, 1.0),  # (arousal, time_ratio)
            EmotionalState.FEAR: (0.9, 1.4),  # Time slows 40%
            EmotionalState.EXCITEMENT: (0.8, 1.3),  # Time slows 30%
            EmotionalState.BOREDOM: (0.2, 1.2),  # Time drags 20%
            EmotionalState.FLOW: (0.6, 0.7),  # Time flies, 30% faster
            EmotionalState.ANXIETY: (0.85, 1.35),  # Time slows 35%
        }

        self.current_state = EmotionalState.NEUTRAL
        self.arousal_history: List[Tuple[float, float]] = []

    def set_state(self, state: EmotionalState) -> None:
        """Set current emotional state."""
        self.current_state = state

    def get_arousal(self) -> float:
        """Get current arousal level."""
        return self.arousal_effects[self.current_state][0]

    def modulate(self, duration: float, state: Optional[EmotionalState] = None) -> ModulationEffect:
        """Modulate perceived duration based on emotion.

        Args:
            duration: Actual duration
            state: Emotional state (uses current if None)

        Returns:
            Modulation effect
        """
        if state is None:
            state = self.current_state

        arousal, base_ratio = self.arousal_effects[state]

        # Add variability based on arousal
        # Higher arousal = more variability in time perception
        noise = np.random.normal(0, 0.05 * arousal)
        ratio = base_ratio + noise

        # Store in history
        self.arousal_history.append((arousal, ratio))

        descriptions = {
            EmotionalState.NEUTRAL: "Baseline time perception",
            EmotionalState.FEAR: "Time slowed due to fear response",
            EmotionalState.EXCITEMENT: "Time slowed due to excitement",
            EmotionalState.BOREDOM: "Time dragging due to boredom",
            EmotionalState.FLOW: "Time flying in flow state",
            EmotionalState.ANXIETY: "Time distorted by anxiety",
        }

        return ModulationEffect(
            factor_name="emotion",
            distortion_ratio=ratio,
            confidence_modifier=1.0 - 0.3 * arousal,  # High arousal = less confidence
            description=descriptions[state],
        )

    def process_event(
        self,
        event_valence: float,  # -1 to 1
        event_intensity: float,  # 0 to 1
    ) -> float:
        """Process an emotional event and return arousal change.

        Args:
            event_valence: Positive or negative valence
            event_intensity: How intense the event is

        Returns:
            New arousal level
        """
        current_arousal = self.get_arousal()

        # Events affect arousal
        arousal_change = event_intensity * 0.3

        # Negative events have stronger arousal effect
        if event_valence < 0:
            arousal_change *= 1.5

        new_arousal = np.clip(current_arousal + arousal_change, 0, 1)

        # Update state based on arousal
        if new_arousal > 0.8:
            if event_valence < 0:
                self.current_state = EmotionalState.FEAR
            else:
                self.current_state = EmotionalState.EXCITEMENT
        elif new_arousal < 0.3:
            self.current_state = EmotionalState.BOREDOM
        else:
            self.current_state = EmotionalState.NEUTRAL

        return new_arousal


class AttentionalModulator:
    """Modulates time perception based on attention.

    More attention to time = longer perceived duration.
    This is the "watched pot" effect.
    """

    def __init__(self, base_attention: float = 0.5):
        self.base_attention = base_attention
        self.current_attention = base_attention

        # Attention capacity is limited
        self.attention_capacity = 1.0
        self.attention_allocated_to_time = 0.0

    def allocate_attention(self, to_time: float, to_task: float = 0.0) -> float:
        """Allocate attention between time and task.

        Args:
            to_time: Attention allocated to time monitoring
            to_task: Attention allocated to task

        Returns:
            Actual attention to time after capacity limits
        """
        total_requested = to_time + to_task

        if total_requested > self.attention_capacity:
            # Scale down proportionally
            scale = self.attention_capacity / total_requested
            to_time *= scale

        self.attention_allocated_to_time = to_time
        self.current_attention = to_time

        return to_time

    def modulate(
        self, duration: float, attention_to_time: Optional[float] = None
    ) -> ModulationEffect:
        """Modulate perceived duration based on attention.

        Args:
            duration: Actual duration
            attention_to_time: Explicit attention level (uses current if None)

        Returns:
            Modulation effect
        """
        if attention_to_time is None:
            attention_to_time = self.current_attention

        # More attention = longer perceived duration
        # Relationship is roughly linear
        ratio = 1.0 + 0.4 * (attention_to_time - 0.5)

        # Add noise that increases with less attention
        noise = np.random.normal(0, 0.1 * (1 - attention_to_time))
        ratio += noise

        # Confidence is higher with more attention
        confidence_mod = 0.5 + 0.5 * attention_to_time

        return ModulationEffect(
            factor_name="attention",
            distortion_ratio=ratio,
            confidence_modifier=confidence_mod,
            description=f"Attention level: {attention_to_time:.2f}",
        )

    def dual_task_effect(self, task_difficulty: float) -> float:
        """Calculate time perception effect of dual-task condition.

        When doing a demanding task, less attention for time = shorter perceived duration.

        Args:
            task_difficulty: How demanding the concurrent task is (0-1)

        Returns:
            Attention remaining for time monitoring
        """
        # Harder task = less attention for time
        attention_for_time = 1.0 - task_difficulty * 0.8
        attention_for_time = max(0.1, attention_for_time)

        self.allocate_attention(attention_for_time, task_difficulty)

        return attention_for_time


class DopamineModulator:
    """Modulates time perception based on dopamine levels.

    Dopamine affects the speed of the internal clock:
    - Higher dopamine = faster clock = time passes faster
    - Lower dopamine = slower clock = time drags

    Relevant to Parkinson's disease, stimulant effects, reward.
    """

    def __init__(self, baseline_level: float = 1.0):
        self.baseline_level = baseline_level
        self.current_level = baseline_level

        # Dopamine depletion/excess effects
        self.level_history: List[float] = []

    def set_level(self, level: float) -> None:
        """Set dopamine level relative to baseline.

        Args:
            level: Dopamine level (1.0 = baseline)
        """
        self.current_level = max(0.1, min(3.0, level))
        self.level_history.append(self.current_level)

    def modulate(self, duration: float, dopamine_level: Optional[float] = None) -> ModulationEffect:
        """Modulate perceived duration based on dopamine.

        Args:
            duration: Actual duration
            dopamine_level: Explicit level (uses current if None)

        Returns:
            Modulation effect
        """
        if dopamine_level is None:
            dopamine_level = self.current_level

        # Higher dopamine = faster internal clock = time feels faster
        # (same objective duration produces fewer subjective units)
        # ratio < 1 means perceived shorter
        ratio = 1.0 / dopamine_level

        # Add noise
        noise = np.random.normal(0, 0.05)
        ratio += noise

        if dopamine_level > 1.5:
            description = "Time accelerated (high dopamine)"
        elif dopamine_level < 0.7:
            description = "Time slowed (low dopamine)"
        else:
            description = "Normal dopamine levels"

        return ModulationEffect(
            factor_name="dopamine",
            distortion_ratio=ratio,
            confidence_modifier=1.0,  # Dopamine doesn't affect confidence
            description=description,
        )

    def simulate_reward(self, reward_magnitude: float) -> None:
        """Simulate dopamine release from reward.

        Args:
            reward_magnitude: Size of reward (0-1)
        """
        # Rewards cause dopamine release
        dopamine_boost = reward_magnitude * 0.5
        self.current_level += dopamine_boost

        # Decay back toward baseline
        self.current_level = self.baseline_level + (self.current_level - self.baseline_level) * 0.9

    def simulate_stimulant(self, dose: float) -> None:
        """Simulate stimulant drug effect.

        Args:
            dose: Drug dose (0-1)
        """
        # Stimulants increase dopamine
        self.current_level = self.baseline_level + dose * 1.0

    def simulate_parkinsons(self, severity: float) -> None:
        """Simulate Parkinson's disease dopamine depletion.

        Args:
            severity: Disease severity (0-1)
        """
        # PD causes dopamine loss
        self.current_level = self.baseline_level * (1 - severity * 0.7)


class AgeModulator:
    """Modulates time perception based on age.

    As we age, subjective time accelerates:
    - Proportional theory: Each year is smaller fraction of life
    - Novelty theory: Fewer novel experiences, less memory markers
    - Metabolic theory: Slower metabolism, fewer internal "ticks"
    """

    def __init__(self, current_age: int = 30):
        self.current_age = current_age
        self.reference_age = 20  # Age at which time feels "normal"

    def set_age(self, age: int) -> None:
        """Set current age."""
        self.current_age = max(1, age)

    def modulate(self, duration: float, age: Optional[int] = None) -> ModulationEffect:
        """Modulate perceived duration based on age.

        Args:
            duration: Actual duration
            age: Age to simulate (uses current if None)

        Returns:
            Modulation effect
        """
        if age is None:
            age = self.current_age

        # Proportional theory: time feels faster as we age
        # ratio < 1 means time feels shorter (faster)
        if age >= self.reference_age:
            ratio = self.reference_age / age
        else:
            # Children experience time as slower
            ratio = 1.0 + (self.reference_age - age) * 0.05

        # Add noise
        noise = np.random.normal(0, 0.02)
        ratio += noise

        if age < 10:
            description = "Childhood: time moves slowly"
        elif age < 30:
            description = "Young adult: baseline time perception"
        elif age < 50:
            description = "Middle age: time beginning to accelerate"
        elif age < 70:
            description = "Senior: time passing quickly"
        else:
            description = "Elderly: time feels very fast"

        return ModulationEffect(
            factor_name="age",
            distortion_ratio=ratio,
            confidence_modifier=1.0 - (age - self.reference_age) * 0.005,
            description=description,
        )

    def get_subjective_year_length(self, age: int) -> float:
        """Get how long a year feels at a given age.

        Returns:
            Subjective year length relative to age 20
        """
        if age <= 0:
            return 1.0

        # Proportional theory
        return self.reference_age / age


class TimeModulationSystem:
    """Integrated time modulation system.

    Combines all modulation factors to compute overall
    distortion of time perception.
    """

    def __init__(
        self,
        emotional: Optional[EmotionalModulator] = None,
        attentional: Optional[AttentionalModulator] = None,
        dopamine: Optional[DopamineModulator] = None,
        age: Optional[AgeModulator] = None,
    ):
        self.emotional = emotional or EmotionalModulator()
        self.attentional = attentional or AttentionalModulator()
        self.dopamine = dopamine or DopamineModulator()
        self.age = age or AgeModulator()

        # Factor weights (importance in final calculation)
        self.weights = {"emotion": 0.35, "attention": 0.30, "dopamine": 0.25, "age": 0.10}

    def modulate_duration(
        self,
        actual_duration: float,
        emotional_state: Optional[EmotionalState] = None,
        attention: Optional[float] = None,
        dopamine_level: Optional[float] = None,
        age: Optional[int] = None,
    ) -> Tuple[float, Dict[str, ModulationEffect]]:
        """Apply all modulation factors to get perceived duration.

        Args:
            actual_duration: True duration in seconds
            emotional_state: Current emotional state
            attention: Attention to time (0-1)
            dopamine_level: Dopamine level relative to baseline
            age: Age in years

        Returns:
            Tuple of (perceived_duration, dict of individual effects)
        """
        effects = {}

        # Get individual effects
        effects["emotion"] = self.emotional.modulate(actual_duration, emotional_state)
        effects["attention"] = self.attentional.modulate(actual_duration, attention)
        effects["dopamine"] = self.dopamine.modulate(actual_duration, dopamine_level)
        effects["age"] = self.age.modulate(actual_duration, age)

        # Compute weighted distortion ratio
        total_ratio = 0.0
        total_weight = 0.0

        for factor, weight in self.weights.items():
            effect = effects[factor]
            total_ratio += weight * effect.distortion_ratio
            total_weight += weight

        final_ratio = total_ratio / total_weight if total_weight > 0 else 1.0

        # Apply to duration
        perceived_duration = actual_duration * final_ratio

        return perceived_duration, effects

    def set_state(
        self,
        emotional_state: Optional[EmotionalState] = None,
        attention: Optional[float] = None,
        dopamine: Optional[float] = None,
        age: Optional[int] = None,
    ) -> None:
        """Set current state for all modulators."""
        if emotional_state is not None:
            self.emotional.set_state(emotional_state)
        if attention is not None:
            self.attentional.allocate_attention(attention)
        if dopamine is not None:
            self.dopamine.set_level(dopamine)
        if age is not None:
            self.age.set_age(age)

    def get_current_state(self) -> Dict:
        """Get current state of all modulators."""
        return {
            "emotional_state": self.emotional.current_state.value,
            "arousal": self.emotional.get_arousal(),
            "attention": self.attentional.current_attention,
            "dopamine": self.dopamine.current_level,
            "age": self.age.current_age,
        }
