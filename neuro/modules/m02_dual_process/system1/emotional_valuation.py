"""
Emotional Valuation - Amygdala Simulation

Implements rapid threat/reward assessment that happens before conscious awareness.
The amygdala provides fast, rough evaluations that guide behavior.

Key properties:
- FAST - millisecond-scale decisions
- Based on partial feature matching
- Generalizes from learned associations
- Modulates attention and memory
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ValenceType(Enum):
    """Basic emotional valences"""

    THREAT = "threat"
    REWARD = "reward"
    NEUTRAL = "neutral"


@dataclass
class Valence:
    """Emotional valuation of a stimulus"""

    threat: float  # 0-1, danger level
    reward: float  # 0-1, reward potential
    arousal: float  # 0-1, activation level

    @property
    def valence_type(self) -> ValenceType:
        if self.threat > self.reward and self.threat > 0.3:
            return ValenceType.THREAT
        elif self.reward > self.threat and self.reward > 0.3:
            return ValenceType.REWARD
        return ValenceType.NEUTRAL

    @property
    def intensity(self) -> float:
        return max(self.threat, self.reward)


@dataclass
class EmotionalMemory:
    """Stored emotional association"""

    features: np.ndarray
    valence: Valence
    strength: float = 1.0
    update_count: int = 1


class EmotionalValuation:
    """
    Fast threat/reward evaluation system.

    Simulates amygdala:
    - Rapid (<50ms equivalent) evaluation
    - Generalizes from partial features
    - Learns associations between stimuli and outcomes
    - Modulates other systems (attention, memory, arousal)
    """

    def __init__(
        self,
        threat_bias: float = 1.2,  # Negativity bias
        generalization_threshold: float = 0.5,
    ):
        self.memories: List[EmotionalMemory] = []
        self.threat_bias = threat_bias  # Threats weighted more heavily
        self.generalization_threshold = generalization_threshold

        # Feature-level valence associations
        # Some features are inherently threatening/rewarding
        self.innate_valences: Dict[int, Tuple[float, float]] = {}

    def set_innate_valence(self, feature_index: int, threat: float, reward: float):
        """
        Set innate emotional response to specific features.

        Some things are inherently scary (loud noises, snakes) or
        rewarding (sweet tastes) without learning.
        """
        self.innate_valences[feature_index] = (threat, reward)

    def evaluate(self, stimulus: np.ndarray, fast_mode: bool = True) -> Valence:
        """
        Rapidly evaluate stimulus for threat/reward.

        This is the core amygdala function - FAST assessment
        based on learned associations and feature matching.

        fast_mode: If True, uses only top matches (faster but less accurate)
        """
        threat_sum = 0.0
        reward_sum = 0.0
        arousal_sum = 0.0
        weight_sum = 0.0

        # Check innate valences (hardwired responses)
        for idx, (t, r) in self.innate_valences.items():
            if idx < len(stimulus):
                activation = abs(stimulus[idx])
                threat_sum += t * activation
                reward_sum += r * activation
                arousal_sum += (t + r) * activation
                weight_sum += activation

        # Check learned associations
        stimulus_norm = stimulus / (np.linalg.norm(stimulus) + 1e-8)

        matches = []
        for memory in self.memories:
            memory_norm = memory.features / (np.linalg.norm(memory.features) + 1e-8)
            similarity = float(np.dot(stimulus_norm, memory_norm))

            if similarity >= self.generalization_threshold:
                matches.append((memory, similarity))

        # Sort by similarity for fast mode
        if fast_mode and len(matches) > 3:
            matches.sort(key=lambda x: x[1], reverse=True)
            matches = matches[:3]  # Only use top 3

        for memory, similarity in matches:
            weight = similarity * memory.strength
            threat_sum += memory.valence.threat * weight * self.threat_bias
            reward_sum += memory.valence.reward * weight
            arousal_sum += memory.valence.arousal * weight
            weight_sum += weight

        # Normalize
        if weight_sum > 0:
            threat = np.clip(threat_sum / weight_sum, 0, 1)
            reward = np.clip(reward_sum / weight_sum, 0, 1)
            arousal = np.clip(arousal_sum / weight_sum, 0, 1)
        else:
            # No associations - neutral but slightly arousing (novel)
            threat = 0.1
            reward = 0.1
            arousal = 0.3

        return Valence(threat=threat, reward=reward, arousal=arousal)

    def learn_association(
        self,
        stimulus: np.ndarray,
        outcome_threat: float,
        outcome_reward: float,
        outcome_arousal: Optional[float] = None,
    ):
        """
        Learn emotional association from experience.

        The amygdala rapidly learns that certain stimuli predict
        positive or negative outcomes.
        """
        if outcome_arousal is None:
            outcome_arousal = max(outcome_threat, outcome_reward)

        valence = Valence(
            threat=np.clip(outcome_threat, 0, 1),
            reward=np.clip(outcome_reward, 0, 1),
            arousal=np.clip(outcome_arousal, 0, 1),
        )

        # Check for existing similar memory
        stimulus_norm = stimulus / (np.linalg.norm(stimulus) + 1e-8)

        for memory in self.memories:
            memory_norm = memory.features / (np.linalg.norm(memory.features) + 1e-8)
            similarity = float(np.dot(stimulus_norm, memory_norm))

            if similarity > 0.9:  # Very similar - update existing
                # Running average
                n = memory.update_count
                memory.valence.threat = (memory.valence.threat * n + valence.threat) / (n + 1)
                memory.valence.reward = (memory.valence.reward * n + valence.reward) / (n + 1)
                memory.valence.arousal = (memory.valence.arousal * n + valence.arousal) / (n + 1)
                memory.update_count += 1
                memory.strength = min(1.0, memory.strength + 0.1)
                return

        # New memory
        self.memories.append(EmotionalMemory(features=stimulus.copy(), valence=valence))

    def generalize(self, novel_stimulus: np.ndarray) -> Valence:
        """
        Evaluate novel stimulus by generalizing from known associations.

        Key amygdala property - generalizes threat/reward assessments
        to similar but novel stimuli (e.g., all snake-like things seem scary).
        """
        # Use lower threshold for generalization
        original_threshold = self.generalization_threshold
        self.generalization_threshold = original_threshold * 0.6

        valence = self.evaluate(novel_stimulus, fast_mode=False)

        self.generalization_threshold = original_threshold
        return valence

    def is_threatening(self, stimulus: np.ndarray, threshold: float = 0.5) -> bool:
        """Quick threat check"""
        return self.evaluate(stimulus).threat >= threshold

    def is_rewarding(self, stimulus: np.ndarray, threshold: float = 0.5) -> bool:
        """Quick reward check"""
        return self.evaluate(stimulus).reward >= threshold

    def get_approach_avoid(self, stimulus: np.ndarray) -> float:
        """
        Get approach/avoid tendency.

        Positive = approach (reward > threat)
        Negative = avoid (threat > reward)

        This drives basic behavior - approach rewards, avoid threats.
        """
        valence = self.evaluate(stimulus)
        return valence.reward - valence.threat * self.threat_bias

    def extinction(self, stimulus: np.ndarray, extinction_rate: float = 0.1):
        """
        Reduce emotional association through repeated non-reinforcement.

        This is how fears/desires can be unlearned.
        """
        stimulus_norm = stimulus / (np.linalg.norm(stimulus) + 1e-8)

        for memory in self.memories:
            memory_norm = memory.features / (np.linalg.norm(memory.features) + 1e-8)
            similarity = float(np.dot(stimulus_norm, memory_norm))

            if similarity > self.generalization_threshold:
                # Reduce strength
                memory.strength = max(0.0, memory.strength - extinction_rate)
                # Move valence toward neutral
                memory.valence.threat *= 1 - extinction_rate
                memory.valence.reward *= 1 - extinction_rate
                memory.valence.arousal *= 1 - extinction_rate * 0.5

        # Remove very weak memories
        self.memories = [m for m in self.memories if m.strength > 0.05]

    def modulate_attention(self, stimulus: np.ndarray) -> float:
        """
        Get attention modulation factor.

        Emotionally salient stimuli capture attention.
        """
        valence = self.evaluate(stimulus)
        # Both threats and rewards capture attention
        return 1.0 + valence.intensity * 0.5

    def modulate_memory(self, stimulus: np.ndarray) -> float:
        """
        Get memory encoding strength modulation.

        Emotional events are remembered better.
        """
        valence = self.evaluate(stimulus)
        return 1.0 + valence.arousal * 0.7
