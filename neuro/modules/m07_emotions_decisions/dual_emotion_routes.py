"""
Dual Emotion Routes: Fast vs Slow pathways.

Based on research from: https://link.springer.com/article/10.1007/s00429-023-02644-9

Dual Routes of Emotion:
| Route | Speed  | Path                         | Function               |
|-------|--------|------------------------------|------------------------|
| Fast  | ~12ms  | Thalamus → Amygdala          | Rapid threat detection |
| Slow  | ~100ms | Thalamus → Cortex → Amygdala | Contextual evaluation  |
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from enum import Enum
import numpy as np


class ResponseType(Enum):
    """Type of emotional response."""

    THREAT = "threat"
    REWARD = "reward"
    NEUTRAL = "neutral"
    OVERRIDE = "override"


@dataclass
class EmotionRouteResponse:
    """Response from an emotion processing route."""

    response_type: ResponseType
    intensity: float  # 0-1
    latency_ms: float
    confidence: float  # 0-1
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


class ThalamusRelay:
    """
    Sensory gateway - routes information to fast/slow paths.

    The thalamus receives sensory input and distributes it to:
    1. Direct amygdala pathway (fast, coarse)
    2. Cortical pathway (slow, detailed)
    """

    def __init__(self):
        self.current_input: Optional[np.ndarray] = None
        self.input_timestamp: float = 0.0

    def receive_sensory(self, stimulus: np.ndarray) -> np.ndarray:
        """Receive and store sensory input."""
        self.current_input = stimulus.copy()
        self.input_timestamp += 1  # Simulated time increment
        return self.current_input

    def route_to_amygdala(self) -> np.ndarray:
        """
        Direct fast path - coarse representation.

        Returns a low-resolution version of the input for fast processing.
        """
        if self.current_input is None:
            return np.array([0.0])

        # Coarse representation: downsample/summarize
        if len(self.current_input) > 4:
            # Take key statistics instead of full detail
            coarse = np.array(
                [
                    np.mean(self.current_input),
                    np.max(self.current_input),
                    np.min(self.current_input),
                    np.std(self.current_input),
                ]
            )
        else:
            coarse = self.current_input

        return coarse

    def route_to_cortex(self) -> np.ndarray:
        """
        Indirect slow path - full detailed representation.

        Returns full-resolution input for detailed cortical analysis.
        """
        if self.current_input is None:
            return np.array([0.0])

        return self.current_input.copy()


class FastEmotionRoute:
    """
    Thalamus → Amygdala direct path (~12ms).

    Characteristics:
    - Very fast (subcortical)
    - Coarse representation (low resolution)
    - "Better safe than sorry" - biased toward threat detection
    - Cannot be consciously controlled
    """

    LATENCY_MS = 12.0

    def __init__(self, threat_bias: float = 0.2):
        self.threat_bias = threat_bias  # Tendency to interpret ambiguity as threat

        # Innate threat detectors (evolutionary)
        self.innate_threat_features = {
            "high_contrast": 0.3,
            "rapid_approach": 0.5,
            "large_size": 0.2,
            "sudden_onset": 0.4,
        }

        # Learned threat patterns
        self.learned_threats: Dict[int, float] = {}

    def process(self, stimulus: np.ndarray) -> EmotionRouteResponse:
        """
        Fast threat/reward evaluation.

        This fires BEFORE conscious awareness.
        """
        # Innate threat features
        innate_threat = self._detect_innate_threats(stimulus)

        # Learned threats (conditioned)
        stimulus_hash = hash(stimulus.tobytes())
        learned_threat = self.learned_threats.get(stimulus_hash, 0.0)

        # Total threat (biased toward detecting threats)
        total_threat = max(innate_threat, learned_threat) + self.threat_bias * 0.5

        # Determine response
        if total_threat > 0.5:
            return EmotionRouteResponse(
                response_type=ResponseType.THREAT,
                intensity=min(1.0, total_threat),
                latency_ms=self.LATENCY_MS,
                confidence=0.6,  # Fast route has lower confidence
                details={
                    "innate_threat": innate_threat,
                    "learned_threat": learned_threat,
                    "path": "fast_subcortical",
                },
            )
        else:
            # Check for reward signals
            reward = self._detect_reward(stimulus)
            if reward > 0.4:
                return EmotionRouteResponse(
                    response_type=ResponseType.REWARD,
                    intensity=reward,
                    latency_ms=self.LATENCY_MS,
                    confidence=0.5,
                    details={"path": "fast_subcortical"},
                )

            return EmotionRouteResponse(
                response_type=ResponseType.NEUTRAL,
                intensity=0.0,
                latency_ms=self.LATENCY_MS,
                confidence=0.4,
                details={"path": "fast_subcortical"},
            )

    def _detect_innate_threats(self, stimulus: np.ndarray) -> float:
        """Detect evolutionarily-prepared threat features."""
        threat_score = 0.0

        # High contrast (potential predator)
        contrast = np.max(stimulus) - np.min(stimulus) if len(stimulus) > 1 else 0
        if contrast > 0.7:
            threat_score += self.innate_threat_features["high_contrast"]

        # Sudden large values (looming)
        if np.max(np.abs(stimulus)) > 0.8:
            threat_score += self.innate_threat_features["large_size"]

        # High variance (unpredictable)
        if np.std(stimulus) > 0.4:
            threat_score += self.innate_threat_features["sudden_onset"]

        return min(1.0, threat_score)

    def _detect_reward(self, stimulus: np.ndarray) -> float:
        """Detect potential reward signals."""
        # Simplified: moderate positive values with low threat
        if np.mean(stimulus) > 0.3 and np.std(stimulus) < 0.3:
            return np.mean(stimulus)
        return 0.0

    def condition_threat(self, stimulus: np.ndarray, threat_level: float):
        """Learn new threat association."""
        stimulus_hash = hash(stimulus.tobytes())
        self.learned_threats[stimulus_hash] = min(1.0, threat_level)


class SlowEmotionRoute:
    """
    Thalamus → Cortex → Amygdala path (~100ms).

    Characteristics:
    - Slower (requires cortical processing)
    - Detailed representation (high resolution)
    - Context-aware evaluation
    - Can override fast route responses
    """

    LATENCY_MS = 100.0

    def __init__(self):
        # Contextual knowledge
        self.context_memory: Dict[str, Any] = {}

        # Category knowledge (what things are)
        self.category_knowledge: Dict[int, str] = {}

        # Safety overrides (things that look threatening but aren't)
        self.safety_overrides: Dict[int, str] = {}

    def process(self, stimulus: np.ndarray, context: Dict[str, Any] = None) -> EmotionRouteResponse:
        """
        Detailed contextual evaluation.

        This can recognize that the "snake" is actually a stick.
        """
        context = context or {}
        stimulus_hash = hash(stimulus.tobytes())

        # Check for safety overrides
        if stimulus_hash in self.safety_overrides:
            return EmotionRouteResponse(
                response_type=ResponseType.OVERRIDE,
                intensity=0.0,
                latency_ms=self.LATENCY_MS,
                confidence=0.9,
                details={
                    "override_reason": self.safety_overrides[stimulus_hash],
                    "path": "slow_cortical",
                },
            )

        # Detailed feature analysis
        threat_assessment = self._detailed_threat_analysis(stimulus, context)
        reward_assessment = self._detailed_reward_analysis(stimulus, context)

        # Context modulation
        if context.get("safe_environment", False):
            threat_assessment *= 0.5

        if context.get("known_danger", False):
            threat_assessment = max(threat_assessment, 0.8)

        # Determine response
        if threat_assessment > 0.5:
            response_type = ResponseType.THREAT
            intensity = threat_assessment
        elif reward_assessment > 0.4:
            response_type = ResponseType.REWARD
            intensity = reward_assessment
        else:
            response_type = ResponseType.NEUTRAL
            intensity = abs(threat_assessment - reward_assessment)

        return EmotionRouteResponse(
            response_type=response_type,
            intensity=intensity,
            latency_ms=self.LATENCY_MS,
            confidence=0.85,  # Slow route has higher confidence
            details={
                "threat_assessment": threat_assessment,
                "reward_assessment": reward_assessment,
                "context_used": list(context.keys()),
                "path": "slow_cortical",
            },
        )

    def _detailed_threat_analysis(self, stimulus: np.ndarray, context: Dict[str, Any]) -> float:
        """
        Detailed threat analysis using full stimulus and context.
        """
        base_threat = 0.0

        # Analyze stimulus properties
        variance = np.var(stimulus)
        mean_val = np.mean(stimulus)
        max_val = np.max(np.abs(stimulus))

        # High variance + negative mean suggests threat
        if variance > 0.3 and mean_val < 0:
            base_threat += 0.4

        # Very high values
        if max_val > 0.9:
            base_threat += 0.3

        # Context modulation
        past_experiences = context.get("past_experiences", [])
        if past_experiences:
            avg_outcome = np.mean(past_experiences)
            if avg_outcome < 0:
                base_threat += 0.2

        return min(1.0, base_threat)

    def _detailed_reward_analysis(self, stimulus: np.ndarray, context: Dict[str, Any]) -> float:
        """Detailed reward analysis using full stimulus and context."""
        base_reward = 0.0

        mean_val = np.mean(stimulus)
        variance = np.var(stimulus)

        # Positive mean with low variance suggests reward
        if mean_val > 0.3 and variance < 0.3:
            base_reward += mean_val

        # Context: known rewarding situation
        if context.get("reward_context", False):
            base_reward += 0.3

        return min(1.0, base_reward)

    def learn_safety_override(self, stimulus: np.ndarray, reason: str):
        """
        Learn that something is safe despite looking threatening.

        Example: The stick is not a snake.
        """
        stimulus_hash = hash(stimulus.tobytes())
        self.safety_overrides[stimulus_hash] = reason

    def learn_category(self, stimulus: np.ndarray, category: str):
        """Learn what category a stimulus belongs to."""
        stimulus_hash = hash(stimulus.tobytes())
        self.category_knowledge[stimulus_hash] = category


class DualRouteProcessor:
    """
    Coordinates fast and slow emotional processing.

    Key behaviors:
    1. Fast route fires first (~12ms)
    2. Slow route provides context (~100ms)
    3. Slow route can override fast route
    4. Under high stress, fast route dominates
    """

    def __init__(self, stress_level: float = 0.0):
        self.thalamus = ThalamusRelay()
        self.fast_route = FastEmotionRoute()
        self.slow_route = SlowEmotionRoute()
        self.stress_level = stress_level

    def process(
        self, stimulus: np.ndarray, context: Dict[str, Any] = None, wait_for_slow: bool = True
    ) -> Tuple[EmotionRouteResponse, Optional[EmotionRouteResponse]]:
        """
        Process stimulus through both routes.

        Args:
            stimulus: Input stimulus
            context: Contextual information
            wait_for_slow: Whether to wait for slow route (normally yes)

        Returns:
            Tuple of (fast_response, slow_response or None)
        """
        context = context or {}

        # Thalamus receives input
        self.thalamus.receive_sensory(stimulus)

        # Fast route processes coarse representation
        coarse_input = self.thalamus.route_to_amygdala()
        fast_response = self.fast_route.process(coarse_input)

        # Under extreme stress, may not wait for slow route
        if self.stress_level > 0.9:
            return fast_response, None

        if not wait_for_slow:
            return fast_response, None

        # Slow route processes detailed representation
        detailed_input = self.thalamus.route_to_cortex()
        slow_response = self.slow_route.process(detailed_input, context)

        return fast_response, slow_response

    def get_final_response(
        self, stimulus: np.ndarray, context: Dict[str, Any] = None
    ) -> EmotionRouteResponse:
        """
        Get the final reconciled response from both routes.
        """
        fast_response, slow_response = self.process(stimulus, context)

        if slow_response is None:
            return fast_response

        # Reconciliation logic
        return self._reconcile(fast_response, slow_response)

    def _reconcile(
        self, fast: EmotionRouteResponse, slow: EmotionRouteResponse
    ) -> EmotionRouteResponse:
        """
        Reconcile fast and slow route responses.

        Rules:
        1. If slow says OVERRIDE, trust slow (it's a stick, not a snake)
        2. If both agree, combine with weighted confidence
        3. If they disagree, weight by confidence and stress level
        """
        # Override takes precedence
        if slow.response_type == ResponseType.OVERRIDE:
            return EmotionRouteResponse(
                response_type=ResponseType.NEUTRAL,
                intensity=slow.intensity,
                latency_ms=slow.latency_ms,
                confidence=slow.confidence,
                details={
                    "reconciled": True,
                    "fast_overridden": True,
                    "override_reason": slow.details.get("override_reason", "unknown"),
                },
            )

        # If both agree on response type
        if fast.response_type == slow.response_type:
            # Combine intensities weighted by confidence
            combined_intensity = (
                fast.intensity * fast.confidence + slow.intensity * slow.confidence
            ) / (fast.confidence + slow.confidence)

            combined_confidence = (fast.confidence + slow.confidence) / 2

            return EmotionRouteResponse(
                response_type=fast.response_type,
                intensity=combined_intensity,
                latency_ms=slow.latency_ms,
                confidence=combined_confidence,
                details={
                    "reconciled": True,
                    "agreement": True,
                    "fast_intensity": fast.intensity,
                    "slow_intensity": slow.intensity,
                },
            )

        # Disagreement: weight by confidence and stress
        fast_weight = fast.confidence * (1 + self.stress_level)
        slow_weight = slow.confidence * (1 - self.stress_level * 0.5)

        if fast_weight > slow_weight:
            winner = fast
        else:
            winner = slow

        # Modulate intensity by disagreement
        disagreement_penalty = 0.8

        return EmotionRouteResponse(
            response_type=winner.response_type,
            intensity=winner.intensity * disagreement_penalty,
            latency_ms=slow.latency_ms,
            confidence=abs(fast_weight - slow_weight) / (fast_weight + slow_weight),
            details={
                "reconciled": True,
                "agreement": False,
                "winner": "fast" if winner == fast else "slow",
                "stress_level": self.stress_level,
            },
        )

    def set_stress_level(self, level: float):
        """Set current stress level (affects fast/slow balance)."""
        self.stress_level = np.clip(level, 0.0, 1.0)

    def learn_safety(self, stimulus: np.ndarray, reason: str):
        """Teach that something is safe (slow route learning)."""
        self.slow_route.learn_safety_override(stimulus, reason)

    def condition_fear(self, stimulus: np.ndarray, threat_level: float):
        """Classical fear conditioning (fast route learning)."""
        coarse = self.thalamus.receive_sensory(stimulus)
        coarse = self.thalamus.route_to_amygdala()
        self.fast_route.condition_threat(coarse, threat_level)
