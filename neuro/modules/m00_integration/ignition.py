"""
Ignition Detection: Determines when content becomes globally available.

In Global Workspace Theory, "ignition" is the transition from local processing
to global broadcast. When activation exceeds a threshold, content ignites and
becomes globally available to all modules - this is the neural correlate of
conscious access.

Based on:
- Dehaene's "ignition" concept in Global Neuronal Workspace
- All-or-none transition from subliminal to conscious processing
- arXiv:2103.01197 - capacity limitations encourage ignition
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import numpy as np
import time

try:
    from .module_interface import ModuleProposal, ContentType  # noqa: F401
except ImportError:
    from module_interface import ModuleProposal


class IgnitionState(Enum):
    """States of the ignition process."""

    SUBLIMINAL = "subliminal"  # Below threshold, local processing only
    THRESHOLD = "threshold"  # Near threshold, may ignite
    IGNITED = "ignited"  # Above threshold, global broadcast
    SUSTAINED = "sustained"  # Maintained ignition
    FADING = "fading"  # Activation declining


@dataclass
class IgnitionParams:
    """Parameters for ignition detection."""

    threshold: float = 0.7  # Activation threshold for ignition
    hysteresis: float = 0.1  # Hysteresis to prevent oscillation
    min_duration: float = 0.05  # Minimum duration for valid ignition (seconds)
    max_sustained: float = 2.0  # Maximum sustained ignition (seconds)
    coherence_weight: float = 0.3  # Weight of buffer coherence in threshold
    diversity_bonus: float = 0.1  # Bonus for diverse sources
    n_features: int = 64


@dataclass
class IgnitionEvent:
    """Record of an ignition event."""

    ignited: bool
    activation: float
    threshold_used: float
    state: IgnitionState
    duration: float
    trigger_proposal: Optional[ModuleProposal]
    coherence: float
    timestamp: float = field(default_factory=time.time)


class IgnitionDetector:
    """
    Detector for ignition events in the global workspace.

    Ignition represents the transition from unconscious to conscious
    processing. Key features:

    1. **Threshold**: Activation must exceed threshold for ignition
    2. **Hysteresis**: Prevents rapid oscillation between states
    3. **Duration**: Ignition must be sustained for minimum duration
    4. **Coherence**: More coherent buffer content lowers threshold
    5. **Diversity**: Multiple sources contributing raises probability

    The all-or-none nature of ignition is a key prediction of GWT:
    content is either fully available (ignited) or not at all (subliminal).
    """

    def __init__(self, params: Optional[IgnitionParams] = None):
        self.params = params or IgnitionParams()

        # Current state
        self._state = IgnitionState.SUBLIMINAL
        self._current_activation = 0.0
        self._ignition_start_time: Optional[float] = None
        self._last_ignition_time = 0.0

        # History
        self._history: List[IgnitionEvent] = []
        self._activation_history: List[Tuple[float, float]] = []  # (time, activation)

    def detect(
        self,
        activation: float,
        buffer: List[ModuleProposal],
    ) -> IgnitionEvent:
        """
        Detect if ignition should occur.

        Args:
            activation: Current workspace activation level
            buffer: Current workspace buffer contents

        Returns:
            IgnitionEvent with detection result
        """
        current_time = time.time()
        self._current_activation = activation

        # Record activation history
        self._activation_history.append((current_time, activation))
        if len(self._activation_history) > 100:
            self._activation_history.pop(0)

        # Compute dynamic threshold based on buffer coherence
        coherence = self._compute_coherence(buffer)
        diversity = self._compute_diversity(buffer)

        dynamic_threshold = (
            self.params.threshold
            - self.params.coherence_weight * coherence
            - self.params.diversity_bonus * diversity
        )
        dynamic_threshold = max(0.3, dynamic_threshold)  # Minimum threshold

        # Determine trigger proposal (strongest in buffer)
        trigger_proposal = None
        if buffer:
            trigger_proposal = max(buffer, key=lambda p: p.activation)

        # State machine logic
        ignited = False
        duration = 0.0

        if self._state == IgnitionState.SUBLIMINAL:
            if activation > dynamic_threshold:
                self._state = IgnitionState.THRESHOLD
                self._ignition_start_time = current_time

        elif self._state == IgnitionState.THRESHOLD:
            if activation > dynamic_threshold:
                elapsed = current_time - (self._ignition_start_time or current_time)
                if elapsed >= self.params.min_duration:
                    self._state = IgnitionState.IGNITED
                    ignited = True
                    duration = elapsed
            else:
                # Failed to maintain threshold
                self._state = IgnitionState.SUBLIMINAL
                self._ignition_start_time = None

        elif self._state == IgnitionState.IGNITED:
            ignited = True
            if self._ignition_start_time:
                duration = current_time - self._ignition_start_time

            if activation > dynamic_threshold - self.params.hysteresis:
                # Still above (with hysteresis)
                if duration > self.params.max_sustained:
                    self._state = IgnitionState.FADING
                else:
                    self._state = IgnitionState.SUSTAINED
            else:
                self._state = IgnitionState.FADING

        elif self._state == IgnitionState.SUSTAINED:
            ignited = True
            if self._ignition_start_time:
                duration = current_time - self._ignition_start_time

            if activation < dynamic_threshold - self.params.hysteresis:
                self._state = IgnitionState.FADING
            elif duration > self.params.max_sustained:
                self._state = IgnitionState.FADING

        elif self._state == IgnitionState.FADING:
            if self._ignition_start_time:
                duration = current_time - self._ignition_start_time

            if activation < dynamic_threshold - self.params.hysteresis * 2:
                self._state = IgnitionState.SUBLIMINAL
                self._ignition_start_time = None
                self._last_ignition_time = current_time
            elif activation > dynamic_threshold:
                # Rekindled
                self._state = IgnitionState.IGNITED
                ignited = True

        # Create event
        event = IgnitionEvent(
            ignited=ignited,
            activation=activation,
            threshold_used=dynamic_threshold,
            state=self._state,
            duration=duration,
            trigger_proposal=trigger_proposal,
            coherence=coherence,
        )

        # Record history
        self._history.append(event)
        if len(self._history) > 100:
            self._history.pop(0)

        return event

    def _compute_coherence(self, buffer: List[ModuleProposal]) -> float:
        """
        Compute coherence of buffer contents.

        High coherence means buffer contents are related/consistent,
        which lowers the ignition threshold.
        """
        if len(buffer) < 2:
            return 0.0

        # Compute pairwise content similarity
        similarities = []
        for i in range(len(buffer)):
            for j in range(i + 1, len(buffer)):
                sim = self._content_similarity(
                    buffer[i].content,
                    buffer[j].content,
                )
                similarities.append(sim)

        return float(np.mean(similarities)) if similarities else 0.0

    def _compute_diversity(self, buffer: List[ModuleProposal]) -> float:
        """
        Compute source diversity of buffer contents.

        Multiple modules contributing suggests important content,
        which lowers the ignition threshold.
        """
        if not buffer:
            return 0.0

        unique_sources = len(set(p.source_module for p in buffer))
        max_possible = len(buffer)

        return unique_sources / max_possible

    def _content_similarity(self, content1: np.ndarray, content2: np.ndarray) -> float:
        """Compute cosine similarity between content vectors."""
        # Handle size mismatch
        if len(content1) != len(content2):
            min_len = min(len(content1), len(content2))
            content1 = content1[:min_len]
            content2 = content2[:min_len]

        norm1 = np.linalg.norm(content1)
        norm2 = np.linalg.norm(content2)

        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0

        return float(np.dot(content1, content2) / (norm1 * norm2))

    def is_ignited(self) -> bool:
        """Check if currently ignited."""
        return self._state in [IgnitionState.IGNITED, IgnitionState.SUSTAINED]

    def get_state(self) -> IgnitionState:
        """Get current ignition state."""
        return self._state

    def get_activation(self) -> float:
        """Get current activation level."""
        return self._current_activation

    def get_recent_events(self, n: int = 10) -> List[IgnitionEvent]:
        """Get recent ignition events."""
        return self._history[-n:]

    def get_ignition_rate(self) -> float:
        """Get rate of ignition events."""
        if not self._history:
            return 0.0
        ignited_count = sum(1 for e in self._history if e.ignited)
        return ignited_count / len(self._history)

    def get_statistics(self) -> Dict[str, Any]:
        """Get ignition statistics."""
        if not self._history:
            return {
                "total_events": 0,
                "ignition_rate": 0.0,
                "avg_duration": 0.0,
                "avg_activation": 0.0,
                "current_state": self._state.value,
            }

        ignited_events = [e for e in self._history if e.ignited]

        return {
            "total_events": len(self._history),
            "ignition_rate": len(ignited_events) / len(self._history),
            "avg_duration": np.mean([e.duration for e in ignited_events])
            if ignited_events
            else 0.0,
            "avg_activation": np.mean([e.activation for e in self._history]),
            "avg_coherence": np.mean([e.coherence for e in self._history]),
            "current_state": self._state.value,
            "current_activation": self._current_activation,
        }

    def reset(self) -> None:
        """Reset ignition detector state."""
        self._state = IgnitionState.SUBLIMINAL
        self._current_activation = 0.0
        self._ignition_start_time = None
        self._last_ignition_time = 0.0
        self._history = []
        self._activation_history = []
