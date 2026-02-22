"""
Omission Detector

Detects when expected inputs FAIL to arrive - a key aspect of predictive coding.
The brain generates prediction errors not just for unexpected inputs,
but also for the ABSENCE of expected inputs.

This implements the finding that predictive coding systems show
"vigorous prediction error" for omitted expected stimuli.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field
from enum import Enum
from collections import deque


class OmissionType(Enum):
    """Types of omission events"""

    COMPLETE = "complete"  # Expected input entirely absent
    PARTIAL = "partial"  # Input present but reduced
    TEMPORAL = "temporal"  # Input delayed beyond expectation
    PATTERN = "pattern"  # Pattern break (sequence violation)


@dataclass
class ExpectedInput:
    """Specification of an expected input"""

    value: np.ndarray
    time_window: float  # When we expect it (relative)
    tolerance: float = 0.1  # Temporal tolerance
    magnitude_threshold: float = 0.5  # Minimum similarity for "present"
    id: str = ""


@dataclass
class OmissionEvent:
    """Record of a detected omission"""

    omission_type: OmissionType
    expected: ExpectedInput
    actual: Optional[np.ndarray]
    time: float
    confidence: float
    error_magnitude: float


class OmissionDetector:
    """Detects when expected inputs fail to arrive.

    Core mechanism:
    1. Learn to expect inputs based on context/history
    2. Track time windows for expected inputs
    3. Generate prediction error when expectation is violated
    """

    def __init__(
        self, input_dim: int, expectation_decay: float = 0.1, temporal_precision: float = 0.05
    ):
        """Initialize omission detector.

        Args:
            input_dim: Dimensionality of inputs
            expectation_decay: Rate at which expectations decay
            temporal_precision: Precision of timing expectations
        """
        self.input_dim = input_dim
        self.expectation_decay = expectation_decay
        self.temporal_precision = temporal_precision

        # Current expectations
        self.expectations: List[ExpectedInput] = []

        # Input history
        self.input_history: deque = deque(maxlen=1000)
        self.time_history: deque = deque(maxlen=1000)

        # Omission events
        self.omission_events: List[OmissionEvent] = []

        # Current time
        self.current_time = 0.0

        # Learned temporal patterns
        self.temporal_patterns: Dict[str, List[float]] = {}

        # Omission error signal
        self.omission_error = np.zeros(input_dim)

    def add_expectation(
        self, value: np.ndarray, time_window: float, tolerance: float = 0.1, id: str = ""
    ) -> None:
        """Add an expected input.

        Args:
            value: Expected input value
            time_window: When to expect it (relative to now)
            tolerance: Temporal tolerance window
            id: Optional identifier
        """
        expectation = ExpectedInput(
            value=value.copy(),
            time_window=time_window,
            tolerance=tolerance,
            id=id if id else f"exp_{len(self.expectations)}",
        )
        self.expectations.append(expectation)

    def receive_input(self, value: np.ndarray, timestamp: float) -> Optional[OmissionEvent]:
        """Process an incoming input.

        Args:
            value: Input value
            timestamp: Current time

        Returns:
            OmissionEvent if an omission was detected, None otherwise
        """
        self.current_time = timestamp

        # Store in history
        self.input_history.append(value.copy())
        self.time_history.append(timestamp)

        # Check each expectation
        omission = None
        satisfied_indices = []

        for i, exp in enumerate(self.expectations):
            expected_time = exp.time_window

            # Check if we're in the time window
            if abs(timestamp - expected_time) < exp.tolerance:
                # Input arrived in window - check if it matches
                similarity = self._compute_similarity(value, exp.value)

                if similarity >= exp.magnitude_threshold:
                    # Expectation satisfied
                    satisfied_indices.append(i)
                else:
                    # Partial omission - input present but wrong
                    omission = self._create_omission_event(
                        OmissionType.PARTIAL, exp, value, timestamp
                    )

            elif timestamp > expected_time + exp.tolerance:
                # Window passed without satisfying input - temporal omission
                omission = self._create_omission_event(OmissionType.TEMPORAL, exp, value, timestamp)
                satisfied_indices.append(i)  # Remove expired expectation

        # Remove satisfied/expired expectations
        self.expectations = [
            exp for i, exp in enumerate(self.expectations) if i not in satisfied_indices
        ]

        # Update omission error signal
        if omission is not None:
            self._update_omission_error(omission)

        return omission

    def check_omissions(self, timestamp: float) -> List[OmissionEvent]:
        """Check for omissions at given time (when no input arrives).

        Call this periodically to detect complete omissions.

        Args:
            timestamp: Current time

        Returns:
            List of detected omissions
        """
        self.current_time = timestamp
        omissions = []
        expired_indices = []

        for i, exp in enumerate(self.expectations):
            if timestamp > exp.time_window + exp.tolerance:
                # Complete omission - expected input never arrived
                omission = self._create_omission_event(OmissionType.COMPLETE, exp, None, timestamp)
                omissions.append(omission)
                expired_indices.append(i)

        # Remove expired expectations
        self.expectations = [
            exp for i, exp in enumerate(self.expectations) if i not in expired_indices
        ]

        # Update omission error
        for omission in omissions:
            self._update_omission_error(omission)

        return omissions

    def generate_omission_error(self) -> np.ndarray:
        """Generate prediction error signal for omissions.

        Returns the current omission error (decays over time).
        """
        return self.omission_error.copy()

    def _compute_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute similarity between two inputs"""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0

        return float(np.dot(a, b) / (norm_a * norm_b))

    def _create_omission_event(
        self,
        omission_type: OmissionType,
        expected: ExpectedInput,
        actual: Optional[np.ndarray],
        timestamp: float,
    ) -> OmissionEvent:
        """Create an omission event record"""
        # Compute error magnitude
        if actual is not None:
            error_mag = float(np.linalg.norm(expected.value - actual))
        else:
            error_mag = float(np.linalg.norm(expected.value))

        # Confidence based on expectation precision
        confidence = 1.0 / (1.0 + expected.tolerance)

        event = OmissionEvent(
            omission_type=omission_type,
            expected=expected,
            actual=actual,
            time=timestamp,
            confidence=confidence,
            error_magnitude=error_mag,
        )

        self.omission_events.append(event)
        return event

    def _update_omission_error(self, event: OmissionEvent) -> None:
        """Update the omission error signal"""
        # Error is the expected value (what should have been there)
        if event.actual is not None:
            error = event.expected.value - event.actual
        else:
            error = event.expected.value

        # Weight by confidence
        self.omission_error = event.confidence * error

    def decay_error(self, dt: float) -> None:
        """Apply temporal decay to omission error"""
        self.omission_error *= np.exp(-self.expectation_decay * dt)

    def reset(self) -> None:
        """Reset detector state"""
        self.expectations = []
        self.input_history.clear()
        self.time_history.clear()
        self.omission_error = np.zeros(self.input_dim)
        self.omission_events = []


class SequenceOmissionDetector:
    """Detects omissions in learned sequences.

    Learns sequential patterns and detects when expected
    sequence elements are missing.
    """

    def __init__(self, input_dim: int, sequence_length: int = 10, learning_rate: float = 0.1):
        """Initialize sequence omission detector.

        Args:
            input_dim: Dimensionality of sequence elements
            sequence_length: Maximum sequence length to learn
            learning_rate: Learning rate for sequence patterns
        """
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate

        # Sequence memory: maps context to expected next
        # Using simple associative memory
        self.context_dim = input_dim * (sequence_length - 1)
        self.W = np.zeros((input_dim, self.context_dim))

        # Current context buffer
        self.context_buffer: deque = deque(maxlen=sequence_length - 1)

        # Base omission detector
        self.omission_detector = OmissionDetector(input_dim)

    def observe(self, value: np.ndarray, timestamp: float) -> Optional[np.ndarray]:
        """Observe a sequence element.

        Args:
            value: Current sequence element
            timestamp: Current time

        Returns:
            Prediction error if omission/mismatch detected
        """
        error = None

        # If we have context, check prediction
        if len(self.context_buffer) > 0:
            context = self._get_context_vector()
            prediction = self.W @ context

            # Check for mismatch
            error = value - prediction
            error_mag = np.linalg.norm(error)

            if error_mag > 0.5:  # Threshold for significant error
                # This could be an omission or unexpected input
                self.omission_detector.omission_error = error

        # Learn the association
        if len(self.context_buffer) == self.sequence_length - 1:
            context = self._get_context_vector()
            # Hebbian-like learning
            self.W += self.learning_rate * np.outer(value - self.W @ context, context)

        # Update context buffer
        self.context_buffer.append(value.copy())

        return error

    def predict_next(self) -> np.ndarray:
        """Predict the next sequence element"""
        if len(self.context_buffer) < self.sequence_length - 1:
            return np.zeros(self.input_dim)

        context = self._get_context_vector()
        return self.W @ context

    def _get_context_vector(self) -> np.ndarray:
        """Flatten context buffer into vector"""
        if len(self.context_buffer) == 0:
            return np.zeros(self.context_dim)

        # Pad if needed
        padded = list(self.context_buffer)
        while len(padded) < self.sequence_length - 1:
            padded.insert(0, np.zeros(self.input_dim))

        return np.concatenate(padded)

    def reset(self) -> None:
        """Reset detector"""
        self.context_buffer.clear()
        self.omission_detector.reset()


class RhythmicOmissionDetector:
    """Detects omissions in rhythmic/periodic patterns.

    Learns periodic structure and detects when expected
    rhythmic beats are missing.
    """

    def __init__(self, input_dim: int, max_period: float = 2.0, period_resolution: int = 100):
        """Initialize rhythmic omission detector.

        Args:
            input_dim: Dimensionality of inputs
            max_period: Maximum period to detect (seconds)
            period_resolution: Number of phase bins
        """
        self.input_dim = input_dim
        self.max_period = max_period
        self.period_resolution = period_resolution

        # Estimated period
        self.period: Optional[float] = None
        self.period_confidence = 0.0

        # Phase-locked expectations
        self.phase_bins = np.zeros((period_resolution, input_dim))
        self.phase_counts = np.zeros(period_resolution)

        # Input history for period estimation
        self.event_times: List[float] = []
        self.max_events = 100

        # Base detector
        self.omission_detector = OmissionDetector(input_dim)
        self.current_time = 0.0

    def observe(
        self, value: np.ndarray, timestamp: float
    ) -> Tuple[Optional[float], Optional[np.ndarray]]:
        """Observe a rhythmic event.

        Args:
            value: Event value
            timestamp: Event time

        Returns:
            Tuple of (period estimate, omission error if any)
        """
        self.current_time = timestamp
        self.event_times.append(timestamp)

        if len(self.event_times) > self.max_events:
            self.event_times.pop(0)

        # Estimate period from inter-event intervals
        if len(self.event_times) >= 3:
            self._estimate_period()

        # Check for omission relative to expected phase
        omission_error = None
        if self.period is not None:
            phase = self._get_phase(timestamp)
            phase_bin = int(phase * self.period_resolution) % self.period_resolution

            expected = self.phase_bins[phase_bin]
            if self.phase_counts[phase_bin] > 0:
                expected = expected / self.phase_counts[phase_bin]
                error = value - expected
                if np.linalg.norm(error) > 0.5:
                    omission_error = error

            # Update phase bin
            self.phase_bins[phase_bin] += value
            self.phase_counts[phase_bin] += 1

        return self.period, omission_error

    def check_expected_beat(self, timestamp: float) -> Optional[OmissionEvent]:
        """Check if a beat was expected at this time.

        Call this when no input is received.
        """
        if self.period is None:
            return None

        phase = self._get_phase(timestamp)
        phase_bin = int(phase * self.period_resolution) % self.period_resolution

        # Check if this phase typically has strong activity
        if self.phase_counts[phase_bin] > 2:
            expected_magnitude = np.linalg.norm(
                self.phase_bins[phase_bin] / self.phase_counts[phase_bin]
            )

            if expected_magnitude > 0.3:  # Expect significant input at this phase
                # Generate omission
                expected = ExpectedInput(
                    value=self.phase_bins[phase_bin] / self.phase_counts[phase_bin],
                    time_window=timestamp,
                )
                return OmissionEvent(
                    omission_type=OmissionType.COMPLETE,
                    expected=expected,
                    actual=None,
                    time=timestamp,
                    confidence=self.period_confidence,
                    error_magnitude=expected_magnitude,
                )

        return None

    def _estimate_period(self) -> None:
        """Estimate period from event times"""
        if len(self.event_times) < 3:
            return

        # Compute inter-event intervals
        intervals = np.diff(self.event_times[-20:])

        if len(intervals) < 2:
            return

        # Find dominant interval
        # Simple approach: median of intervals
        median_interval = np.median(intervals)

        if 0 < median_interval < self.max_period:
            # Update period estimate with momentum
            if self.period is None:
                self.period = median_interval
            else:
                self.period = 0.9 * self.period + 0.1 * median_interval

            # Confidence based on interval consistency
            interval_std = np.std(intervals)
            self.period_confidence = 1.0 / (1.0 + interval_std / median_interval)

    def _get_phase(self, timestamp: float) -> float:
        """Get phase at given time (0 to 1)"""
        if self.period is None or self.period == 0:
            return 0.0
        return (timestamp % self.period) / self.period

    def reset(self) -> None:
        """Reset detector"""
        self.period = None
        self.period_confidence = 0.0
        self.phase_bins = np.zeros((self.period_resolution, self.input_dim))
        self.phase_counts = np.zeros(self.period_resolution)
        self.event_times = []
        self.omission_detector.reset()
