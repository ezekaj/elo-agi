"""
Interval Timing Models

Computational models of how the brain times intervals:
- Pacemaker-Accumulator: Classic model with clock pulses
- Striatal Beat Frequency: Oscillator-based timing
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum


class TimingMode(Enum):
    """Timing operation modes"""

    PROSPECTIVE = "prospective"  # Timing in advance (attention to time)
    RETROSPECTIVE = "retrospective"  # Timing after the fact (memory-based)
    PRODUCTION = "production"  # Producing a target duration
    REPRODUCTION = "reproduction"  # Reproducing a presented duration


@dataclass
class TimingResult:
    """Result of a timing operation"""

    estimated_duration: float
    actual_duration: float
    error: float  # estimated - actual
    relative_error: float  # error / actual
    mode: TimingMode
    confidence: float


class PacemakerAccumulator:
    """Classic Pacemaker-Accumulator model of interval timing.

    Components:
    - Pacemaker: Generates pulses at some rate
    - Switch: Gates pulses to accumulator (attention-controlled)
    - Accumulator: Counts pulses
    - Memory: Stores reference durations
    - Comparator: Compares accumulated count to memory

    Weber's Law: Timing variability scales with duration (scalar property)
    """

    def __init__(
        self,
        base_rate: float = 10.0,  # Pulses per second
        switch_latency: float = 0.05,  # Seconds to open switch
        memory_noise: float = 0.1,
    ):
        self.base_rate = base_rate
        self.switch_latency = switch_latency
        self.memory_noise = memory_noise

        # State
        self.accumulator = 0.0
        self.switch_open = False
        self.current_time = 0.0

        # Reference memory (learned durations)
        self.reference_memory: Dict[str, float] = {}

        # Clock speed modulation (dopamine effect)
        self.clock_speed_factor = 1.0

    @property
    def effective_rate(self) -> float:
        """Current pulse rate with modulation."""
        return self.base_rate * self.clock_speed_factor

    def start_timing(self) -> None:
        """Start timing (open switch after latency)."""
        self.switch_open = True
        self.accumulator = 0.0

    def stop_timing(self) -> float:
        """Stop timing and return accumulated pulses."""
        self.switch_open = False
        return self.accumulator

    def accumulate(self, duration: float, attention: float = 1.0) -> float:
        """Accumulate pulses over a duration.

        Args:
            duration: Duration to time
            attention: Attention level (affects pulse gating)

        Returns:
            Accumulated pulse count
        """
        if not self.switch_open:
            return self.accumulator

        # Pulses generated
        pulses = duration * self.effective_rate

        # Attention gates pulses (less attention = pulses leak)
        gated_pulses = pulses * attention

        # Add to accumulator with some noise
        noise = np.random.normal(0, np.sqrt(gated_pulses) * 0.1)
        self.accumulator += gated_pulses + noise

        self.current_time += duration

        return self.accumulator

    def estimate_duration(
        self,
        actual_duration: float,
        attention: float = 1.0,
        mode: TimingMode = TimingMode.PROSPECTIVE,
    ) -> TimingResult:
        """Estimate a duration.

        Args:
            actual_duration: True duration in seconds
            attention: Attention to timing task
            mode: Prospective vs retrospective timing

        Returns:
            Timing result with estimate and error
        """
        self.start_timing()

        # For retrospective timing, attention is lower
        effective_attention = attention if mode == TimingMode.PROSPECTIVE else attention * 0.7

        # Accumulate pulses
        self.accumulate(actual_duration, effective_attention)

        # Convert pulses back to duration estimate
        estimated_duration = self.accumulator / self.effective_rate

        # Add scalar timing noise (Weber's law)
        noise_std = 0.1 * actual_duration  # ~10% coefficient of variation
        noise = np.random.normal(0, noise_std)
        estimated_duration += noise

        self.stop_timing()

        error = estimated_duration - actual_duration
        relative_error = error / actual_duration if actual_duration > 0 else 0

        # Confidence based on attention and duration
        confidence = effective_attention / (1 + 0.1 * actual_duration)

        return TimingResult(
            estimated_duration=max(0, estimated_duration),
            actual_duration=actual_duration,
            error=error,
            relative_error=relative_error,
            mode=mode,
            confidence=confidence,
        )

    def produce_duration(self, target_duration: float, tolerance: float = 0.1) -> TimingResult:
        """Produce a target duration (stop when you think time is up).

        Args:
            target_duration: Duration to produce
            tolerance: Acceptable error tolerance

        Returns:
            Timing result
        """
        # Target pulse count
        target_pulses = target_duration * self.effective_rate

        # Add noise to target (memory variability)
        noisy_target = target_pulses * (1 + np.random.normal(0, self.memory_noise))

        self.start_timing()

        # Simulate accumulation until target reached
        dt = 0.01
        elapsed = 0.0

        while self.accumulator < noisy_target:
            self.accumulate(dt)
            elapsed += dt

        self.stop_timing()

        error = elapsed - target_duration
        relative_error = error / target_duration if target_duration > 0 else 0

        return TimingResult(
            estimated_duration=elapsed,
            actual_duration=target_duration,
            error=error,
            relative_error=relative_error,
            mode=TimingMode.PRODUCTION,
            confidence=1.0 / (1 + abs(relative_error)),
        )

    def store_reference(self, name: str, duration: float) -> None:
        """Store a reference duration in memory."""
        # Add memory noise
        noisy_duration = duration * (1 + np.random.normal(0, self.memory_noise))
        self.reference_memory[name] = noisy_duration

    def compare_to_reference(self, duration: float, reference_name: str) -> Tuple[str, float]:
        """Compare duration to stored reference.

        Returns:
            Tuple of (comparison result, ratio)
        """
        if reference_name not in self.reference_memory:
            return ("unknown", 0.0)

        reference = self.reference_memory[reference_name]
        ratio = duration / reference

        if ratio < 0.9:
            return ("shorter", ratio)
        elif ratio > 1.1:
            return ("longer", ratio)
        else:
            return ("equal", ratio)

    def set_clock_speed(self, factor: float) -> None:
        """Modulate clock speed (e.g., by dopamine)."""
        self.clock_speed_factor = max(0.5, min(2.0, factor))

    def reset(self) -> None:
        """Reset timing state."""
        self.accumulator = 0.0
        self.switch_open = False
        self.current_time = 0.0


class StriatalBeatFrequency:
    """Striatal Beat Frequency model of interval timing.

    Time is encoded by patterns of oscillator activity:
    - Multiple oscillators at different frequencies
    - Striatal neurons detect coincident oscillator states
    - Duration encoded as time to return to a pattern

    More biologically plausible than pacemaker-accumulator.
    """

    def __init__(
        self,
        n_oscillators: int = 100,
        freq_range: Tuple[float, float] = (1.0, 20.0),
        detection_threshold: float = 0.8,
    ):
        self.n_oscillators = n_oscillators
        self.freq_range = freq_range
        self.detection_threshold = detection_threshold

        # Initialize oscillators with different frequencies
        self.frequencies = np.linspace(freq_range[0], freq_range[1], n_oscillators)

        # Random initial phases
        self.initial_phases = np.random.uniform(0, 2 * np.pi, n_oscillators)

        # Striatal detector weights (learned)
        self.detector_weights = np.ones(n_oscillators) / n_oscillators

        # Learned duration patterns
        self.learned_patterns: Dict[str, np.ndarray] = {}

        self.current_time = 0.0

    def get_oscillator_states(self, time: float) -> np.ndarray:
        """Get oscillator states at a given time.

        Args:
            time: Time point

        Returns:
            Array of oscillator phases (as sine values)
        """
        phases = 2 * np.pi * self.frequencies * time + self.initial_phases
        return np.sin(phases)

    def encode_duration(self, duration: float, name: Optional[str] = None) -> np.ndarray:
        """Encode a duration as an oscillator pattern.

        Args:
            duration: Duration to encode
            name: Optional name to store pattern

        Returns:
            Pattern at end of duration
        """
        pattern = self.get_oscillator_states(duration)

        if name is not None:
            self.learned_patterns[name] = pattern.copy()

        return pattern

    def estimate_duration(
        self, actual_duration: float, mode: TimingMode = TimingMode.PROSPECTIVE
    ) -> TimingResult:
        """Estimate duration by pattern matching.

        Args:
            actual_duration: True duration
            mode: Timing mode

        Returns:
            Timing result
        """
        # Get target pattern
        target_pattern = self.encode_duration(actual_duration)

        # Find when pattern approximately recurs (with noise)
        # Simulate pattern matching with some error
        noise = np.random.normal(0, 0.1 * actual_duration)
        estimated_duration = actual_duration + noise

        error = estimated_duration - actual_duration
        relative_error = error / actual_duration if actual_duration > 0 else 0

        # Confidence based on how distinctive the pattern is
        pattern_distinctiveness = np.std(target_pattern)
        confidence = min(1.0, pattern_distinctiveness * 2)

        return TimingResult(
            estimated_duration=max(0, estimated_duration),
            actual_duration=actual_duration,
            error=error,
            relative_error=relative_error,
            mode=mode,
            confidence=confidence,
        )

    def detect_learned_duration(self, pattern: np.ndarray) -> Tuple[Optional[str], float]:
        """Detect which learned duration matches a pattern.

        Args:
            pattern: Current oscillator pattern

        Returns:
            Tuple of (duration name, match confidence)
        """
        best_match = None
        best_similarity = 0.0

        for name, learned_pattern in self.learned_patterns.items():
            # Cosine similarity
            similarity = np.dot(pattern, learned_pattern) / (
                np.linalg.norm(pattern) * np.linalg.norm(learned_pattern) + 1e-8
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name

        if best_similarity >= self.detection_threshold:
            return (best_match, best_similarity)
        else:
            return (None, best_similarity)

    def time_until_pattern(self, target_pattern: np.ndarray, max_time: float = 60.0) -> float:
        """Estimate time until oscillators return to a pattern.

        Args:
            target_pattern: Target oscillator pattern
            max_time: Maximum time to search

        Returns:
            Estimated time to pattern match
        """
        dt = 0.01
        t = 0.0
        best_match = 0.0
        best_time = 0.0

        while t < max_time:
            current = self.get_oscillator_states(t)
            similarity = np.dot(current, target_pattern) / (
                np.linalg.norm(current) * np.linalg.norm(target_pattern) + 1e-8
            )

            if similarity > best_match:
                best_match = similarity
                best_time = t

            if similarity >= self.detection_threshold:
                return t

            t += dt

        return best_time

    def reset_phases(self) -> None:
        """Reset oscillator phases (start of new timing)."""
        self.initial_phases = np.random.uniform(0, 2 * np.pi, self.n_oscillators)
        self.current_time = 0.0


class IntervalTimer:
    """Combined interval timing system.

    Uses both pacemaker-accumulator and striatal beat frequency
    models for robust interval timing.
    """

    def __init__(
        self,
        pacemaker: Optional[PacemakerAccumulator] = None,
        sbf: Optional[StriatalBeatFrequency] = None,
        model_weights: Tuple[float, float] = (0.6, 0.4),
    ):
        self.pacemaker = pacemaker or PacemakerAccumulator()
        self.sbf = sbf or StriatalBeatFrequency()
        self.model_weights = model_weights

        self.timing_history: List[TimingResult] = []

    def estimate_duration(
        self,
        actual_duration: float,
        attention: float = 1.0,
        mode: TimingMode = TimingMode.PROSPECTIVE,
    ) -> TimingResult:
        """Estimate duration using both models.

        Args:
            actual_duration: True duration
            attention: Attention level
            mode: Timing mode

        Returns:
            Combined timing result
        """
        # Get estimates from both models
        pa_result = self.pacemaker.estimate_duration(actual_duration, attention, mode)
        sbf_result = self.sbf.estimate_duration(actual_duration, mode)

        # Weighted combination
        w_pa, w_sbf = self.model_weights

        combined_estimate = (
            w_pa * pa_result.estimated_duration + w_sbf * sbf_result.estimated_duration
        )

        combined_confidence = w_pa * pa_result.confidence + w_sbf * sbf_result.confidence

        error = combined_estimate - actual_duration
        relative_error = error / actual_duration if actual_duration > 0 else 0

        result = TimingResult(
            estimated_duration=combined_estimate,
            actual_duration=actual_duration,
            error=error,
            relative_error=relative_error,
            mode=mode,
            confidence=combined_confidence,
        )

        self.timing_history.append(result)

        return result

    def produce_duration(self, target: float) -> TimingResult:
        """Produce a target duration."""
        return self.pacemaker.produce_duration(target)

    def learn_duration(self, name: str, duration: float) -> None:
        """Learn a named duration for later comparison."""
        self.pacemaker.store_reference(name, duration)
        self.sbf.encode_duration(duration, name)

    def compare_duration(self, duration: float, reference_name: str) -> Tuple[str, float]:
        """Compare duration to learned reference."""
        return self.pacemaker.compare_to_reference(duration, reference_name)

    def set_dopamine_level(self, level: float) -> None:
        """Set dopamine level (modulates clock speed)."""
        # Higher dopamine = faster clock = time feels faster
        self.pacemaker.set_clock_speed(level)

    def get_timing_statistics(self) -> Dict:
        """Get statistics from timing history."""
        if not self.timing_history:
            return {"n_trials": 0, "mean_error": 0.0, "mean_relative_error": 0.0}

        errors = [r.error for r in self.timing_history]
        rel_errors = [r.relative_error for r in self.timing_history]

        return {
            "n_trials": len(self.timing_history),
            "mean_error": np.mean(errors),
            "std_error": np.std(errors),
            "mean_relative_error": np.mean(rel_errors),
            "mean_confidence": np.mean([r.confidence for r in self.timing_history]),
        }

    def reset(self) -> None:
        """Reset both models."""
        self.pacemaker.reset()
        self.sbf.reset_phases()
        self.timing_history = []
