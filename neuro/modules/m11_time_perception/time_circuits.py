"""
Neural Circuits for Time Perception

Brain regions involved in temporal processing:
- Insula: Primary interoceptive cortex, processes body signals
- SMA: Supplementary Motor Area, body-environment interaction
- Basal Ganglia: Interval timing, temporal working memory
- Cerebellum: Precise timing (<1s), motor coordination
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum


class TimingScale(Enum):
    """Temporal scales processed by different circuits"""
    MILLISECONDS = "milliseconds"  # <1s, cerebellum
    SECONDS = "seconds"            # 1-60s, basal ganglia
    MINUTES = "minutes"            # >60s, prefrontal
    CIRCADIAN = "circadian"        # ~24h, SCN


@dataclass
class TemporalSignal:
    """A timing signal from a brain region"""
    source: str
    duration_estimate: float      # Estimated duration in seconds
    confidence: float             # 0-1, how confident in estimate
    timestamp: float              # When this estimate was made
    scale: TimingScale
    raw_signal: Optional[np.ndarray] = None


class Insula:
    """Primary interoceptive cortex for time perception.

    The insula processes internal body signals (heartbeat, breathing,
    gut feelings) which provide a fundamental basis for time perception.
    Time is grounded in the rhythm of bodily processes.
    """

    def __init__(
        self,
        heartbeat_rate: float = 70.0,  # BPM
        breathing_rate: float = 15.0,   # breaths per minute
        interoceptive_accuracy: float = 0.7
    ):
        self.heartbeat_rate = heartbeat_rate
        self.breathing_rate = breathing_rate
        self.interoceptive_accuracy = interoceptive_accuracy

        # Internal state tracking
        self.accumulated_heartbeats = 0
        self.accumulated_breaths = 0
        self.current_time = 0.0

        # Body signal history
        self.signal_history: List[Dict] = []

    def process_interoceptive_signal(
        self,
        duration: float,
        arousal_level: float = 0.5
    ) -> TemporalSignal:
        """Process body signals over a duration.

        Args:
            duration: Actual duration in seconds
            arousal_level: Current arousal (0-1), affects perception

        Returns:
            Temporal signal with duration estimate
        """
        # Count heartbeats during interval
        # Higher arousal = faster heartbeat = more beats = longer perceived time
        effective_hr = self.heartbeat_rate * (1 + 0.5 * (arousal_level - 0.5))
        heartbeats = duration * effective_hr / 60.0

        # Count breaths
        effective_br = self.breathing_rate * (1 + 0.3 * (arousal_level - 0.5))
        breaths = duration * effective_br / 60.0

        # Estimate duration from body signals
        # Add noise based on interoceptive accuracy
        noise = np.random.normal(0, 0.1 * (1 - self.interoceptive_accuracy))

        # Duration estimate based on accumulated body signals
        estimated_duration = duration * (1 + noise)

        # Arousal stretches time (high arousal = time feels slower)
        arousal_factor = 1 + 0.2 * (arousal_level - 0.5)
        estimated_duration *= arousal_factor

        self.accumulated_heartbeats += heartbeats
        self.accumulated_breaths += breaths
        self.current_time += duration

        self.signal_history.append({
            'duration': duration,
            'heartbeats': heartbeats,
            'breaths': breaths,
            'arousal': arousal_level,
            'estimate': estimated_duration
        })

        return TemporalSignal(
            source="insula",
            duration_estimate=estimated_duration,
            confidence=self.interoceptive_accuracy * (1 - abs(arousal_level - 0.5)),
            timestamp=self.current_time,
            scale=TimingScale.SECONDS
        )

    def get_body_rhythm(self) -> Dict[str, float]:
        """Get current body rhythm parameters."""
        return {
            'heartbeat_rate': self.heartbeat_rate,
            'breathing_rate': self.breathing_rate,
            'heartbeat_period': 60.0 / self.heartbeat_rate,
            'breathing_period': 60.0 / self.breathing_rate
        }

    def set_arousal_state(self, arousal: float) -> None:
        """Update physiological state based on arousal."""
        # Arousal affects heart and breathing rate
        self.heartbeat_rate = 70.0 * (1 + 0.5 * arousal)
        self.breathing_rate = 15.0 * (1 + 0.3 * arousal)

    def reset(self) -> None:
        """Reset accumulated signals."""
        self.accumulated_heartbeats = 0
        self.accumulated_breaths = 0
        self.current_time = 0.0
        self.signal_history = []


class SMA:
    """Supplementary Motor Area for time perception.

    The SMA supports body-environment interaction and motor timing.
    It tracks time through the rhythm of actions and movements.
    """

    def __init__(
        self,
        motor_precision: float = 0.9,
        action_tempo: float = 1.0  # Actions per second baseline
    ):
        self.motor_precision = motor_precision
        self.action_tempo = action_tempo

        # Motor timing state
        self.action_count = 0
        self.current_time = 0.0
        self.motor_sequences: List[Dict] = []

    def time_motor_sequence(
        self,
        duration: float,
        movement_complexity: float = 0.5
    ) -> TemporalSignal:
        """Time a motor sequence.

        Args:
            duration: Actual duration in seconds
            movement_complexity: Complexity of movement (0-1)

        Returns:
            Temporal signal based on motor timing
        """
        # More complex movements = more temporal markers = better timing
        effective_precision = self.motor_precision * (0.5 + 0.5 * movement_complexity)

        # Count action units during interval
        actions = duration * self.action_tempo

        # Estimate duration from motor count
        noise = np.random.normal(0, 0.1 * (1 - effective_precision))
        estimated_duration = duration * (1 + noise)

        self.action_count += actions
        self.current_time += duration

        self.motor_sequences.append({
            'duration': duration,
            'actions': actions,
            'complexity': movement_complexity,
            'estimate': estimated_duration
        })

        return TemporalSignal(
            source="sma",
            duration_estimate=estimated_duration,
            confidence=effective_precision,
            timestamp=self.current_time,
            scale=TimingScale.SECONDS
        )

    def coordinate_body_environment(
        self,
        external_rhythm: float,
        duration: float
    ) -> TemporalSignal:
        """Synchronize timing with external environmental rhythm.

        Args:
            external_rhythm: External rhythm frequency (Hz)
            duration: Duration to time

        Returns:
            Temporal signal synchronized with environment
        """
        # Count external rhythm cycles
        cycles = duration * external_rhythm

        # Better timing when synchronized with external rhythm
        sync_precision = self.motor_precision * 1.2
        sync_precision = min(1.0, sync_precision)

        noise = np.random.normal(0, 0.05 * (1 - sync_precision))
        estimated_duration = duration * (1 + noise)

        self.current_time += duration

        return TemporalSignal(
            source="sma",
            duration_estimate=estimated_duration,
            confidence=sync_precision,
            timestamp=self.current_time,
            scale=TimingScale.SECONDS
        )

    def reset(self) -> None:
        """Reset motor timing state."""
        self.action_count = 0
        self.current_time = 0.0
        self.motor_sequences = []


class BasalGanglia:
    """Basal Ganglia for interval timing.

    The basal ganglia support interval timing in the seconds-to-minutes
    range and maintain temporal working memory. Dopamine modulates
    the speed of the internal clock.
    """

    def __init__(
        self,
        clock_speed: float = 1.0,  # Relative clock speed
        dopamine_level: float = 1.0,
        memory_capacity: int = 7
    ):
        self.base_clock_speed = clock_speed
        self.dopamine_level = dopamine_level
        self.memory_capacity = memory_capacity

        # Temporal working memory
        self.temporal_memory: List[float] = []

        # Accumulated pulses (pacemaker-accumulator model)
        self.accumulated_pulses = 0.0
        self.current_time = 0.0

        # Reference durations learned from experience
        self.reference_durations: Dict[str, float] = {}

    @property
    def clock_speed(self) -> float:
        """Effective clock speed modulated by dopamine."""
        # Higher dopamine = faster clock = time feels faster
        return self.base_clock_speed * self.dopamine_level

    def time_interval(
        self,
        duration: float,
        attention_level: float = 0.5
    ) -> TemporalSignal:
        """Time an interval using the striatal clock.

        Args:
            duration: Actual duration in seconds
            attention_level: Attention to time (0-1)

        Returns:
            Temporal signal with interval estimate
        """
        # Accumulate pulses at clock speed
        pulses = duration * self.clock_speed * 10  # 10 Hz base rate

        # More attention = more pulses accumulated = longer perceived duration
        attention_factor = 1 + 0.3 * (attention_level - 0.5)
        effective_pulses = pulses * attention_factor

        # Convert pulses back to duration estimate
        estimated_duration = effective_pulses / (self.clock_speed * 10)

        # Add noise (scalar property: variability proportional to duration)
        noise_std = 0.1 * np.sqrt(duration)  # Weber's law
        noise = np.random.normal(0, noise_std)
        estimated_duration += noise

        self.accumulated_pulses += effective_pulses
        self.current_time += duration

        # Store in temporal working memory
        self._store_in_memory(estimated_duration)

        # Confidence decreases with duration (scalar property)
        confidence = 1.0 / (1 + 0.1 * duration)

        return TemporalSignal(
            source="basal_ganglia",
            duration_estimate=max(0, estimated_duration),
            confidence=confidence,
            timestamp=self.current_time,
            scale=TimingScale.SECONDS if duration < 60 else TimingScale.MINUTES
        )

    def _store_in_memory(self, duration: float) -> None:
        """Store duration in temporal working memory."""
        self.temporal_memory.append(duration)
        if len(self.temporal_memory) > self.memory_capacity:
            self.temporal_memory.pop(0)

    def compare_to_reference(
        self,
        duration: float,
        reference_name: str
    ) -> float:
        """Compare duration to a learned reference.

        Args:
            duration: Duration to compare
            reference_name: Name of reference duration

        Returns:
            Ratio of duration to reference (1.0 = equal)
        """
        if reference_name not in self.reference_durations:
            return 1.0

        reference = self.reference_durations[reference_name]
        return duration / reference

    def learn_reference(self, name: str, duration: float) -> None:
        """Learn a reference duration."""
        self.reference_durations[name] = duration

    def set_dopamine_level(self, level: float) -> None:
        """Set dopamine level (affects clock speed)."""
        self.dopamine_level = max(0.1, min(2.0, level))

    def get_temporal_memory(self) -> List[float]:
        """Get contents of temporal working memory."""
        return self.temporal_memory.copy()

    def reset(self) -> None:
        """Reset timing state."""
        self.accumulated_pulses = 0.0
        self.current_time = 0.0
        self.temporal_memory = []


class Cerebellum:
    """Cerebellum for precise sub-second timing.

    The cerebellum handles precise timing in the millisecond range,
    essential for motor coordination and rhythm perception.
    """

    def __init__(
        self,
        precision: float = 0.95,
        max_duration: float = 1.0  # Optimal for durations under 1s
    ):
        self.precision = precision
        self.max_duration = max_duration

        self.current_time = 0.0
        self.timing_events: List[Dict] = []

    def time_precise_interval(
        self,
        duration: float
    ) -> TemporalSignal:
        """Time a precise sub-second interval.

        Args:
            duration: Actual duration in seconds (optimal < 1s)

        Returns:
            Temporal signal with precise estimate
        """
        # Precision degrades for longer durations
        if duration <= self.max_duration:
            effective_precision = self.precision
        else:
            # Exponential degradation beyond optimal range
            degradation = np.exp(-(duration - self.max_duration))
            effective_precision = self.precision * degradation

        # Very precise for short intervals
        noise_std = 0.02 * duration * (1 - effective_precision)
        noise = np.random.normal(0, noise_std)
        estimated_duration = duration + noise

        self.current_time += duration

        self.timing_events.append({
            'duration': duration,
            'estimate': estimated_duration,
            'precision': effective_precision
        })

        return TemporalSignal(
            source="cerebellum",
            duration_estimate=max(0, estimated_duration),
            confidence=effective_precision,
            timestamp=self.current_time,
            scale=TimingScale.MILLISECONDS
        )

    def time_rhythm(
        self,
        beat_interval: float,
        n_beats: int
    ) -> List[float]:
        """Time a rhythmic sequence.

        Args:
            beat_interval: Time between beats
            n_beats: Number of beats

        Returns:
            List of inter-beat intervals (with timing variability)
        """
        intervals = []

        for _ in range(n_beats):
            signal = self.time_precise_interval(beat_interval)
            intervals.append(signal.duration_estimate)

        return intervals

    def synchronize_to_beat(
        self,
        target_interval: float,
        current_estimate: float
    ) -> float:
        """Compute correction to synchronize with a beat.

        Args:
            target_interval: Target beat interval
            current_estimate: Current timing estimate

        Returns:
            Correction factor to apply
        """
        error = target_interval - current_estimate

        # Proportional correction with cerebellar precision
        correction = error * self.precision * 0.5

        return correction

    def reset(self) -> None:
        """Reset timing state."""
        self.current_time = 0.0
        self.timing_events = []


class TimeCircuit:
    """Integrated time perception circuit.

    Combines inputs from all brain regions for comprehensive
    time perception across multiple scales.
    """

    def __init__(
        self,
        insula: Optional[Insula] = None,
        sma: Optional[SMA] = None,
        basal_ganglia: Optional[BasalGanglia] = None,
        cerebellum: Optional[Cerebellum] = None
    ):
        self.insula = insula or Insula()
        self.sma = sma or SMA()
        self.basal_ganglia = basal_ganglia or BasalGanglia()
        self.cerebellum = cerebellum or Cerebellum()

        # Integration weights (can be learned/adjusted)
        self.weights = {
            'insula': 0.25,
            'sma': 0.25,
            'basal_ganglia': 0.35,
            'cerebellum': 0.15
        }

        self.current_time = 0.0

    def estimate_duration(
        self,
        actual_duration: float,
        arousal: float = 0.5,
        attention: float = 0.5,
        movement: bool = False
    ) -> TemporalSignal:
        """Estimate duration using all circuits.

        Args:
            actual_duration: True duration in seconds
            arousal: Arousal level (0-1)
            attention: Attention to time (0-1)
            movement: Whether movement is involved

        Returns:
            Integrated temporal estimate
        """
        signals = []

        # Get estimates from each region
        insula_signal = self.insula.process_interoceptive_signal(
            actual_duration, arousal
        )
        signals.append(('insula', insula_signal))

        bg_signal = self.basal_ganglia.time_interval(
            actual_duration, attention
        )
        signals.append(('basal_ganglia', bg_signal))

        if movement:
            sma_signal = self.sma.time_motor_sequence(
                actual_duration, movement_complexity=0.5
            )
            signals.append(('sma', sma_signal))

        if actual_duration < 1.0:
            cerebellum_signal = self.cerebellum.time_precise_interval(
                actual_duration
            )
            signals.append(('cerebellum', cerebellum_signal))

        # Weighted integration
        total_weight = 0.0
        weighted_estimate = 0.0
        weighted_confidence = 0.0

        for source, signal in signals:
            weight = self.weights[source] * signal.confidence
            weighted_estimate += weight * signal.duration_estimate
            weighted_confidence += weight * signal.confidence
            total_weight += weight

        if total_weight > 0:
            final_estimate = weighted_estimate / total_weight
            final_confidence = weighted_confidence / total_weight
        else:
            final_estimate = actual_duration
            final_confidence = 0.5

        self.current_time += actual_duration

        # Determine scale
        if actual_duration < 1.0:
            scale = TimingScale.MILLISECONDS
        elif actual_duration < 60:
            scale = TimingScale.SECONDS
        else:
            scale = TimingScale.MINUTES

        return TemporalSignal(
            source="integrated",
            duration_estimate=final_estimate,
            confidence=final_confidence,
            timestamp=self.current_time,
            scale=scale
        )

    def set_dopamine(self, level: float) -> None:
        """Set dopamine level (affects basal ganglia clock)."""
        self.basal_ganglia.set_dopamine_level(level)

    def set_arousal(self, level: float) -> None:
        """Set arousal level (affects insula processing)."""
        self.insula.set_arousal_state(level)

    def get_statistics(self) -> Dict:
        """Get timing statistics from all circuits."""
        return {
            'insula_signals': len(self.insula.signal_history),
            'sma_sequences': len(self.sma.motor_sequences),
            'bg_memory': self.basal_ganglia.get_temporal_memory(),
            'cerebellum_events': len(self.cerebellum.timing_events),
            'current_time': self.current_time
        }

    def reset(self) -> None:
        """Reset all circuits."""
        self.insula.reset()
        self.sma.reset()
        self.basal_ganglia.reset()
        self.cerebellum.reset()
        self.current_time = 0.0
