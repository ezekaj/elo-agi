"""
Sleep Stage Controller and Oscillation Generator

Implements the distinct sleep stages with their characteristic
neural oscillations and memory functions:
- NREM1: Sleep spindles begin (12-16 Hz)
- NREM2: Sleep spindles + K-complexes
- SWS: Slow waves (0.5-4 Hz), sharp-wave ripples (80-120 Hz)
- REM: Theta waves (4-8 Hz)
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field
from enum import Enum


class SleepStage(Enum):
    """Sleep stages with distinct neural signatures"""

    WAKE = "wake"
    NREM1 = "nrem1"  # Light sleep, transition stage
    NREM2 = "nrem2"  # Light sleep, sleep spindles
    SWS = "sws"  # Slow-wave sleep (N3), deep sleep
    REM = "rem"  # Rapid eye movement, dreaming


@dataclass
class StageProperties:
    """Properties for each sleep stage"""

    oscillation_freq: Tuple[float, float]  # Hz range (min, max)
    duration_typical: float  # Typical duration in minutes
    memory_function: str  # Primary memory role
    plasticity_mode: str  # "encoding" or "consolidation"
    arousal_threshold: float  # How hard to wake (0-1)

    @classmethod
    def for_stage(cls, stage: SleepStage) -> "StageProperties":
        """Get properties for a specific stage"""
        properties = {
            SleepStage.WAKE: cls(
                oscillation_freq=(8.0, 30.0),  # Alpha-beta
                duration_typical=0.0,
                memory_function="encoding",
                plasticity_mode="encoding",
                arousal_threshold=0.0,
            ),
            SleepStage.NREM1: cls(
                oscillation_freq=(4.0, 8.0),  # Theta
                duration_typical=5.0,
                memory_function="replay_initiation",
                plasticity_mode="consolidation",
                arousal_threshold=0.3,
            ),
            SleepStage.NREM2: cls(
                oscillation_freq=(12.0, 16.0),  # Sleep spindles
                duration_typical=25.0,
                memory_function="memory_replay",
                plasticity_mode="consolidation",
                arousal_threshold=0.5,
            ),
            SleepStage.SWS: cls(
                oscillation_freq=(0.5, 4.0),  # Delta/slow waves
                duration_typical=20.0,
                memory_function="systems_consolidation",
                plasticity_mode="consolidation",
                arousal_threshold=0.9,
            ),
            SleepStage.REM: cls(
                oscillation_freq=(4.0, 8.0),  # Theta
                duration_typical=20.0,
                memory_function="emotional_processing",
                plasticity_mode="consolidation",
                arousal_threshold=0.4,
            ),
        }
        return properties[stage]


@dataclass
class OscillationEvent:
    """A discrete oscillatory event (spindle, ripple, etc.)"""

    event_type: str
    start_time: float
    duration: float
    frequency: float
    amplitude: float
    waveform: Optional[np.ndarray] = None


class OscillationGenerator:
    """Generates sleep-specific neural oscillations.

    Each sleep stage has characteristic oscillatory patterns:
    - Sleep spindles (12-16 Hz): Memory consolidation markers
    - Slow oscillations (0.5-4 Hz): Hippocampal-cortical dialogue
    - Sharp-wave ripples (80-120 Hz): Memory replay events
    - Theta waves (4-8 Hz): REM sleep signature
    """

    def __init__(self, sample_rate: float = 1000.0):
        """Initialize oscillation generator.

        Args:
            sample_rate: Samples per second for waveform generation
        """
        self.sample_rate = sample_rate
        self.current_time = 0.0

        # Spindle parameters
        self.spindle_freq_range = (12.0, 16.0)
        self.spindle_duration_range = (0.5, 2.0)  # seconds

        # Slow oscillation parameters
        self.slow_osc_freq_range = (0.5, 1.0)

        # Sharp-wave ripple parameters
        self.swr_freq_range = (80.0, 120.0)
        self.swr_duration = 0.05  # 50ms typical

        # Theta parameters
        self.theta_freq_range = (4.0, 8.0)

    def generate_sine_wave(
        self, frequency: float, duration: float, amplitude: float = 1.0, phase: float = 0.0
    ) -> np.ndarray:
        """Generate a simple sine wave."""
        n_samples = int(duration * self.sample_rate)
        t = np.arange(n_samples) / self.sample_rate
        return amplitude * np.sin(2 * np.pi * frequency * t + phase)

    def sleep_spindle(
        self, duration: Optional[float] = None, amplitude: float = 1.0
    ) -> OscillationEvent:
        """Generate a sleep spindle (12-16 Hz burst).

        Sleep spindles are waxing-waning oscillations associated with
        memory consolidation during NREM sleep.
        """
        if duration is None:
            duration = np.random.uniform(*self.spindle_duration_range)

        freq = np.random.uniform(*self.spindle_freq_range)
        n_samples = int(duration * self.sample_rate)
        t = np.arange(n_samples) / self.sample_rate

        # Spindles have a characteristic envelope (waxing-waning)
        envelope = np.sin(np.pi * t / duration) ** 2
        waveform = amplitude * envelope * np.sin(2 * np.pi * freq * t)

        event = OscillationEvent(
            event_type="spindle",
            start_time=self.current_time,
            duration=duration,
            frequency=freq,
            amplitude=amplitude,
            waveform=waveform,
        )
        self.current_time += duration
        return event

    def slow_oscillation(self, duration: float = 1.0, amplitude: float = 1.0) -> OscillationEvent:
        """Generate slow oscillation (0.5-4 Hz).

        Slow oscillations coordinate hippocampal-cortical dialogue
        during SWS, grouping spindles and ripples.
        """
        freq = np.random.uniform(*self.slow_osc_freq_range)
        waveform = self.generate_sine_wave(freq, duration, amplitude)

        event = OscillationEvent(
            event_type="slow_oscillation",
            start_time=self.current_time,
            duration=duration,
            frequency=freq,
            amplitude=amplitude,
            waveform=waveform,
        )
        self.current_time += duration
        return event

    def sharp_wave_ripple(self, amplitude: float = 1.0) -> OscillationEvent:
        """Generate a sharp-wave ripple (80-120 Hz, ~50ms).

        Sharp-wave ripples occur during SWS and are associated with
        memory replay in compressed time.
        """
        freq = np.random.uniform(*self.swr_freq_range)
        duration = self.swr_duration
        n_samples = int(duration * self.sample_rate)
        t = np.arange(n_samples) / self.sample_rate

        # Ripples have a sharp envelope
        envelope = np.exp(-((t - duration / 2) ** 2) / (2 * (duration / 6) ** 2))
        waveform = amplitude * envelope * np.sin(2 * np.pi * freq * t)

        event = OscillationEvent(
            event_type="sharp_wave_ripple",
            start_time=self.current_time,
            duration=duration,
            frequency=freq,
            amplitude=amplitude,
            waveform=waveform,
        )
        self.current_time += duration
        return event

    def theta_wave(self, duration: float = 1.0, amplitude: float = 1.0) -> OscillationEvent:
        """Generate theta wave (4-8 Hz).

        Theta oscillations dominate during REM sleep and are associated
        with emotional memory processing and schema formation.
        """
        freq = np.random.uniform(*self.theta_freq_range)
        waveform = self.generate_sine_wave(freq, duration, amplitude)

        event = OscillationEvent(
            event_type="theta_wave",
            start_time=self.current_time,
            duration=duration,
            frequency=freq,
            amplitude=amplitude,
            waveform=waveform,
        )
        self.current_time += duration
        return event

    def k_complex(self, amplitude: float = 2.0) -> OscillationEvent:
        """Generate a K-complex.

        K-complexes are sharp negative-positive deflections that
        occur during NREM2 sleep.
        """
        duration = 0.5
        n_samples = int(duration * self.sample_rate)
        t = np.arange(n_samples) / self.sample_rate

        # K-complex: sharp negative followed by positive
        neg_peak = -amplitude * np.exp(-((t - 0.15) ** 2) / 0.005)
        pos_peak = (amplitude * 0.7) * np.exp(-((t - 0.3) ** 2) / 0.01)
        waveform = neg_peak + pos_peak

        event = OscillationEvent(
            event_type="k_complex",
            start_time=self.current_time,
            duration=duration,
            frequency=2.0,  # ~2 Hz equivalent
            amplitude=amplitude,
            waveform=waveform,
        )
        self.current_time += duration
        return event

    def generate_stage_activity(self, stage: SleepStage, duration: float) -> List[OscillationEvent]:
        """Generate oscillatory activity for a sleep stage.

        Args:
            stage: The sleep stage
            duration: Duration in seconds

        Returns:
            List of oscillation events
        """
        events = []
        elapsed = 0.0

        if stage == SleepStage.WAKE:
            # Wake: continuous alpha-beta activity
            while elapsed < duration:
                evt_duration = min(1.0, duration - elapsed)
                freq = np.random.uniform(8.0, 30.0)
                waveform = self.generate_sine_wave(freq, evt_duration)
                events.append(
                    OscillationEvent(
                        event_type="wake_activity",
                        start_time=self.current_time,
                        duration=evt_duration,
                        frequency=freq,
                        amplitude=1.0,
                        waveform=waveform,
                    )
                )
                elapsed += evt_duration
                self.current_time += evt_duration

        elif stage == SleepStage.NREM1:
            # NREM1: Theta with occasional spindles beginning
            while elapsed < duration:
                if np.random.random() < 0.1:  # 10% chance of early spindle
                    evt = self.sleep_spindle(amplitude=0.5)
                else:
                    evt = self.theta_wave(duration=min(1.0, duration - elapsed))
                events.append(evt)
                elapsed += evt.duration

        elif stage == SleepStage.NREM2:
            # NREM2: Spindles and K-complexes
            while elapsed < duration:
                r = np.random.random()
                if r < 0.3:  # 30% spindles
                    evt = self.sleep_spindle()
                elif r < 0.4:  # 10% K-complexes
                    evt = self.k_complex()
                else:  # Background theta-delta
                    evt = self.theta_wave(duration=min(0.5, duration - elapsed))
                events.append(evt)
                elapsed += evt.duration

        elif stage == SleepStage.SWS:
            # SWS: Slow oscillations with nested spindles and ripples
            while elapsed < duration:
                # Generate slow oscillation with nested events
                slow_osc = self.slow_oscillation(duration=min(2.0, duration - elapsed))
                events.append(slow_osc)
                elapsed += slow_osc.duration

                # Nested spindles during up-states
                if np.random.random() < 0.4:
                    spindle = self.sleep_spindle(duration=0.5, amplitude=0.7)
                    events.append(spindle)
                    elapsed += spindle.duration

                # Sharp-wave ripples
                if np.random.random() < 0.3:
                    swr = self.sharp_wave_ripple()
                    events.append(swr)
                    elapsed += swr.duration

        elif stage == SleepStage.REM:
            # REM: Continuous theta with occasional bursts
            while elapsed < duration:
                evt = self.theta_wave(duration=min(1.0, duration - elapsed))
                events.append(evt)
                elapsed += evt.duration

        return events

    def reset(self) -> None:
        """Reset generator state"""
        self.current_time = 0.0


class SleepStageController:
    """Manages sleep stage transitions and timing.

    Implements the ultradian rhythm of sleep (~90 minute cycles)
    with changing proportions of SWS and REM across the night.
    """

    def __init__(self):
        self.current_stage = SleepStage.WAKE
        self.stage_start_time = 0.0
        self.total_time = 0.0
        self.cycle_count = 0

        # Oscillation generator
        self.oscillation_gen = OscillationGenerator()

        # Stage history for analysis
        self.stage_history: List[Tuple[SleepStage, float, float]] = []

        # Typical cycle duration (~90 minutes)
        self.cycle_duration = 90.0  # minutes

        # Stage sequence within a cycle (simplified)
        self.stage_sequence = [
            SleepStage.NREM1,
            SleepStage.NREM2,
            SleepStage.SWS,
            SleepStage.NREM2,
            SleepStage.REM,
        ]
        self.sequence_index = 0

    def transition(self, target_stage: SleepStage) -> bool:
        """Initiate transition to a new sleep stage.

        Args:
            target_stage: Stage to transition to

        Returns:
            True if transition successful
        """
        # Record previous stage
        if self.current_stage != target_stage:
            duration = self.total_time - self.stage_start_time
            self.stage_history.append((self.current_stage, self.stage_start_time, duration))

            # Detect cycle completion: transitioning from REM to NREM1
            if self.current_stage == SleepStage.REM and target_stage == SleepStage.NREM1:
                self.cycle_count += 1

            self.current_stage = target_stage
            self.stage_start_time = self.total_time
            return True
        return False

    def advance_time(self, dt: float) -> Optional[SleepStage]:
        """Advance time and check for stage transition.

        Args:
            dt: Time step in minutes

        Returns:
            New stage if transition occurred, None otherwise
        """
        self.total_time += dt
        stage_duration = self.total_time - self.stage_start_time

        # Get typical duration for current stage
        props = StageProperties.for_stage(self.current_stage)

        # Check if stage should transition
        # Duration varies by cycle (more REM later in night)
        adjusted_duration = self._adjust_duration_for_cycle(
            props.duration_typical, self.current_stage
        )

        if stage_duration >= adjusted_duration:
            return self._next_stage()

        return None

    def _adjust_duration_for_cycle(self, base_duration: float, stage: SleepStage) -> float:
        """Adjust stage duration based on cycle number.

        SWS decreases across the night, REM increases.
        """
        if stage == SleepStage.SWS:
            # SWS decreases: 100% -> 50% -> 25% -> ...
            factor = 0.5**self.cycle_count
            return base_duration * max(factor, 0.1)
        elif stage == SleepStage.REM:
            # REM increases: starts short, grows
            factor = 1.0 + 0.5 * self.cycle_count
            return base_duration * min(factor, 3.0)
        return base_duration

    def _next_stage(self) -> SleepStage:
        """Determine and transition to next stage."""
        self.sequence_index += 1

        if self.sequence_index >= len(self.stage_sequence):
            # Complete cycle, start new one
            self.sequence_index = 0
            # Note: cycle_count is incremented in transition() when going from REM to NREM1

        next_stage = self.stage_sequence[self.sequence_index]
        self.transition(next_stage)
        return next_stage

    def start_sleep(self) -> SleepStage:
        """Begin sleep from wake state."""
        self.cycle_count = 0
        self.sequence_index = 0
        self.transition(SleepStage.NREM1)
        return self.current_stage

    def wake_up(self) -> None:
        """Transition to wake state."""
        self.transition(SleepStage.WAKE)
        self.sequence_index = 0

    def generate_oscillation(self, duration: float) -> List[OscillationEvent]:
        """Generate oscillatory activity for current stage.

        Args:
            duration: Duration in seconds

        Returns:
            List of oscillation events
        """
        return self.oscillation_gen.generate_stage_activity(self.current_stage, duration)

    def get_stage_properties(self) -> StageProperties:
        """Get properties of current stage."""
        return StageProperties.for_stage(self.current_stage)

    def get_time_in_stage(self) -> float:
        """Get time spent in current stage (minutes)."""
        return self.total_time - self.stage_start_time

    def get_total_sleep_time(self) -> float:
        """Get total time since sleep started (minutes)."""
        return self.total_time

    def get_stage_summary(self) -> Dict[SleepStage, float]:
        """Get summary of time spent in each stage."""
        summary = {stage: 0.0 for stage in SleepStage}

        for stage, start, duration in self.stage_history:
            summary[stage] += duration

        # Add current stage
        current_duration = self.total_time - self.stage_start_time
        summary[self.current_stage] += current_duration

        return summary

    def reset(self) -> None:
        """Reset controller to initial state."""
        self.current_stage = SleepStage.WAKE
        self.stage_start_time = 0.0
        self.total_time = 0.0
        self.cycle_count = 0
        self.sequence_index = 0
        self.stage_history = []
        self.oscillation_gen.reset()
