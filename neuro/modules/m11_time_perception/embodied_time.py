"""
Embodied Time Perception

Key insight from 2025 research:
Time is not perceived abstractly - it's grounded in embodied experience.

Time perception relies on:
- Body-environment interaction (SMA)
- Internal body signals (Insula)
- Motor rhythms and actions
- Interoceptive signals (heartbeat, breathing)
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class BodyState:
    """Current state of body rhythms"""

    heart_rate: float = 70.0  # BPM
    breathing_rate: float = 15.0  # breaths/min
    body_temperature: float = 37.0  # Celsius
    metabolic_rate: float = 1.0  # Relative to baseline
    movement_level: float = 0.0  # 0 = still, 1 = active


@dataclass
class InteroceptiveSignal:
    """Signal from internal body monitoring"""

    source: str  # heartbeat, breathing, gut, etc.
    timestamp: float
    value: float
    count: int  # Accumulated count


class InteroceptiveTimer:
    """Time perception based on internal body signals.

    The insula tracks heartbeats, breaths, and other body rhythms
    which serve as an internal clock for time perception.
    People with better interoceptive awareness are better at timing.
    """

    def __init__(self, interoceptive_accuracy: float = 0.7, body_state: Optional[BodyState] = None):
        self.interoceptive_accuracy = interoceptive_accuracy
        self.body_state = body_state or BodyState()

        # Accumulated counts
        self.heartbeat_count = 0
        self.breath_count = 0
        self.current_time = 0.0

        # Signal history
        self.signals: List[InteroceptiveSignal] = []

    def count_heartbeats(self, duration: float) -> int:
        """Count heartbeats during an interval.

        Args:
            duration: Duration in seconds

        Returns:
            Number of perceived heartbeats
        """
        # Actual heartbeats
        actual_beats = duration * self.body_state.heart_rate / 60.0

        # Perception error based on interoceptive accuracy
        error = np.random.normal(0, actual_beats * (1 - self.interoceptive_accuracy) * 0.2)
        perceived_beats = int(actual_beats + error)

        self.heartbeat_count += perceived_beats

        self.signals.append(
            InteroceptiveSignal(
                source="heartbeat",
                timestamp=self.current_time,
                value=perceived_beats,
                count=self.heartbeat_count,
            )
        )

        return perceived_beats

    def count_breaths(self, duration: float) -> int:
        """Count breaths during an interval.

        Args:
            duration: Duration in seconds

        Returns:
            Number of perceived breaths
        """
        actual_breaths = duration * self.body_state.breathing_rate / 60.0

        error = np.random.normal(0, actual_breaths * (1 - self.interoceptive_accuracy) * 0.15)
        perceived_breaths = int(actual_breaths + error)

        self.breath_count += perceived_breaths

        self.signals.append(
            InteroceptiveSignal(
                source="breathing",
                timestamp=self.current_time,
                value=perceived_breaths,
                count=self.breath_count,
            )
        )

        return perceived_breaths

    def estimate_duration_from_heartbeats(self, actual_duration: float) -> float:
        """Estimate duration by counting heartbeats.

        Args:
            actual_duration: True duration in seconds

        Returns:
            Estimated duration
        """
        perceived_beats = self.count_heartbeats(actual_duration)

        # Convert back to duration using baseline heart rate assumption
        # (people often use ~70 BPM as reference)
        reference_hr = 70.0
        estimated_duration = perceived_beats / (reference_hr / 60.0)

        self.current_time += actual_duration

        return max(0, estimated_duration)

    def estimate_duration_from_breathing(self, actual_duration: float) -> float:
        """Estimate duration by counting breaths.

        Args:
            actual_duration: True duration in seconds

        Returns:
            Estimated duration
        """
        perceived_breaths = self.count_breaths(actual_duration)

        # Convert back using baseline breathing rate
        reference_br = 15.0
        estimated_duration = perceived_breaths / (reference_br / 60.0)

        self.current_time += actual_duration

        return max(0, estimated_duration)

    def combined_estimate(self, actual_duration: float) -> float:
        """Combine heartbeat and breathing for duration estimate.

        Args:
            actual_duration: True duration

        Returns:
            Combined estimate
        """
        hb_estimate = self.estimate_duration_from_heartbeats(actual_duration)
        br_estimate = self.estimate_duration_from_breathing(actual_duration)

        # Weight by reliability (heartbeat often more reliable for short intervals)
        if actual_duration < 30:
            weight_hb = 0.7
        else:
            weight_hb = 0.5

        combined = weight_hb * hb_estimate + (1 - weight_hb) * br_estimate

        return combined

    def set_body_state(self, state: BodyState) -> None:
        """Update body state."""
        self.body_state = state

    def set_arousal(self, arousal: float) -> None:
        """Adjust body state for arousal level.

        Args:
            arousal: Arousal level (0-1)
        """
        # Arousal increases heart and breathing rate
        self.body_state.heart_rate = 70 * (1 + 0.5 * arousal)
        self.body_state.breathing_rate = 15 * (1 + 0.3 * arousal)

    def reset(self) -> None:
        """Reset accumulated counts."""
        self.heartbeat_count = 0
        self.breath_count = 0
        self.current_time = 0.0
        self.signals = []


class MotorTimer:
    """Time perception based on motor actions and movements.

    The SMA tracks body-environment interaction through movement.
    Actions serve as temporal markers, and motor planning
    involves implicit timing.
    """

    def __init__(
        self,
        motor_precision: float = 0.85,
        base_tempo: float = 2.0,  # Actions per second
    ):
        self.motor_precision = motor_precision
        self.base_tempo = base_tempo

        self.action_count = 0
        self.current_time = 0.0
        self.movement_history: List[Dict] = []

    def time_through_action(self, duration: float, action_rate: Optional[float] = None) -> float:
        """Estimate duration through number of actions.

        Args:
            duration: Actual duration
            action_rate: Actions per second (uses base_tempo if None)

        Returns:
            Estimated duration
        """
        if action_rate is None:
            action_rate = self.base_tempo

        # Count actions during interval
        actions = duration * action_rate

        # Add motor timing noise
        noise = np.random.normal(0, actions * (1 - self.motor_precision) * 0.1)
        perceived_actions = actions + noise

        # Convert back to duration
        estimated_duration = perceived_actions / self.base_tempo

        self.action_count += int(perceived_actions)
        self.current_time += duration

        self.movement_history.append(
            {"duration": duration, "actions": perceived_actions, "estimate": estimated_duration}
        )

        return max(0, estimated_duration)

    def time_movement_sequence(
        self,
        movements: List[float],  # Duration of each movement
    ) -> Tuple[float, float]:
        """Time a sequence of movements.

        Args:
            movements: List of movement durations

        Returns:
            Tuple of (total estimated duration, actual duration)
        """
        actual_total = sum(movements)
        estimated_total = 0.0

        for mov_duration in movements:
            estimated_total += self.time_through_action(mov_duration)

        return estimated_total, actual_total

    def tap_rhythm(self, target_interval: float, n_taps: int) -> List[float]:
        """Produce taps at a target rhythm.

        Args:
            target_interval: Desired interval between taps
            n_taps: Number of taps to produce

        Returns:
            List of actual inter-tap intervals
        """
        intervals = []

        for _ in range(n_taps):
            # Motor timing noise (Weber's law applies)
            noise_std = 0.05 * target_interval * (1 - self.motor_precision)
            noise = np.random.normal(0, noise_std)
            actual_interval = target_interval + noise

            intervals.append(max(0.01, actual_interval))

        return intervals

    def synchronize_to_beat(self, beat_interval: float, n_beats: int) -> Tuple[List[float], float]:
        """Attempt to synchronize taps to an external beat.

        Args:
            beat_interval: External beat interval
            n_beats: Number of beats to synchronize to

        Returns:
            Tuple of (tap intervals, mean asynchrony)
        """
        taps = self.tap_rhythm(beat_interval, n_beats)

        # Calculate asynchrony (difference from target)
        asynchronies = [abs(tap - beat_interval) for tap in taps]
        mean_asynchrony = np.mean(asynchronies)

        return taps, mean_asynchrony

    def reset(self) -> None:
        """Reset motor timing state."""
        self.action_count = 0
        self.current_time = 0.0
        self.movement_history = []


class BodyEnvironmentCoupler:
    """Couples body rhythms with environmental rhythms.

    Time perception is enhanced when body and environment
    are synchronized (entrainment).
    """

    def __init__(self):
        self.coupling_strength = 0.5
        self.environmental_rhythm: Optional[float] = None
        self.body_rhythm: float = 1.0  # Hz

        self.entrainment_history: List[Dict] = []

    def set_environmental_rhythm(self, frequency: float) -> None:
        """Set external rhythm to entrain to.

        Args:
            frequency: External rhythm frequency in Hz
        """
        self.environmental_rhythm = frequency

    def set_body_rhythm(self, frequency: float) -> None:
        """Set internal body rhythm.

        Args:
            frequency: Body rhythm frequency in Hz
        """
        self.body_rhythm = frequency

    def compute_entrainment(self) -> float:
        """Compute degree of entrainment between body and environment.

        Returns:
            Entrainment factor (0 = none, 1 = perfect)
        """
        if self.environmental_rhythm is None:
            return 0.0

        # Entrainment is strongest when frequencies are similar or harmonic
        ratio = self.body_rhythm / self.environmental_rhythm

        # Check for harmonic relationships
        harmonics = [0.5, 1.0, 2.0, 3.0]  # 1:2, 1:1, 2:1, 3:1
        min_distance = min(abs(ratio - h) for h in harmonics)

        # Entrainment factor
        entrainment = np.exp(-min_distance * 5)

        return entrainment

    def modulate_timing(
        self,
        duration: float,
        body_rhythm: Optional[float] = None,
        env_rhythm: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Modulate time perception based on body-environment coupling.

        Args:
            duration: Actual duration
            body_rhythm: Body rhythm frequency
            env_rhythm: Environmental rhythm frequency

        Returns:
            Tuple of (estimated duration, entrainment factor)
        """
        if body_rhythm is not None:
            self.body_rhythm = body_rhythm
        if env_rhythm is not None:
            self.environmental_rhythm = env_rhythm

        entrainment = self.compute_entrainment()

        # High entrainment improves timing accuracy
        # Low entrainment increases variability
        noise_scale = 0.15 * (1 - entrainment)
        noise = np.random.normal(0, duration * noise_scale)

        estimated_duration = duration + noise

        self.entrainment_history.append(
            {"duration": duration, "entrainment": entrainment, "estimate": estimated_duration}
        )

        return max(0, estimated_duration), entrainment

    def reset(self) -> None:
        """Reset coupler state."""
        self.environmental_rhythm = None
        self.entrainment_history = []


class EmbodiedTimeSystem:
    """Complete embodied time perception system.

    Integrates interoceptive, motor, and coupling mechanisms
    for time perception grounded in body experience.
    """

    def __init__(
        self,
        interoceptive: Optional[InteroceptiveTimer] = None,
        motor: Optional[MotorTimer] = None,
        coupler: Optional[BodyEnvironmentCoupler] = None,
        integration_weights: Tuple[float, float, float] = (0.4, 0.35, 0.25),
    ):
        self.interoceptive = interoceptive or InteroceptiveTimer()
        self.motor = motor or MotorTimer()
        self.coupler = coupler or BodyEnvironmentCoupler()
        self.weights = integration_weights  # (intero, motor, coupling)

    def estimate_duration(
        self,
        actual_duration: float,
        movement_present: bool = False,
        external_rhythm: Optional[float] = None,
    ) -> Tuple[float, Dict]:
        """Estimate duration using embodied signals.

        Args:
            actual_duration: True duration
            movement_present: Whether movement is involved
            external_rhythm: External rhythm to sync with

        Returns:
            Tuple of (estimated duration, component estimates)
        """
        estimates = {}
        active_weights = []

        # Interoceptive estimate
        intero_estimate = self.interoceptive.combined_estimate(actual_duration)
        estimates["interoceptive"] = intero_estimate
        active_weights.append(("interoceptive", self.weights[0]))

        # Motor estimate (only if moving)
        if movement_present:
            motor_estimate = self.motor.time_through_action(actual_duration)
            estimates["motor"] = motor_estimate
            active_weights.append(("motor", self.weights[1]))

        # Coupling estimate (only if external rhythm)
        if external_rhythm is not None:
            coupling_estimate, entrainment = self.coupler.modulate_timing(
                actual_duration, env_rhythm=external_rhythm
            )
            estimates["coupling"] = coupling_estimate
            estimates["entrainment"] = entrainment
            active_weights.append(("coupling", self.weights[2]))

        # Weighted combination
        total_weight = sum(w for _, w in active_weights)
        if total_weight > 0:
            final_estimate = sum(
                estimates[name] * weight / total_weight for name, weight in active_weights
            )
        else:
            final_estimate = actual_duration

        return final_estimate, estimates

    def set_body_state(self, state: BodyState) -> None:
        """Update body state for all components."""
        self.interoceptive.set_body_state(state)

        # Adjust motor tempo based on metabolic rate
        self.motor.base_tempo = 2.0 * state.metabolic_rate

        # Adjust body rhythm for coupling
        # Heart rate as body rhythm
        self.coupler.body_rhythm = state.heart_rate / 60.0

    def set_arousal(self, arousal: float) -> None:
        """Set arousal level."""
        self.interoceptive.set_arousal(arousal)

    def get_statistics(self) -> Dict:
        """Get embodied timing statistics."""
        return {
            "heartbeat_count": self.interoceptive.heartbeat_count,
            "breath_count": self.interoceptive.breath_count,
            "action_count": self.motor.action_count,
            "n_entrainment_events": len(self.coupler.entrainment_history),
        }

    def reset(self) -> None:
        """Reset all components."""
        self.interoceptive.reset()
        self.motor.reset()
        self.coupler.reset()
