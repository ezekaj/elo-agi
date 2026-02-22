"""
Temporal Integration System

Orchestrates all components of time perception:
- Neural circuits (insula, SMA, basal ganglia, cerebellum)
- Interval timing models
- Modulation factors (emotion, attention, dopamine, age)
- Embodied time mechanisms
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .time_circuits import TimeCircuit, TemporalSignal, TimingScale
from .interval_timing import IntervalTimer, TimingMode, TimingResult
from .time_modulation import TimeModulationSystem, EmotionalState, ModulationEffect
from .embodied_time import EmbodiedTimeSystem, BodyState


@dataclass
class TemporalEstimate:
    """Complete temporal estimate with all components"""

    perceived_duration: float
    actual_duration: float
    error: float
    relative_error: float
    confidence: float
    scale: TimingScale

    # Component contributions
    circuit_estimate: float
    interval_estimate: float
    embodied_estimate: float
    modulation_ratio: float

    # Active factors
    emotional_state: Optional[str] = None
    attention_level: Optional[float] = None
    dopamine_level: Optional[float] = None
    age: Optional[int] = None


class SubjectiveTimeSystem:
    """Computes subjective time experience.

    Integrates neural, cognitive, and embodied components
    to produce the subjective experience of duration.
    """

    def __init__(
        self,
        circuits: Optional[TimeCircuit] = None,
        interval_timer: Optional[IntervalTimer] = None,
        modulation: Optional[TimeModulationSystem] = None,
        embodied: Optional[EmbodiedTimeSystem] = None,
    ):
        self.circuits = circuits or TimeCircuit()
        self.interval_timer = interval_timer or IntervalTimer()
        self.modulation = modulation or TimeModulationSystem()
        self.embodied = embodied or EmbodiedTimeSystem()

        # Integration weights
        self.component_weights = {"circuits": 0.30, "interval": 0.35, "embodied": 0.35}

        # History
        self.estimate_history: List[TemporalEstimate] = []

    def estimate_duration(
        self,
        actual_duration: float,
        emotional_state: Optional[EmotionalState] = None,
        attention: float = 0.5,
        dopamine: float = 1.0,
        age: int = 30,
        movement: bool = False,
        external_rhythm: Optional[float] = None,
        mode: TimingMode = TimingMode.PROSPECTIVE,
    ) -> TemporalEstimate:
        """Estimate subjective duration with all factors.

        Args:
            actual_duration: True duration in seconds
            emotional_state: Current emotional state
            attention: Attention to time (0-1)
            dopamine: Dopamine level relative to baseline
            age: Age in years
            movement: Whether movement is involved
            external_rhythm: External rhythm frequency (Hz)
            mode: Prospective vs retrospective timing

        Returns:
            Complete temporal estimate
        """
        # Get base estimates from each system

        # 1. Neural circuits
        circuit_signal = self.circuits.estimate_duration(
            actual_duration,
            arousal=0.5
            if emotional_state is None
            else self.modulation.emotional.arousal_effects.get(emotional_state, (0.5, 1.0))[0],
            attention=attention,
            movement=movement,
        )
        circuit_estimate = circuit_signal.duration_estimate

        # 2. Interval timing
        interval_result = self.interval_timer.estimate_duration(actual_duration, attention, mode)
        interval_estimate = interval_result.estimated_duration

        # 3. Embodied timing
        embodied_estimate, _ = self.embodied.estimate_duration(
            actual_duration, movement_present=movement, external_rhythm=external_rhythm
        )

        # Combine base estimates
        base_estimate = (
            self.component_weights["circuits"] * circuit_estimate
            + self.component_weights["interval"] * interval_estimate
            + self.component_weights["embodied"] * embodied_estimate
        )

        # 4. Apply modulation factors
        modulated_duration, effects = self.modulation.modulate_duration(
            base_estimate,
            emotional_state=emotional_state,
            attention=attention,
            dopamine_level=dopamine,
            age=age,
        )

        # Calculate overall modulation ratio
        modulation_ratio = modulated_duration / base_estimate if base_estimate > 0 else 1.0

        # Compute confidence (average of components)
        confidence = (
            circuit_signal.confidence + interval_result.confidence + 0.7  # Embodied baseline
        ) / 3.0

        # Adjust confidence based on modulation effects
        for effect in effects.values():
            confidence *= effect.confidence_modifier

        # Calculate errors
        error = modulated_duration - actual_duration
        relative_error = error / actual_duration if actual_duration > 0 else 0

        # Determine scale
        if actual_duration < 1.0:
            scale = TimingScale.MILLISECONDS
        elif actual_duration < 60:
            scale = TimingScale.SECONDS
        else:
            scale = TimingScale.MINUTES

        estimate = TemporalEstimate(
            perceived_duration=modulated_duration,
            actual_duration=actual_duration,
            error=error,
            relative_error=relative_error,
            confidence=confidence,
            scale=scale,
            circuit_estimate=circuit_estimate,
            interval_estimate=interval_estimate,
            embodied_estimate=embodied_estimate,
            modulation_ratio=modulation_ratio,
            emotional_state=emotional_state.value if emotional_state else None,
            attention_level=attention,
            dopamine_level=dopamine,
            age=age,
        )

        self.estimate_history.append(estimate)

        return estimate

    def produce_duration(
        self, target_duration: float, emotional_state: Optional[EmotionalState] = None
    ) -> TemporalEstimate:
        """Produce a target duration (behavioral timing).

        Args:
            target_duration: Duration to produce
            emotional_state: Current emotional state

        Returns:
            Estimate of produced duration
        """
        # Produce using interval timer
        result = self.interval_timer.produce_duration(target_duration)

        return TemporalEstimate(
            perceived_duration=result.estimated_duration,
            actual_duration=target_duration,
            error=result.error,
            relative_error=result.relative_error,
            confidence=result.confidence,
            scale=TimingScale.SECONDS,
            circuit_estimate=result.estimated_duration,
            interval_estimate=result.estimated_duration,
            embodied_estimate=result.estimated_duration,
            modulation_ratio=1.0,
            emotional_state=emotional_state.value if emotional_state else None,
        )

    def set_dopamine(self, level: float) -> None:
        """Set dopamine level across systems."""
        self.circuits.set_dopamine(level)
        self.interval_timer.set_dopamine_level(level)
        self.modulation.dopamine.set_level(level)

    def set_arousal(self, level: float) -> None:
        """Set arousal level across systems."""
        self.circuits.set_arousal(level)
        self.embodied.set_arousal(level)

    def set_body_state(self, state: BodyState) -> None:
        """Set body state for embodied timing."""
        self.embodied.set_body_state(state)

    def get_statistics(self) -> Dict:
        """Get timing statistics."""
        if not self.estimate_history:
            return {"n_estimates": 0, "mean_error": 0.0, "mean_relative_error": 0.0}

        errors = [e.error for e in self.estimate_history]
        rel_errors = [e.relative_error for e in self.estimate_history]

        return {
            "n_estimates": len(self.estimate_history),
            "mean_error": np.mean(errors),
            "std_error": np.std(errors),
            "mean_relative_error": np.mean(rel_errors),
            "mean_confidence": np.mean([e.confidence for e in self.estimate_history]),
        }

    def reset(self) -> None:
        """Reset all systems."""
        self.circuits.reset()
        self.interval_timer.reset()
        self.embodied.reset()
        self.estimate_history = []


class TimePerceptionOrchestrator:
    """Top-level orchestrator for time perception.

    Provides high-level interface for simulating time perception
    under various conditions.
    """

    def __init__(self):
        self.subjective_system = SubjectiveTimeSystem()

        # Scenario presets
        self.scenarios = {
            "baseline": {"emotional_state": None, "attention": 0.5, "dopamine": 1.0, "age": 30},
            "fear": {
                "emotional_state": EmotionalState.FEAR,
                "attention": 0.8,
                "dopamine": 1.2,
                "age": 30,
            },
            "boredom": {
                "emotional_state": EmotionalState.BOREDOM,
                "attention": 0.3,
                "dopamine": 0.9,
                "age": 30,
            },
            "flow": {
                "emotional_state": EmotionalState.FLOW,
                "attention": 0.2,  # Attention not on time
                "dopamine": 1.1,
                "age": 30,
            },
            "elderly": {"emotional_state": None, "attention": 0.5, "dopamine": 0.85, "age": 70},
            "child": {"emotional_state": None, "attention": 0.6, "dopamine": 1.1, "age": 8},
            "stimulant": {
                "emotional_state": EmotionalState.EXCITEMENT,
                "attention": 0.7,
                "dopamine": 1.5,
                "age": 30,
            },
            "parkinsons": {"emotional_state": None, "attention": 0.5, "dopamine": 0.5, "age": 65},
        }

    def estimate(
        self, duration: float, scenario: Optional[str] = None, **kwargs
    ) -> TemporalEstimate:
        """Estimate duration with optional scenario preset.

        Args:
            duration: Actual duration in seconds
            scenario: Named scenario preset
            **kwargs: Override specific parameters

        Returns:
            Temporal estimate
        """
        # Start with baseline or scenario
        if scenario and scenario in self.scenarios:
            params = self.scenarios[scenario].copy()
        else:
            params = self.scenarios["baseline"].copy()

        # Apply overrides
        params.update(kwargs)

        return self.subjective_system.estimate_duration(
            duration,
            emotional_state=params.get("emotional_state"),
            attention=params.get("attention", 0.5),
            dopamine=params.get("dopamine", 1.0),
            age=params.get("age", 30),
            movement=params.get("movement", False),
            external_rhythm=params.get("external_rhythm"),
            mode=params.get("mode", TimingMode.PROSPECTIVE),
        )

    def compare_scenarios(
        self, duration: float, scenarios: List[str]
    ) -> Dict[str, TemporalEstimate]:
        """Compare time perception across scenarios.

        Args:
            duration: Duration to estimate
            scenarios: List of scenario names

        Returns:
            Dict mapping scenario name to estimate
        """
        results = {}

        for scenario in scenarios:
            if scenario in self.scenarios:
                results[scenario] = self.estimate(duration, scenario)

        return results

    def simulate_event(
        self, duration: float, event_type: str, intensity: float = 0.5
    ) -> TemporalEstimate:
        """Simulate time perception during an event.

        Args:
            duration: Event duration
            event_type: Type of event (threatening, rewarding, boring, etc.)
            intensity: Event intensity (0-1)

        Returns:
            Temporal estimate
        """
        if event_type == "threatening":
            return self.estimate(
                duration,
                emotional_state=EmotionalState.FEAR,
                attention=0.9,
                dopamine=1.0 + intensity * 0.3,
            )
        elif event_type == "rewarding":
            return self.estimate(
                duration,
                emotional_state=EmotionalState.EXCITEMENT,
                attention=0.6,
                dopamine=1.0 + intensity * 0.5,
            )
        elif event_type == "boring":
            return self.estimate(
                duration,
                emotional_state=EmotionalState.BOREDOM,
                attention=0.2 + intensity * 0.2,
                dopamine=0.9 - intensity * 0.2,
            )
        elif event_type == "engaging":
            return self.estimate(
                duration,
                emotional_state=EmotionalState.FLOW,
                attention=0.1,  # Absorbed in task, not time
                dopamine=1.1,
            )
        else:
            return self.estimate(duration)

    def simulate_day(
        self, events: List[Tuple[float, str, float]]
    ) -> List[Tuple[str, TemporalEstimate]]:
        """Simulate time perception throughout a day.

        Args:
            events: List of (duration, event_type, intensity) tuples

        Returns:
            List of (event_type, estimate) tuples
        """
        results = []

        for duration, event_type, intensity in events:
            estimate = self.simulate_event(duration, event_type, intensity)
            results.append((event_type, estimate))

        return results

    def get_subjective_day_length(
        self, events: List[Tuple[float, str, float]]
    ) -> Tuple[float, float]:
        """Calculate subjective vs objective day length.

        Args:
            events: List of (duration, event_type, intensity) tuples

        Returns:
            Tuple of (subjective_total, objective_total)
        """
        results = self.simulate_day(events)

        objective_total = sum(duration for duration, _, _ in events)
        subjective_total = sum(est.perceived_duration for _, est in results)

        return subjective_total, objective_total

    def reset(self) -> None:
        """Reset all systems."""
        self.subjective_system.reset()
