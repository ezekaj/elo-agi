"""
Demo: Embodied Time Perception

Demonstrates the 2025 research insight:
Time is not perceived abstractly - it's grounded in embodied experience.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embodied_time import (
    InteroceptiveTimer,
    MotorTimer,
    BodyEnvironmentCoupler,
    EmbodiedTimeSystem,
    BodyState
)
from src.time_circuits import TimeCircuit


def demo_heartbeat_timing():
    """Demonstrate timing through heartbeat counting."""
    print("=" * 60)
    print("Demo 1: Heartbeat-Based Time Perception")
    print("=" * 60)

    print("\nPeople with high interoceptive awareness can estimate time")
    print("by counting their heartbeats.\n")

    # Compare good vs poor interoceptive awareness
    good_awareness = InteroceptiveTimer(interoceptive_accuracy=0.9)
    poor_awareness = InteroceptiveTimer(interoceptive_accuracy=0.4)

    durations = [5, 10, 30, 60]

    print(f"{'Duration (s)':<15} {'Good Awareness':<20} {'Poor Awareness':<20}")
    print("-" * 55)

    for duration in durations:
        good_est = good_awareness.estimate_duration_from_heartbeats(duration)
        good_awareness.reset()
        poor_est = poor_awareness.estimate_duration_from_heartbeats(duration)
        poor_awareness.reset()

        good_error = abs(good_est - duration) / duration * 100
        poor_error = abs(poor_est - duration) / duration * 100

        print(f"{duration:<15} {good_est:.1f}s ({good_error:.0f}% err)"
              f"{'':5} {poor_est:.1f}s ({poor_error:.0f}% err)")


def demo_arousal_body_timing():
    """Demonstrate how arousal affects body-based timing."""
    print("\n" + "=" * 60)
    print("Demo 2: Arousal Changes Body Rhythms")
    print("=" * 60)

    print("\nHigh arousal increases heart rate, affecting time perception.\n")

    timer = InteroceptiveTimer()
    duration = 30.0

    arousal_levels = [0.1, 0.3, 0.5, 0.7, 0.9]

    print(f"{'Arousal':<12} {'Heart Rate':<15} {'Perceived (s)':<18} {'Actual: 30s'}")
    print("-" * 60)

    for arousal in arousal_levels:
        timer.set_arousal(arousal)
        hr = timer.body_state.heart_rate
        estimate = timer.estimate_duration_from_heartbeats(duration)
        timer.reset()

        diff = estimate - duration
        sign = "+" if diff > 0 else ""

        print(f"{arousal:<12.1f} {hr:<15.0f} BPM"
              f"{estimate:<10.1f}s ({sign}{diff:.1f}s)")


def demo_motor_timing():
    """Demonstrate timing through motor actions."""
    print("\n" + "=" * 60)
    print("Demo 3: Motor Timing (SMA)")
    print("=" * 60)

    print("\nThe brain tracks time through the rhythm of actions.\n")

    timer = MotorTimer(motor_precision=0.9)

    # Tap a rhythm
    target_interval = 0.5  # 500ms
    n_taps = 20

    intervals = timer.tap_rhythm(target_interval, n_taps)

    mean_interval = np.mean(intervals)
    std_interval = np.std(intervals)
    cv = std_interval / mean_interval * 100

    print(f"Target interval: {target_interval * 1000:.0f} ms")
    print(f"Actual mean:     {mean_interval * 1000:.0f} ms")
    print(f"Std deviation:   {std_interval * 1000:.1f} ms")
    print(f"Coefficient of variation: {cv:.1f}%")

    # Synchronization test
    print("\nSynchronization to external beat:")
    taps, asynchrony = timer.synchronize_to_beat(0.5, 20)
    print(f"Mean asynchrony: {asynchrony * 1000:.1f} ms")


def demo_body_environment_coupling():
    """Demonstrate body-environment entrainment."""
    print("\n" + "=" * 60)
    print("Demo 4: Body-Environment Synchronization")
    print("=" * 60)

    print("\nTime perception improves when body rhythm syncs with environment.\n")

    coupler = BodyEnvironmentCoupler()

    # Test different rhythm relationships
    body_rhythm = 1.0  # 1 Hz (e.g., heartbeat at 60 BPM)

    env_rhythms = [
        (0.5, "Half speed (1:2)"),
        (1.0, "Same speed (1:1)"),
        (1.3, "Non-harmonic"),
        (2.0, "Double speed (2:1)"),
        (3.0, "Triple speed (3:1)")
    ]

    print(f"Body rhythm: {body_rhythm} Hz\n")
    print(f"{'Env Rhythm':<15} {'Relationship':<20} {'Entrainment':<15} {'Timing Quality'}")
    print("-" * 70)

    for env_rhythm, name in env_rhythms:
        coupler.set_body_rhythm(body_rhythm)
        coupler.set_environmental_rhythm(env_rhythm)

        entrainment = coupler.compute_entrainment()

        if entrainment > 0.7:
            quality = "Excellent"
        elif entrainment > 0.4:
            quality = "Good"
        else:
            quality = "Poor"

        print(f"{env_rhythm:<15.1f} Hz {name:<20} {entrainment:<15.2f} {quality}")


def demo_embodied_integration():
    """Demonstrate integrated embodied time perception."""
    print("\n" + "=" * 60)
    print("Demo 5: Integrated Embodied Time System")
    print("=" * 60)

    system = EmbodiedTimeSystem()
    duration = 30.0

    print(f"\nActual duration: {duration} seconds\n")

    # Different conditions
    conditions = [
        ("At rest", False, None, BodyState()),
        ("While moving", True, None, BodyState(movement_level=0.7)),
        ("With music (1 Hz)", False, 1.0, BodyState()),
        ("Moving + music", True, 1.0, BodyState(movement_level=0.7)),
        ("Aroused (exercise)", False, None, BodyState(heart_rate=120, breathing_rate=25)),
    ]

    print(f"{'Condition':<25} {'Perceived (s)':<18} {'Components Used'}")
    print("-" * 65)

    for name, movement, ext_rhythm, body_state in conditions:
        system.set_body_state(body_state)

        estimate, components = system.estimate_duration(
            duration,
            movement_present=movement,
            external_rhythm=ext_rhythm
        )

        comp_names = list(components.keys())
        comp_str = ", ".join([c for c in comp_names if c != 'entrainment'])

        print(f"{name:<25} {estimate:<18.1f} {comp_str}")

        system.reset()


def demo_brain_circuits():
    """Demonstrate neural circuits for time perception."""
    print("\n" + "=" * 60)
    print("Demo 6: Neural Circuits for Time Perception")
    print("=" * 60)

    print("\n| Structure     | Function                                 |")
    print("|---------------|------------------------------------------|")
    print("| Insula        | Interoceptive signals (heartbeat, etc.)  |")
    print("| SMA           | Motor timing, body-environment coupling  |")
    print("| Basal ganglia | Interval timing, temporal working memory |")
    print("| Cerebellum    | Precise sub-second timing                |")

    circuit = TimeCircuit()

    print("\n\nDuration estimates from each circuit:")
    print("-" * 50)

    duration = 10.0

    # Get individual circuit estimates
    insula_sig = circuit.insula.process_interoceptive_signal(duration, 0.5)
    print(f"Insula (interoceptive): {insula_sig.duration_estimate:.2f}s")
    circuit.insula.reset()

    sma_sig = circuit.sma.time_motor_sequence(duration, 0.5)
    print(f"SMA (motor):            {sma_sig.duration_estimate:.2f}s")
    circuit.sma.reset()

    bg_sig = circuit.basal_ganglia.time_interval(duration, 0.5)
    print(f"Basal ganglia:          {bg_sig.duration_estimate:.2f}s")
    circuit.basal_ganglia.reset()

    cb_sig = circuit.cerebellum.time_precise_interval(duration)
    print(f"Cerebellum:             {cb_sig.duration_estimate:.2f}s")

    print(f"\nActual duration:        {duration:.2f}s")


def create_visualization():
    """Create visualization of embodied timing."""
    print("\n" + "=" * 60)
    print("Creating Visualization")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Interoceptive accuracy effect
    accuracies = np.linspace(0.3, 1.0, 8)
    errors = []

    for acc in accuracies:
        timer = InteroceptiveTimer(interoceptive_accuracy=acc)
        trial_errors = []
        for _ in range(20):
            est = timer.estimate_duration_from_heartbeats(30.0)
            trial_errors.append(abs(est - 30.0) / 30.0 * 100)
            timer.reset()
        errors.append(np.mean(trial_errors))

    axes[0, 0].plot(accuracies, errors, 'b-o')
    axes[0, 0].set_xlabel('Interoceptive Accuracy')
    axes[0, 0].set_ylabel('Timing Error (%)')
    axes[0, 0].set_title('Better Body Awareness = Better Timing')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Heart rate and timing
    heart_rates = [50, 60, 70, 80, 90, 100, 110, 120]
    estimates = []

    for hr in heart_rates:
        timer = InteroceptiveTimer()
        timer.body_state.heart_rate = hr
        est = timer.estimate_duration_from_heartbeats(30.0)
        estimates.append(est)

    axes[0, 1].plot(heart_rates, estimates, 'r-o')
    axes[0, 1].axhline(y=30.0, color='black', linestyle='--', label='Actual')
    axes[0, 1].set_xlabel('Heart Rate (BPM)')
    axes[0, 1].set_ylabel('Perceived Duration (s)')
    axes[0, 1].set_title('Heart Rate Affects Time Perception')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Motor precision and rhythm
    motor = MotorTimer(motor_precision=0.9)
    target = 0.5
    intervals = motor.tap_rhythm(target, 50)

    axes[1, 0].hist(np.array(intervals) * 1000, bins=20, color='green', alpha=0.7)
    axes[1, 0].axvline(x=target * 1000, color='red', linestyle='--', label='Target')
    axes[1, 0].set_xlabel('Inter-tap Interval (ms)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Motor Timing Variability')
    axes[1, 0].legend()

    # 4. Entrainment effect
    coupler = BodyEnvironmentCoupler()
    coupler.set_body_rhythm(1.0)

    env_rhythms = np.linspace(0.3, 3.0, 50)
    entrainments = []

    for env in env_rhythms:
        coupler.set_environmental_rhythm(env)
        entrainments.append(coupler.compute_entrainment())

    axes[1, 1].plot(env_rhythms, entrainments, 'purple')
    axes[1, 1].set_xlabel('Environmental Rhythm (Hz)')
    axes[1, 1].set_ylabel('Entrainment Factor')
    axes[1, 1].set_title('Body-Environment Synchronization')
    axes[1, 1].axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='1:1')
    axes[1, 1].axvline(x=2.0, color='blue', linestyle='--', alpha=0.5, label='2:1')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('embodied_time.png', dpi=150)
    print("Figure saved: embodied_time.png")


def main():
    """Run all embodied time demos."""
    print("\n" + "=" * 60)
    print("EMBODIED TIME PERCEPTION DEMOS")
    print("Time Grounded in Body Experience (2025 Research)")
    print("=" * 60)

    demo_heartbeat_timing()
    demo_arousal_body_timing()
    demo_motor_timing()
    demo_body_environment_coupling()
    demo_embodied_integration()
    demo_brain_circuits()
    create_visualization()

    print("\n" + "=" * 60)
    print("Key Insight: Time is not perceived abstractly.")
    print("It's grounded in embodied experience.")
    print("=" * 60)


if __name__ == '__main__':
    main()
