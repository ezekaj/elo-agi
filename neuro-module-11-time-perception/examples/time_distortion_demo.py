"""
Demo: Time Distortion Under Different Conditions

Demonstrates how emotions, attention, dopamine, and age
affect subjective time perception.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.temporal_integration import TimePerceptionOrchestrator
from src.time_modulation import EmotionalState


def demo_emotional_time():
    """Demonstrate emotional effects on time perception."""
    print("=" * 60)
    print("Demo 1: Emotional Modulation of Time")
    print("=" * 60)

    orchestrator = TimePerceptionOrchestrator()
    duration = 10.0  # 10 seconds

    states = [
        EmotionalState.NEUTRAL,
        EmotionalState.FEAR,
        EmotionalState.EXCITEMENT,
        EmotionalState.BOREDOM,
        EmotionalState.FLOW,
        EmotionalState.ANXIETY,
    ]

    print(f"\nActual duration: {duration:.1f} seconds\n")
    print(f"{'Emotional State':<15} {'Perceived (s)':<15} {'Ratio':<10} {'Effect'}")
    print("-" * 55)

    results = {}
    for state in states:
        estimate = orchestrator.estimate(duration, emotional_state=state)
        ratio = estimate.perceived_duration / duration

        if ratio > 1.1:
            effect = "Time slows"
        elif ratio < 0.9:
            effect = "Time flies"
        else:
            effect = "Normal"

        results[state.value] = estimate.perceived_duration

        print(f"{state.value:<15} {estimate.perceived_duration:<15.2f} {ratio:<10.2f} {effect}")

    return results


def demo_attention_effect():
    """Demonstrate attentional effects on time perception."""
    print("\n" + "=" * 60)
    print("Demo 2: Attention Modulates Time (Watched Pot Effect)")
    print("=" * 60)

    orchestrator = TimePerceptionOrchestrator()
    duration = 30.0  # 30 seconds

    attention_levels = [0.1, 0.3, 0.5, 0.7, 0.9]

    print(f"\nActual duration: {duration:.1f} seconds")
    print("(High attention to time = watching the clock)\n")
    print(f"{'Attention Level':<18} {'Perceived (s)':<15} {'Difference'}")
    print("-" * 50)

    results = []
    for attention in attention_levels:
        estimate = orchestrator.estimate(duration, attention=attention)
        diff = estimate.perceived_duration - duration
        results.append((attention, estimate.perceived_duration))

        sign = "+" if diff > 0 else ""
        print(f"{attention:<18.1f} {estimate.perceived_duration:<15.2f} {sign}{diff:.2f}s")

    return results


def demo_dopamine_effect():
    """Demonstrate dopamine effects on time perception."""
    print("\n" + "=" * 60)
    print("Demo 3: Dopamine and the Internal Clock")
    print("=" * 60)

    orchestrator = TimePerceptionOrchestrator()
    duration = 20.0

    conditions = [
        ("Parkinson's (low DA)", 0.5),
        ("Depressed (low DA)", 0.7),
        ("Normal", 1.0),
        ("Rewarded", 1.3),
        ("Stimulant (high DA)", 1.8),
    ]

    print(f"\nActual duration: {duration:.1f} seconds")
    print("Higher dopamine = faster internal clock = time feels faster\n")
    print(f"{'Condition':<25} {'DA Level':<12} {'Perceived (s)':<15}")
    print("-" * 55)

    for name, da_level in conditions:
        estimate = orchestrator.estimate(duration, dopamine=da_level)
        print(f"{name:<25} {da_level:<12.1f} {estimate.perceived_duration:<15.2f}")


def demo_age_effect():
    """Demonstrate age effects on time perception."""
    print("\n" + "=" * 60)
    print("Demo 4: Why Time Flies as We Age")
    print("=" * 60)

    orchestrator = TimePerceptionOrchestrator()
    duration = 60.0  # 1 minute

    ages = [8, 15, 25, 40, 60, 80]

    print(f"\nActual duration: {duration:.1f} seconds")
    print("Proportional theory: Each year is smaller fraction of life\n")
    print(f"{'Age':<10} {'Perceived (s)':<18} {'Relative to Age 20'}")
    print("-" * 50)

    reference = None
    results = []

    for age in ages:
        estimate = orchestrator.estimate(duration, age=age)
        results.append((age, estimate.perceived_duration))

        if age == 25:
            reference = estimate.perceived_duration

        if reference:
            relative = estimate.perceived_duration / reference
            print(f"{age:<10} {estimate.perceived_duration:<18.2f} {relative:.2%}")
        else:
            print(f"{age:<10} {estimate.perceived_duration:<18.2f} -")

    return results


def demo_scenarios():
    """Compare different life scenarios."""
    print("\n" + "=" * 60)
    print("Demo 5: Time Perception in Different Scenarios")
    print("=" * 60)

    orchestrator = TimePerceptionOrchestrator()
    duration = 60.0  # 1 minute

    scenarios = ["baseline", "fear", "boredom", "flow", "elderly", "child", "stimulant"]

    print(f"\nActual duration: {duration:.1f} seconds\n")
    print(f"{'Scenario':<15} {'Perceived (s)':<18} {'Description'}")
    print("-" * 65)

    descriptions = {
        "baseline": "Normal adult, neutral state",
        "fear": "Threatening situation (time slows)",
        "boredom": "Waiting, nothing to do (time drags)",
        "flow": "Absorbed in engaging task (time flies)",
        "elderly": "70 year old (time accelerates with age)",
        "child": "8 year old (time feels slower)",
        "stimulant": "Under stimulant effects (faster clock)",
    }

    for scenario in scenarios:
        estimate = orchestrator.estimate(duration, scenario=scenario)
        desc = descriptions.get(scenario, "")
        print(f"{scenario:<15} {estimate.perceived_duration:<18.2f} {desc}")


def demo_day_simulation():
    """Simulate time perception throughout a day."""
    print("\n" + "=" * 60)
    print("Demo 6: A Day in Subjective Time")
    print("=" * 60)

    orchestrator = TimePerceptionOrchestrator()

    # A typical day's events (duration in seconds, type, intensity)
    day_events = [
        (3600, "boring", 0.6),  # Boring meeting (1 hour)
        (1800, "engaging", 0.8),  # Fun project work (30 min)
        (900, "threatening", 0.7),  # Stressful deadline (15 min)
        (7200, "engaging", 0.7),  # Deep work (2 hours)
        (1800, "boring", 0.5),  # Admin tasks (30 min)
    ]

    print("\nSimulating a work day...")
    print(f"\n{'Event':<25} {'Actual (min)':<15} {'Perceived (min)':<18}")
    print("-" * 60)

    event_names = [
        "Boring meeting",
        "Fun project",
        "Stressful deadline",
        "Deep work",
        "Admin tasks",
    ]

    total_actual = 0
    total_perceived = 0

    for i, (duration, event_type, intensity) in enumerate(day_events):
        estimate = orchestrator.simulate_event(duration, event_type, intensity)

        actual_min = duration / 60
        perceived_min = estimate.perceived_duration / 60

        total_actual += actual_min
        total_perceived += perceived_min

        print(f"{event_names[i]:<25} {actual_min:<15.0f} {perceived_min:<18.1f}")

    print("-" * 60)
    print(f"{'TOTAL':<25} {total_actual:<15.0f} {total_perceived:<18.1f}")

    ratio = total_perceived / total_actual
    print(f"\nSubjective time ratio: {ratio:.2%}")
    if ratio > 1:
        print("Overall, the day felt longer than it was (too much boring stuff)")
    else:
        print("Overall, the day flew by (good mix of engaging work)")


def create_visualization():
    """Create visualization of time distortion effects."""
    print("\n" + "=" * 60)
    print("Creating Visualization")
    print("=" * 60)

    orchestrator = TimePerceptionOrchestrator()
    duration = 10.0

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Emotional effects
    emotions = ["neutral", "fear", "excitement", "boredom", "flow"]
    emotion_durations = []
    for e in emotions:
        state = EmotionalState[e.upper()] if e != "excitement" else EmotionalState.EXCITEMENT
        est = orchestrator.estimate(duration, emotional_state=state)
        emotion_durations.append(est.perceived_duration)

    axes[0, 0].barh(emotions, emotion_durations, color="steelblue")
    axes[0, 0].axvline(x=duration, color="red", linestyle="--", label="Actual")
    axes[0, 0].set_xlabel("Perceived Duration (s)")
    axes[0, 0].set_title("Emotional Effects on Time")
    axes[0, 0].legend()

    # 2. Attention effects
    attention_levels = np.linspace(0.1, 1.0, 10)
    attention_durations = []
    for att in attention_levels:
        est = orchestrator.estimate(duration, attention=att)
        attention_durations.append(est.perceived_duration)

    axes[0, 1].plot(attention_levels, attention_durations, "b-o")
    axes[0, 1].axhline(y=duration, color="red", linestyle="--", label="Actual")
    axes[0, 1].set_xlabel("Attention to Time")
    axes[0, 1].set_ylabel("Perceived Duration (s)")
    axes[0, 1].set_title("Attention Effect (Watched Pot)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Dopamine effects
    dopamine_levels = np.linspace(0.5, 2.0, 10)
    dopamine_durations = []
    for da in dopamine_levels:
        est = orchestrator.estimate(duration, dopamine=da)
        dopamine_durations.append(est.perceived_duration)

    axes[1, 0].plot(dopamine_levels, dopamine_durations, "g-o")
    axes[1, 0].axhline(y=duration, color="red", linestyle="--", label="Actual")
    axes[1, 0].set_xlabel("Dopamine Level (relative to baseline)")
    axes[1, 0].set_ylabel("Perceived Duration (s)")
    axes[1, 0].set_title("Dopamine Effect on Clock Speed")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Age effects
    ages = np.arange(5, 85, 5)
    age_durations = []
    for age in ages:
        est = orchestrator.estimate(duration, age=int(age))
        age_durations.append(est.perceived_duration)

    axes[1, 1].plot(ages, age_durations, "purple", marker="o")
    axes[1, 1].axhline(y=duration, color="red", linestyle="--", label="Actual")
    axes[1, 1].set_xlabel("Age (years)")
    axes[1, 1].set_ylabel("Perceived Duration (s)")
    axes[1, 1].set_title("Age Effect (Time Acceleration)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("time_distortion_effects.png", dpi=150)
    print("Figure saved: time_distortion_effects.png")


def main():
    """Run all time distortion demos."""
    print("\n" + "=" * 60)
    print("TIME DISTORTION DEMOS")
    print("How Emotions, Attention, Dopamine, and Age Affect Time")
    print("=" * 60)

    demo_emotional_time()
    demo_attention_effect()
    demo_dopamine_effect()
    demo_age_effect()
    demo_scenarios()
    demo_day_simulation()
    create_visualization()

    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
