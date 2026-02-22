"""
Demo: Intuition vs Deliberation

Demonstrates when System 1 (intuition) handles tasks alone vs when
System 2 (deliberation) is engaged.

Based on research showing:
- Easy, familiar tasks: System 1 only
- Complex or conflicting: System 2 engaged
- Conflict detection triggers switching
"""

import numpy as np
import sys

sys.path.insert(0, "..")

from src.dual_process_controller import DualProcessController
from src.system1.habit_executor import Action


def setup_controller():
    """Set up a controller with some learned patterns and habits"""
    controller = DualProcessController()

    # Learn some patterns
    # Pattern A: clear, familiar
    controller.learn_pattern(
        "familiar_A",
        [
            np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.9, 0.1, 0.0, 0.0, 0.0]),
        ],
    )

    # Pattern B: another clear pattern
    controller.learn_pattern(
        "familiar_B",
        [
            np.array([0.0, 0.0, 1.0, 0.0, 0.0]),
            np.array([0.0, 0.1, 0.9, 0.0, 0.0]),
        ],
    )

    # Train a habit
    stimulus = np.array([0.5, 0.5, 0.0, 0.0, 0.0])
    action = Action(id="habitual_response")
    controller.train_habit(stimulus, action, repetitions=20)

    # Learn emotional associations
    controller.learn_emotional_association(
        np.array([0.0, 0.0, 0.0, 1.0, 0.0]), threat=0.8, reward=0.1
    )

    return controller


def demo_easy_task():
    """Easy task: System 1 handles it alone"""
    print("=" * 60)
    print("DEMO 1: Easy Task (System 1 Only)")
    print("=" * 60)

    controller = setup_controller()

    # Input that clearly matches pattern A
    easy_input = np.array([0.95, 0.05, 0.0, 0.0, 0.0])

    result = controller.process(easy_input)

    print("Input: Clear match to familiar pattern")
    print(f"System used: {result.system_used}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Conflict detected: {result.conflict_detected}")
    print(f"System 2 engaged: {result.s2_output is not None}")

    if result.system_used == "system1":
        print("✓ System 1 handled this alone (fast, intuitive)")
    print()


def demo_habitual_response():
    """Habitual task: Automatic response"""
    print("=" * 60)
    print("DEMO 2: Habitual Response (Automatic)")
    print("=" * 60)

    controller = setup_controller()

    # Input that triggers learned habit
    habit_trigger = np.array([0.5, 0.5, 0.0, 0.0, 0.0])

    result = controller.process(habit_trigger)

    print("Input: Matches trained habit trigger")
    print(f"System used: {result.system_used}")
    print(f"Response type: {type(result.response).__name__}")

    if result.s1_output and result.s1_output.habit_response:
        print(f"Habit triggered: {result.s1_output.habit_response.triggered}")
        print(f"Habit strength: {result.s1_output.habit_response.habit_strength:.2f}")

    print("✓ Automatic habit execution (no deliberation)")
    print()


def demo_conflict_triggers_s2():
    """Conflicting inputs trigger System 2"""
    print("=" * 60)
    print("DEMO 3: Conflict Triggers System 2")
    print("=" * 60)

    controller = setup_controller()

    # Input that matches BOTH patterns (conflict)
    conflicting_input = np.array([0.5, 0.0, 0.5, 0.0, 0.0])

    result = controller.process(conflicting_input)

    print("Input: Partially matches multiple patterns")
    print(f"Conflict detected: {result.conflict_detected}")
    print(f"System used: {result.system_used}")
    print(f"System 2 engaged: {result.s2_output is not None}")

    if result.conflict_detected:
        print("✓ Conflict between patterns detected")
        if result.s2_output:
            print(f"✓ System 2 deliberated for {result.s2_output.deliberation_steps} steps")
    print()


def demo_threat_triggers_careful_processing():
    """High threat triggers System 2 (careful processing)"""
    print("=" * 60)
    print("DEMO 4: Threat Triggers Careful Processing")
    print("=" * 60)

    controller = setup_controller()

    # Input associated with threat
    threatening_input = np.array([0.0, 0.0, 0.0, 1.0, 0.0])

    result = controller.process(threatening_input)

    print("Input: Associated with learned threat")

    if result.s1_output:
        print(f"Threat level: {result.s1_output.emotional_valence.threat:.2f}")
        print(f"Arousal: {result.s1_output.emotional_valence.arousal:.2f}")

    print(f"System 2 engaged: {result.s2_output is not None}")
    print("✓ High threat triggers more careful, deliberate processing")
    print()


def demo_force_deliberation():
    """Force System 2 to override System 1"""
    print("=" * 60)
    print("DEMO 5: Forced Deliberation (Override Intuition)")
    print("=" * 60)

    controller = setup_controller()

    # Easy input that S1 would normally handle
    easy_input = np.array([0.95, 0.05, 0.0, 0.0, 0.0])

    # Force System 2
    result = controller.override_intuition(easy_input)

    print("Input: Would normally be handled by System 1")
    print("Forced System 2: Yes")
    print(f"System used: {result.system_used}")
    print(f"Override occurred: {result.override_occurred}")

    if result.s2_output:
        print(f"Deliberation steps: {result.s2_output.deliberation_steps}")

    print("✓ Can consciously engage deliberate processing")
    print()


def demo_trust_intuition():
    """Trust intuition despite conflict"""
    print("=" * 60)
    print("DEMO 6: Trust Intuition Despite Conflict")
    print("=" * 60)

    controller = setup_controller()

    # Conflicting input
    conflicting_input = np.array([0.5, 0.0, 0.5, 0.0, 0.0])

    # Trust gut feeling
    result = controller.trust_gut(conflicting_input)

    print("Input: Conflicting (would normally trigger S2)")
    print("Trust intuition: Yes")
    print(f"System used: {result.system_used}")
    print(f"Conflict was present: {result.conflict_detected}")
    print(f"System 2 used anyway: {result.s2_output is not None}")

    print("✓ Can trust intuition even when System 2 would normally engage")
    print()


def demo_processing_statistics():
    """Show processing statistics over multiple trials"""
    print("=" * 60)
    print("DEMO 7: Processing Statistics")
    print("=" * 60)

    controller = setup_controller()

    # Process many inputs
    inputs = [
        np.array([0.95, 0.05, 0.0, 0.0, 0.0]),  # Easy
        np.array([0.05, 0.05, 0.9, 0.0, 0.0]),  # Easy
        np.array([0.5, 0.0, 0.5, 0.0, 0.0]),  # Conflict
        np.array([0.5, 0.5, 0.0, 0.0, 0.0]),  # Habit
        np.array([0.0, 0.0, 0.0, 0.9, 0.0]),  # Threat
        np.array([0.9, 0.0, 0.1, 0.0, 0.0]),  # Easy
        np.array([0.4, 0.0, 0.4, 0.2, 0.0]),  # Conflict
    ]

    for inp in inputs:
        controller.process(inp)

    stats = controller.get_processing_stats()

    print(f"Total processed: {stats['total_processed']}")
    print(f"System 2 engagement rate: {stats['s2_engagement_rate']:.1%}")
    print(f"Conflict rate: {stats['conflict_rate']:.1%}")
    print(f"Override rate: {stats['override_rate']:.1%}")
    print(f"Avg System 1 time: {stats['avg_s1_time'] * 1000:.2f}ms")
    if stats["avg_s2_time"] > 0:
        print(f"Avg System 2 time: {stats['avg_s2_time'] * 1000:.2f}ms")

    print()
    print("✓ System 1 is fast, System 2 is slower but more careful")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DUAL PROCESS DEMONSTRATION")
    print("System 1 (Intuition) vs System 2 (Deliberation)")
    print("=" * 60 + "\n")

    demo_easy_task()
    demo_habitual_response()
    demo_conflict_triggers_s2()
    demo_threat_triggers_careful_processing()
    demo_force_deliberation()
    demo_trust_intuition()
    demo_processing_statistics()

    print("\n" + "=" * 60)
    print("KEY INSIGHTS:")
    print("- System 1 handles familiar, unambiguous situations FAST")
    print("- System 2 engages when there's conflict or uncertainty")
    print("- Threats trigger more careful processing")
    print("- We can consciously override intuition (cognitive effort)")
    print("- We can also trust our gut when appropriate")
    print("=" * 60)
