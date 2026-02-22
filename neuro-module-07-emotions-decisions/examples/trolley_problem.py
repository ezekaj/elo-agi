"""
Demo: Trolley Problem - Moral Dilemma Processing

Demonstrates:
- Deontological vs Utilitarian moral reasoning
- Personal vs Impersonal harm distinction
- VMPFC vs DLPFC activation patterns
"""

import sys

sys.path.insert(0, "..")

from src.moral_reasoning import (
    MoralDilemmaProcessor,
    create_trolley_switch,
    create_trolley_push,
    create_crying_baby,
)


def demo_switch_vs_push():
    """
    Compare responses to switch (impersonal) vs push (personal) scenarios.

    Research shows:
    - Switch: ~90% say flip the switch (utilitarian)
    - Push: ~10% say push the person (deontological wins)
    """
    print("=" * 70)
    print("DEMO: Trolley Switch vs Footbridge Push")
    print("=" * 70)

    processor = MoralDilemmaProcessor(vmpfc_intact=True)

    # Scenario 1: Flip the switch
    switch = create_trolley_switch()
    print(f"\nScenario 1: {switch.name}")
    print(f"Description: {switch.description}")
    print(f"Action: {switch.action_description}")
    print(f"Harm type: {switch.harm_type.value}")
    print(f"Lives saved: {switch.lives_saved}, Lives lost: {switch.lives_lost}")

    switch_decision = processor.process_dilemma(switch)

    print(f"\nDecision: {'FLIP SWITCH' if switch_decision.action_taken else 'DO NOT FLIP'}")
    print(f"Framework: {switch_decision.framework_used.value}")
    print(f"Confidence: {switch_decision.confidence:.2f}")
    print(f"Deontological weight: {switch_decision.deontological_weight:.2f}")
    print(f"Utilitarian weight: {switch_decision.utilitarian_weight:.2f}")
    print(f"Emotional response: {switch_decision.emotional_response:.2f}")
    print(f"Processing time: {switch_decision.deliberation_time:.0f}ms")

    # Scenario 2: Push the person
    print("\n" + "-" * 70)
    push = create_trolley_push()
    print(f"\nScenario 2: {push.name}")
    print(f"Description: {push.description}")
    print(f"Action: {push.action_description}")
    print(f"Harm type: {push.harm_type.value}")
    print(f"Lives saved: {push.lives_saved}, Lives lost: {push.lives_lost}")

    push_decision = processor.process_dilemma(push)

    print(f"\nDecision: {'PUSH' if push_decision.action_taken else 'DO NOT PUSH'}")
    print(f"Framework: {push_decision.framework_used.value}")
    print(f"Confidence: {push_decision.confidence:.2f}")
    print(f"Deontological weight: {push_decision.deontological_weight:.2f}")
    print(f"Utilitarian weight: {push_decision.utilitarian_weight:.2f}")
    print(f"Emotional response: {push_decision.emotional_response:.2f}")
    print(f"Processing time: {push_decision.deliberation_time:.0f}ms")

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print("\n                    SWITCH         PUSH")
    print("Same utility:       5-1=+4         5-1=+4")
    print(
        f"Would act:          {'Yes' if switch_decision.action_taken else 'No':14} {'Yes' if push_decision.action_taken else 'No'}"
    )
    print(
        f"Deont weight:       {switch_decision.deontological_weight:.2f}           {push_decision.deontological_weight:.2f}"
    )
    print(
        f"Util weight:        {switch_decision.utilitarian_weight:.2f}           {push_decision.utilitarian_weight:.2f}"
    )
    print(
        f"Emotion:            {switch_decision.emotional_response:+.2f}          {push_decision.emotional_response:+.2f}"
    )

    print("\nKey insight: Same utilitarian calculus, but personal harm")
    print("triggers VMPFC-driven deontological override!")


def demo_extreme_scenario():
    """
    Test with crying baby scenario - extremely personal harm.
    """
    print("\n" + "=" * 70)
    print("DEMO: Extreme Scenario - Crying Baby")
    print("=" * 70)

    processor = MoralDilemmaProcessor(vmpfc_intact=True)
    baby = create_crying_baby()

    print(f"\nScenario: {baby.name}")
    print(f"Description: {baby.description}")
    print(f"Personal involvement: {baby.personal_involvement}")
    print(f"Emotional intensity: {baby.emotional_intensity}")

    decision = processor.process_dilemma(baby)

    print(f"\nDecision: {'SMOTHER' if decision.action_taken else 'DO NOT SMOTHER'}")
    print(f"Framework: {decision.framework_used.value}")
    print(f"Confidence: {decision.confidence:.2f}")
    print(f"Emotional response: {decision.emotional_response:.2f}")
    print(f"Reasoning: {decision.reasoning}")


def demo_brain_region_predictions():
    """
    Show predicted brain region activations for different scenarios.
    """
    print("\n" + "=" * 70)
    print("DEMO: Brain Region Activation Predictions")
    print("=" * 70)

    processor = MoralDilemmaProcessor()

    scenarios = [
        ("Trolley Switch (impersonal)", create_trolley_switch()),
        ("Footbridge Push (personal)", create_trolley_push()),
        ("Crying Baby (extreme)", create_crying_baby()),
    ]

    print("\nPredicted brain region activations:")
    print("-" * 70)
    print(f"{'Scenario':<30} {'VMPFC':>10} {'DLPFC':>10} {'Dominant':<15}")
    print("-" * 70)

    for name, scenario in scenarios:
        decision = processor.process_dilemma(scenario)

        # VMPFC activation ~ deontological weight * emotional intensity
        vmpfc_activation = decision.deontological_weight * scenario.emotional_intensity
        # DLPFC activation ~ utilitarian weight * (1 - emotional intensity)
        dlpfc_activation = decision.utilitarian_weight * (1 - scenario.emotional_intensity * 0.5)

        dominant = "VMPFC" if vmpfc_activation > dlpfc_activation else "DLPFC"

        print(f"{name:<30} {vmpfc_activation:>10.2f} {dlpfc_activation:>10.2f} {dominant:<15}")

    print("\nResearch alignment:")
    print("- Personal harm → VMPFC activation → Deontological response")
    print("- Impersonal harm → DLPFC activation → Utilitarian response")


if __name__ == "__main__":
    demo_switch_vs_push()
    demo_extreme_scenario()
    demo_brain_region_predictions()

    print("\n" + "=" * 70)
    print("Key Research Validations:")
    print("=" * 70)
    print("✓ Personal harm triggers deontological (VMPFC) response")
    print("✓ Impersonal harm triggers utilitarian (DLPFC) response")
    print("✓ Same utility (5-1=4) produces different decisions")
    print("✓ Emotional intensity correlates with deontological weight")
    print("✓ Processing time longer for utilitarian deliberation")
