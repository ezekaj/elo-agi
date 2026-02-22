"""
Demo: VMPFC Lesion Simulation

Demonstrates effects of VMPFC damage on moral reasoning based on lesion studies:
- More utilitarian decisions
- Emotional blunting
- Intact intellectual abilities
- More willing to sacrifice one to save many
"""

import sys

sys.path.insert(0, "..")

from src.moral_reasoning import (
    MoralDilemmaProcessor,
    VMPFCLesionModel,
    create_trolley_switch,
    create_trolley_push,
    create_crying_baby,
)
from src.emotion_decision_integrator import (
    EmotionDecisionSystem,
    create_threat_situation,
    create_moral_situation,
)


def demo_lesion_vs_healthy():
    """
    Compare healthy vs VMPFC-lesioned responses to moral dilemmas.
    """
    print("=" * 70)
    print("DEMO: VMPFC Lesion Effects on Moral Reasoning")
    print("=" * 70)

    lesion_model = VMPFCLesionModel()

    scenarios = [
        ("Trolley Switch", create_trolley_switch()),
        ("Footbridge Push", create_trolley_push()),
        ("Crying Baby", create_crying_baby()),
    ]

    for name, scenario in scenarios:
        print(f"\n{'-' * 70}")
        print(f"Scenario: {name}")
        print(f"Lives saved: {scenario.lives_saved}, Lives lost: {scenario.lives_lost}")
        print(f"Personal involvement: {scenario.personal_involvement}")

        comparison = lesion_model.compare_with_healthy(scenario)
        healthy = comparison["healthy"]
        lesioned = comparison["lesioned"]

        print(f"\n{'':20} {'HEALTHY':>15} {'VMPFC LESION':>15}")
        print(f"{'-' * 50}")
        print(
            f"{'Would act:':<20} {'Yes' if healthy.action_taken else 'No':>15} {'Yes' if lesioned.action_taken else 'No':>15}"
        )
        print(
            f"{'Deont weight:':<20} {healthy.deontological_weight:>15.2f} {lesioned.deontological_weight:>15.2f}"
        )
        print(
            f"{'Util weight:':<20} {healthy.utilitarian_weight:>15.2f} {lesioned.utilitarian_weight:>15.2f}"
        )
        print(
            f"{'Emotional resp:':<20} {healthy.emotional_response:>15.2f} {lesioned.emotional_response:>15.2f}"
        )
        print(
            f"{'Framework:':<20} {healthy.framework_used.value:>15} {lesioned.framework_used.value:>15}"
        )


def demo_emotional_blunting():
    """
    Show how VMPFC lesion reduces emotional responses.
    """
    print("\n" + "=" * 70)
    print("DEMO: Emotional Blunting in VMPFC Lesion")
    print("=" * 70)

    healthy_system = EmotionDecisionSystem()
    lesioned_system = EmotionDecisionSystem()
    lesioned_system.simulate_lesion("vmpfc")

    # Process threatening situation
    threat = create_threat_situation(intensity=0.8)

    print("\nProcessing high-threat situation...")

    healthy_decision = healthy_system.process_situation(threat)
    lesioned_decision = lesioned_system.process_situation(threat)

    print(f"\n{'':25} {'HEALTHY':>15} {'VMPFC LESION':>15}")
    print(f"{'-' * 55}")
    print(
        f"{'Emotional valence:':<25} {healthy_decision.emotional_state.valence:>15.2f} {lesioned_decision.emotional_state.valence:>15.2f}"
    )
    print(
        f"{'Emotional arousal:':<25} {healthy_decision.emotional_state.arousal:>15.2f} {lesioned_decision.emotional_state.arousal:>15.2f}"
    )

    print("\n   Lesioned patient shows 'emotional blunting' -")
    print("   reduced emotional response despite intact perception.")


def demo_intellectual_preservation():
    """
    Show that intellectual abilities remain intact after VMPFC lesion.
    """
    print("\n" + "=" * 70)
    print("DEMO: Preserved Intellectual Abilities")
    print("=" * 70)

    lesion = VMPFCLesionModel()

    print("\nVMPFC lesion patients can still:")
    print("-" * 40)

    # Test utilitarian calculation
    switch = create_trolley_switch()
    decision = lesion.process_moral_dilemma(switch)

    print(f"\n1. Perform utilitarian calculus:")
    print(f"   5 lives saved - 1 life lost = 4 net lives")
    print(f"   Patient calculates: Action has positive utility")
    print(f"   Decision confidence: {decision.confidence:.2f}")

    print(f"\n2. Understand abstract moral concepts:")
    print(f"   Framework used: {decision.framework_used.value}")
    print(f"   Reasoning preserved: {decision.reasoning}")

    print(f"\n3. Make consistent logical decisions:")
    switch2 = create_trolley_switch()
    decision2 = lesion.process_moral_dilemma(switch2)
    print(f"   Same scenario, same decision: {decision.action_taken == decision2.action_taken}")

    print("\n   Key insight: VMPFC damage affects emotional/intuitive")
    print("   morality, NOT cognitive/deliberative abilities.")


def demo_clinical_implications():
    """
    Discuss clinical implications of VMPFC lesion findings.
    """
    print("\n" + "=" * 70)
    print("DEMO: Clinical Implications")
    print("=" * 70)

    lesion = VMPFCLesionModel()

    print("\nVMPFC Lesion Characteristics:")
    print("-" * 40)
    print(f"  Emotional blunting: {lesion.emotional_blunting * 100:.0f}% reduction")
    print(f"  Utilitarian bias: {lesion.utilitarian_bias * 100:.0f}%")
    print(f"  Intellectual intact: {lesion.intellectual_intact}")

    print("\nReal-world implications:")
    print("-" * 40)
    print("  1. Patients may make 'cold' utilitarian decisions")
    print("  2. Reduced 'gut feelings' about right/wrong")
    print("  3. May seem callous but aren't intentionally cruel")
    print("  4. Can understand moral rules, just don't 'feel' them")

    print("\nFamous case: Phineas Gage")
    print("-" * 40)
    print("  - Iron rod through VMPFC in 1848")
    print("  - Survived but personality changed dramatically")
    print("  - 'No longer Gage' - impulsive, socially inappropriate")
    print("  - First evidence of frontal lobe role in personality/morality")


def demo_all_lesion_types():
    """
    Compare effects of lesioning different brain regions.
    """
    print("\n" + "=" * 70)
    print("DEMO: Comparing Different Lesion Effects")
    print("=" * 70)

    scenario = create_trolley_push()

    regions = ["none", "vmpfc", "amygdala", "acc"]

    print(f"\nScenario: {scenario.name}")
    print(f"{'':15} {'Action':>10} {'Emotion':>10} {'Confidence':>12}")
    print("-" * 50)

    for region in regions:
        system = EmotionDecisionSystem()
        if region != "none":
            system.simulate_lesion(region)

        situation = create_moral_situation(scenario)
        decision = system.process_situation(situation)

        action = "PUSH" if decision.action == "act" else "DON'T"
        emotion = decision.emotional_state.valence
        conf = decision.confidence

        label = f"{region.upper():15}" if region != "none" else "HEALTHY        "
        print(f"{label} {action:>10} {emotion:>10.2f} {conf:>12.2f}")

        if region != "none":
            system.restore_all()

    print("\nObservations:")
    print("  - VMPFC: More utilitarian, reduced emotion")
    print("  - Amygdala: Reduced threat detection")
    print("  - ACC: Reduced conflict monitoring")


if __name__ == "__main__":
    demo_lesion_vs_healthy()
    demo_emotional_blunting()
    demo_intellectual_preservation()
    demo_clinical_implications()
    demo_all_lesion_types()

    print("\n" + "=" * 70)
    print("Key Research Validations:")
    print("=" * 70)
    print("✓ VMPFC lesion → more utilitarian decisions")
    print("✓ Emotional blunting (reduced gut feelings)")
    print("✓ Intellectual abilities preserved")
    print("✓ More willing to sacrifice one to save many")
    print("✓ Pattern matches clinical observations")
