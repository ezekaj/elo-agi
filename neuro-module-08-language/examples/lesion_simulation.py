"""
Lesion Simulation Demo

Demonstrates the 2025 finding that Broca's damage alone
doesn't cause language deficits - language is distributed.

Key insight: The classical model (Broca = production, Wernicke = comprehension)
is oversimplified. Language function requires distributed network integrity.
"""

import numpy as np
import sys

sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from src.language_network import DistributedLanguageNetwork
from src.predictive_language import PredictiveLanguageProcessor


def demo_distributed_network():
    """Show distributed language network architecture"""
    print("=" * 60)
    print("DISTRIBUTED LANGUAGE NETWORK")
    print("Language is not localized to single regions")
    print("=" * 60)

    network = DistributedLanguageNetwork(dim=32)

    print("\n--- Network Components ---")
    print(f"  Broca's area: dim={network.broca.dim}, active={network.broca.is_active}")
    print(f"  Wernicke's area: dim={network.wernicke.dim}, active={network.wernicke.is_active}")
    print(f"  Arcuate fasciculus: dim={network.arcuate.dim}, active={network.arcuate.is_active}")

    print("\n--- Processing Test Input ---")
    test_input = np.random.randn(32)
    result = network.process(test_input)

    print(f"  Broca output norm: {np.linalg.norm(result['broca']):.3f}")
    print(f"  Wernicke output norm: {np.linalg.norm(result['wernicke']):.3f}")
    print(f"  Combined output norm: {np.linalg.norm(result['combined']):.3f}")
    print(f"  Broca inhibition: {result['broca_inhibition']:.3f}")


def demo_broca_lesion():
    """Demonstrate Broca's lesion effects"""
    print("\n" + "=" * 60)
    print("BROCA'S AREA LESION")
    print("2025 finding: Broca's damage alone doesn't cause deficits")
    print("=" * 60)

    network = DistributedLanguageNetwork(dim=32)

    print("\n--- Baseline (Intact) ---")
    test_input = np.random.randn(32)
    baseline = network.process(test_input)
    print(f"  Network functional: {network.is_functional()}")
    print(f"  Combined output norm: {np.linalg.norm(baseline['combined']):.3f}")

    print("\n--- After Broca's Lesion (100% damage) ---")
    network.lesion("broca", 1.0)
    print(f"  Broca active: {network.broca.is_active}")
    print(f"  Broca damage level: {network.broca.damage_level}")

    lesioned = network.process(test_input)
    print(f"\n  Network functional: {network.is_functional()}")
    print(f"  Combined output norm: {np.linalg.norm(lesioned['combined']):.3f}")

    print("\n  KEY FINDING: Network is still functional!")
    print("  Wernicke's can compensate for Broca's damage.")


def demo_progressive_lesion():
    """Show progressive lesion effects"""
    print("\n" + "=" * 60)
    print("PROGRESSIVE LESION STUDY")
    print("How damage level affects function")
    print("=" * 60)

    test_input = np.random.randn(32)
    damage_levels = [0.0, 0.25, 0.5, 0.75, 1.0]

    print("\n--- Broca's Progressive Damage ---")
    print(f"{'Damage':>10} {'Functional':>12} {'Output Norm':>12}")
    print("-" * 36)

    for damage in damage_levels:
        network = DistributedLanguageNetwork(dim=32)
        network.lesion("broca", damage)
        result = network.process(test_input)
        output_norm = np.linalg.norm(result["combined"])
        print(f"{damage:>10.0%} {str(network.is_functional()):>12} {output_norm:>12.3f}")

    print("\n--- Wernicke's Progressive Damage ---")
    print(f"{'Damage':>10} {'Functional':>12} {'Output Norm':>12}")
    print("-" * 36)

    for damage in damage_levels:
        network = DistributedLanguageNetwork(dim=32)
        network.lesion("wernicke", damage)
        result = network.process(test_input)
        output_norm = np.linalg.norm(result["combined"])
        print(f"{damage:>10.0%} {str(network.is_functional()):>12} {output_norm:>12.3f}")


def demo_double_lesion():
    """Show that both regions damaged = non-functional"""
    print("\n" + "=" * 60)
    print("DOUBLE LESION")
    print("Both regions required for function")
    print("=" * 60)

    network = DistributedLanguageNetwork(dim=32)
    np.random.randn(32)

    conditions = [
        ("Intact", 0.0, 0.0),
        ("Broca only", 1.0, 0.0),
        ("Wernicke only", 0.0, 1.0),
        ("Both (mild)", 0.5, 0.5),
        ("Both (severe)", 0.9, 0.9),
        ("Both (complete)", 1.0, 1.0),
    ]

    print(f"\n{'Condition':<20} {'Broca':>8} {'Wernicke':>10} {'Functional':>12}")
    print("-" * 52)

    for name, broca_dmg, wernicke_dmg in conditions:
        network = DistributedLanguageNetwork(dim=32)
        if broca_dmg > 0:
            network.lesion("broca", broca_dmg)
        if wernicke_dmg > 0:
            network.lesion("wernicke", wernicke_dmg)

        functional = network.is_functional()
        print(f"{name:<20} {broca_dmg:>7.0%} {wernicke_dmg:>10.0%} {str(functional):>12}")

    print("\n  KEY FINDING: Both regions contribute to function.")
    print("  Complete loss of either alone is tolerable,")
    print("  but combined severe damage causes dysfunction.")


def demo_full_lesion_experiment():
    """Run full lesion experiment with language processor"""
    print("\n" + "=" * 60)
    print("FULL LESION EXPERIMENT")
    print("Testing language processing with region damage")
    print("=" * 60)

    processor = PredictiveLanguageProcessor(input_dim=64, hidden_dim=32)

    test_sentences = [["the", "cat", "sat"], ["a", "dog", "ran"], ["she", "thinks", "that"]]

    print("\n--- Baseline Processing ---")
    baseline_results = []
    for sentence in test_sentences:
        result = processor.process_utterance(sentence)
        baseline_results.append(result["total_error"])
    baseline_error = np.mean(baseline_results)
    print(f"  Mean error: {baseline_error:.3f}")

    print("\n--- Broca's Lesion Experiment ---")
    result = processor.lesion_experiment("broca", test_sentences, damage_level=1.0)

    print(f"  Region: {result['region']}")
    print(f"  Baseline error: {result['baseline']['mean_error']:.3f}")
    print(f"  Lesioned error: {result['lesioned']['mean_error']:.3f}")
    print(f"  Broca alone functional: {result['broca_alone_functional']}")

    print("\n--- Wernicke's Lesion Experiment ---")
    result = processor.lesion_experiment("wernicke", test_sentences, damage_level=1.0)

    print(f"  Region: {result['region']}")
    print(f"  Baseline error: {result['baseline']['mean_error']:.3f}")
    print(f"  Lesioned error: {result['lesioned']['mean_error']:.3f}")


def demo_recovery_simulation():
    """Simulate recovery after lesion"""
    print("\n" + "=" * 60)
    print("RECOVERY SIMULATION")
    print("Restoring function after damage")
    print("=" * 60)

    network = DistributedLanguageNetwork(dim=32)
    test_input = np.random.randn(32)

    print("\n--- Initial State ---")
    result = network.process(test_input)
    print(f"  Functional: {network.is_functional()}")
    print(f"  Output norm: {np.linalg.norm(result['combined']):.3f}")

    print("\n--- After Broca's Lesion ---")
    network.lesion("broca", 0.8)
    result = network.process(test_input)
    print(f"  Broca damage: {network.broca.damage_level:.0%}")
    print(f"  Functional: {network.is_functional()}")
    print(f"  Output norm: {np.linalg.norm(result['combined']):.3f}")

    print("\n--- After Recovery (Restore) ---")
    network.restore("broca")
    result = network.process(test_input)
    damage_levels = network.get_damage_levels()
    print(f"  Broca damage: {damage_levels['broca']:.0%}")
    print(f"  Functional: {network.is_functional()}")
    print(f"  Output norm: {np.linalg.norm(result['combined']):.3f}")


if __name__ == "__main__":
    demo_distributed_network()
    demo_broca_lesion()
    demo_progressive_lesion()
    demo_double_lesion()
    demo_full_lesion_experiment()
    demo_recovery_simulation()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Key findings demonstrated:
1. Language processing is distributed, not localized
2. Broca's damage alone doesn't cause complete dysfunction
3. Wernicke's can compensate for Broca's damage (and vice versa)
4. Both regions together provide redundancy
5. Only severe bilateral damage causes complete dysfunction

This challenges the classical "Broca = production, Wernicke = comprehension"
model and supports a more distributed view of language in the brain.
""")
