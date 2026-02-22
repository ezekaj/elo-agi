"""
Grammar Learning Demo

Demonstrates selective inhibition for impossible grammars
and language acquisition simulation.

Key 2025 finding: Broca's area selectively inhibits impossible
(not possible) grammars, suggesting innate Universal Grammar constraints.
"""

import numpy as np
import sys

sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from src.grammar_manifold import (
    GrammarConstraintManifold,
    UniversalGrammar,
    ImpossibleGrammarGenerator,
)
from src.predictive_language import PredictiveLanguageProcessor, LanguageAcquisitionSimulator


def demo_grammar_constraints():
    """Show the grammar constraint manifold"""
    print("=" * 60)
    print("GRAMMAR CONSTRAINT MANIFOLD")
    print("Space of possible human grammars")
    print("=" * 60)

    manifold = GrammarConstraintManifold(dim=32)

    print(f"\n--- Manifold Properties ---")
    print(f"  Dimension: {manifold.dim}")
    print(f"  Number of constraints: {len(manifold.constraints)}")
    print(f"  Region radius: {manifold.radius}")

    print(f"\n--- Constraints ---")
    for constraint in manifold.constraints:
        hard_str = "(HARD)" if constraint.is_hard else "(soft)"
        print(f"  {constraint.name}: weight={constraint.weight} {hard_str}")


def demo_possible_vs_impossible():
    """Compare possible vs impossible grammars"""
    print("\n" + "=" * 60)
    print("POSSIBLE vs IMPOSSIBLE GRAMMARS")
    print("Broca's area selectively inhibits impossible grammars")
    print("=" * 60)

    gen = ImpossibleGrammarGenerator(dim=32)
    manifold = gen.manifold

    print("\n--- Generating Example Grammars ---")

    # Possible grammars
    print("\nPossible grammars (near UG center):")
    for i in range(3):
        params = gen.generate_possible()
        state = manifold.evaluate(params)
        inhibition = manifold.inhibition_signal(params)
        print(
            f"  Grammar {i + 1}: violation={state.violation_score:.3f}, "
            f"possible={state.is_possible}, inhibition={inhibition:.3f}"
        )

    # Impossible grammars
    print("\nImpossible grammars (violate UG constraints):")
    for violation_type in ["random", "structure", "unbounded"]:
        params = gen.generate_impossible(violation_type)
        state = manifold.evaluate(params)
        inhibition = manifold.inhibition_signal(params)
        print(
            f"  {violation_type}: violation={state.violation_score:.3f}, "
            f"possible={state.is_possible}, inhibition={inhibition:.3f}"
        )


def demo_universal_grammar():
    """Show Universal Grammar principles"""
    print("\n" + "=" * 60)
    print("UNIVERSAL GRAMMAR")
    print("Innate constraints on learnable grammar")
    print("=" * 60)

    ug = UniversalGrammar(dim=32)

    print(f"\n--- UG Principles ---")
    for principle in ug.principles:
        print(f"  - {principle}")

    print(f"\n--- UG Parameters (Principles & Parameters framework) ---")
    for param, value in ug.parameters.items():
        print(f"  {param}: {value}")

    print(f"\n--- Evaluating Random Grammars ---")
    for i in range(5):
        params = np.random.randn(32) * 0.5
        results = ug.evaluate(params)
        compatible = ug.is_ug_compatible(params)
        print(f"\n  Grammar {i + 1}: UG-compatible={compatible}")
        print(f"    Overall score: {results['overall']:.3f}")
        for principle in ug.principles:
            print(f"    {principle}: {results[principle]:.3f}")


def demo_selective_inhibition():
    """Demonstrate selective inhibition pattern"""
    print("\n" + "=" * 60)
    print("SELECTIVE INHIBITION")
    print("Key finding: Broca's inhibits impossible, not possible grammars")
    print("=" * 60)

    gen = ImpossibleGrammarGenerator(dim=32)
    manifold = gen.manifold

    n_samples = 20
    possible_inhibitions = []
    impossible_inhibitions = []

    for _ in range(n_samples):
        # Possible grammar
        possible = gen.generate_possible()
        possible_inhibitions.append(manifold.inhibition_signal(possible))

        # Impossible grammar
        impossible = gen.generate_impossible()
        impossible_inhibitions.append(manifold.inhibition_signal(impossible))

    print(f"\n--- Inhibition Statistics (n={n_samples} each) ---")
    print(f"\nPossible grammars:")
    print(f"  Mean inhibition: {np.mean(possible_inhibitions):.3f}")
    print(f"  Std inhibition:  {np.std(possible_inhibitions):.3f}")
    print(f"  Max inhibition:  {np.max(possible_inhibitions):.3f}")

    print(f"\nImpossible grammars:")
    print(f"  Mean inhibition: {np.mean(impossible_inhibitions):.3f}")
    print(f"  Std inhibition:  {np.std(impossible_inhibitions):.3f}")
    print(f"  Min inhibition:  {np.min(impossible_inhibitions):.3f}")

    selectivity = np.mean(impossible_inhibitions) - np.mean(possible_inhibitions)
    print(f"\n--- Selectivity ---")
    print(f"  Difference (impossible - possible): {selectivity:.3f}")
    if selectivity > 0:
        print("  ✓ Broca's shows selective inhibition for impossible grammars")
    else:
        print("  Note: Low selectivity may indicate overlapping distributions")


def demo_acquisition_simulation():
    """Simulate language acquisition"""
    print("\n" + "=" * 60)
    print("LANGUAGE ACQUISITION SIMULATION")
    print("Learning possible vs impossible grammars")
    print("=" * 60)

    processor = PredictiveLanguageProcessor(input_dim=64, hidden_dim=32)
    simulator = LanguageAcquisitionSimulator(processor)
    gen = ImpossibleGrammarGenerator(dim=32)

    print("\n--- Attempting to learn POSSIBLE grammar ---")
    possible_samples = [gen.generate_possible() for _ in range(10)]
    result = simulator.attempt_grammar_learning(possible_samples, n_epochs=5)

    print(f"  Success: {result['success']}")
    print(f"  Final inhibition: {result['final_inhibition']:.3f}")
    print(f"  Inhibition history: {[f'{x:.2f}' for x in result['inhibition_history']]}")

    print("\n--- Attempting to learn IMPOSSIBLE grammar ---")
    impossible_samples = [gen.generate_impossible() for _ in range(10)]
    result = simulator.attempt_grammar_learning(impossible_samples, n_epochs=5)

    print(f"  Success: {result['success']}")
    print(f"  Final inhibition: {result['final_inhibition']:.3f}")
    print(f"  Inhibition history: {[f'{x:.2f}' for x in result['inhibition_history']]}")

    print("\n--- Comparison ---")
    comparison = simulator.compare_possible_vs_impossible(possible_samples, impossible_samples)
    print(
        f"  Possible grammar - final inhibition: {comparison['possible']['final_inhibition']:.3f}"
    )
    print(
        f"  Impossible grammar - final inhibition: {comparison['impossible']['final_inhibition']:.3f}"
    )
    print(f"  Selective inhibition: {comparison['selective_inhibition']:.3f}")


if __name__ == "__main__":
    demo_grammar_constraints()
    demo_possible_vs_impossible()
    demo_universal_grammar()
    demo_selective_inhibition()
    demo_acquisition_simulation()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Key findings demonstrated:
1. Grammar space has constraints (manifold of possible grammars)
2. Universal Grammar defines innate principles and parameters
3. Broca's area selectively inhibits impossible grammars
4. This selectivity enables learning only human-possible languages
5. Language acquisition succeeds for possible, fails for impossible grammars
""")
