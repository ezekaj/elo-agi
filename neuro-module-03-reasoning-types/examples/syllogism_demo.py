"""
Syllogism Demonstration - Deductive Reasoning

Classic logical syllogisms demonstrating certain deductive inference.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.logical.deductive import (
    DeductiveReasoner, Proposition, PropositionType
)


def demo_socrates_syllogism():
    """
    The classic Socrates syllogism:
    All men are mortal
    Socrates is a man
    Therefore, Socrates is mortal
    """
    print("=" * 60)
    print("CLASSIC SYLLOGISM: SOCRATES IS MORTAL")
    print("=" * 60)

    reasoner = DeductiveReasoner()

    major = Proposition(
        proposition_id='p1',
        prop_type=PropositionType.UNIVERSAL,
        content='All men are mortal',
        subject='men',
        predicate='mortal'
    )

    minor = Proposition(
        proposition_id='p2',
        prop_type=PropositionType.ATOMIC,
        content='Socrates is a man',
        subject='Socrates',
        predicate='men'
    )

    print("\nPremises:")
    print(f"  Major: {major}")
    print(f"  Minor: {minor}")

    result = reasoner.syllogism(major, minor)

    print(f"\nConclusion:")
    if result.is_valid and result.conclusion:
        print(f"  {result.conclusion}")
        print(f"  Valid: {result.is_valid}")
        print(f"  Explanation: {result.validity_explanation}")
    else:
        print(f"  No valid conclusion")

    print("\n--- Reasoning Chain ---")
    print("1. Major premise establishes: All members of set 'men' have property 'mortal'")
    print("2. Minor premise establishes: 'Socrates' is a member of set 'men'")
    print("3. By universal instantiation: 'Socrates' inherits property 'mortal'")
    print("4. Conclusion is CERTAIN (not probabilistic)")

    return result


def demo_chained_syllogisms():
    """
    Chained syllogisms demonstrating transitivity:
    All Greeks are humans
    All humans are mammals
    All mammals are animals
    Therefore: All Greeks are animals
    """
    print("\n" + "=" * 60)
    print("CHAINED SYLLOGISMS: TRANSITIVE REASONING")
    print("=" * 60)

    reasoner = DeductiveReasoner()

    p1 = Proposition('p1', PropositionType.UNIVERSAL, 'All Greeks are humans',
                    subject='Greeks', predicate='humans')
    p2 = Proposition('p2', PropositionType.UNIVERSAL, 'All humans are mammals',
                    subject='humans', predicate='mammals')
    p3 = Proposition('p3', PropositionType.UNIVERSAL, 'All mammals are animals',
                    subject='mammals', predicate='animals')

    premises = [p1, p2, p3]

    print("\nPremises:")
    for p in premises:
        print(f"  {p}")
        reasoner.add_premise(p)

    print("\n--- Derivation Steps ---")

    step1 = reasoner.derive([p1, p2])
    greeks_mammals = [i for i in step1
                     if i.conclusion.subject == 'Greeks' and i.conclusion.predicate == 'mammals']
    if greeks_mammals:
        print(f"Step 1: From P1 + P2 → {greeks_mammals[0].conclusion}")
        reasoner.add_premise(greeks_mammals[0].conclusion)

        step2 = reasoner.derive([greeks_mammals[0].conclusion, p3])
        greeks_animals = [i for i in step2
                        if i.conclusion.subject == 'Greeks' and i.conclusion.predicate == 'animals']
        if greeks_animals:
            print(f"Step 2: From (P1+P2) + P3 → {greeks_animals[0].conclusion}")

    print("\n--- Final Result ---")
    print("From 'Greeks → humans → mammals → animals'")
    print("We derive: 'All Greeks are animals'")
    print("This is CERTAIN given the premises are true")


def demo_modus_tollens():
    """
    Modus Tollens demonstration:
    If P then Q
    Not Q
    Therefore, Not P
    """
    print("\n" + "=" * 60)
    print("MODUS TOLLENS: DENYING THE CONSEQUENT")
    print("=" * 60)

    reasoner = DeductiveReasoner()

    p = Proposition('p', PropositionType.ATOMIC, 'it is raining',
                   subject='weather', predicate='raining')
    q = Proposition('q', PropositionType.ATOMIC, 'the ground is wet',
                   subject='ground', predicate='wet')

    conditional = Proposition(
        'if_p_then_q',
        PropositionType.CONDITIONAL,
        'If it is raining, then the ground is wet',
        components=[p, q]
    )

    not_q = Proposition(
        'not_q',
        PropositionType.NEGATION,
        'The ground is not wet',
        components=[q]
    )

    print("\nPremises:")
    print(f"  Conditional: {conditional}")
    print(f"  Negation: {not_q}")

    result = reasoner.modus_tollens(conditional, not_q)

    print(f"\nConclusion:")
    if result:
        print(f"  {result}")
        print("\nInterpretation: It is NOT raining")
    else:
        print("  Could not apply modus tollens")

    print("\n--- Reasoning Pattern ---")
    print("1. If P then Q (raining → wet ground)")
    print("2. Not Q (ground is NOT wet)")
    print("3. Therefore, Not P (it is NOT raining)")
    print("\nThis is valid because:")
    print("  - If raining always causes wet ground")
    print("  - And the ground is dry")
    print("  - Then it cannot be raining")


def compare_with_induction():
    """
    Compare deductive certainty with inductive probability
    """
    print("\n" + "=" * 60)
    print("DEDUCTION VS INDUCTION: CERTAINTY VS PROBABILITY")
    print("=" * 60)

    print("\n--- DEDUCTIVE (Certain) ---")
    print("Premise 1: All swans are birds")
    print("Premise 2: All birds have feathers")
    print("Conclusion: All swans have feathers")
    print("Confidence: 100% (if premises are true)")

    print("\n--- INDUCTIVE (Probabilistic) ---")
    print("Observation 1: This swan is white")
    print("Observation 2: That swan is white")
    print("Observation 3: Another swan is white")
    print("...")
    print("Observation N: All observed swans are white")
    print("Hypothesis: All swans are white")
    print("Confidence: High but NOT certain")
    print("(Can be refuted by a single black swan!)")

    print("\n--- KEY DIFFERENCE ---")
    print("Deduction: Truth-preserving, conclusion cannot be false if premises true")
    print("Induction: Ampliative, conclusion goes beyond premises, can be wrong")


if __name__ == '__main__':
    demo_socrates_syllogism()
    demo_chained_syllogisms()
    demo_modus_tollens()
    compare_with_induction()
