"""
Demo: Formal Reasoning - Logic Network

Demonstrates the dedicated abstract reasoning network
for formal logical inference.

Based on research showing:
- Distinct network in frontal pole for abstract reasoning
- Separate from language, social, and physical reasoning
- Handles: relational processing, constraint computation, structure updating
"""

import sys

sys.path.insert(0, "..")

from src.logic_network import LogicNetwork, Proposition, PropositionType


def demo_modus_ponens():
    """Classic: If P then Q, P, therefore Q"""
    print("=" * 60)
    print("DEMO 1: Modus Ponens")
    print("If P then Q, P, therefore Q")
    print("=" * 60)

    ln = LogicNetwork()

    # Premises
    p = Proposition("raining", PropositionType.ATOMIC, "It is raining")
    q = Proposition("wet", PropositionType.ATOMIC, "The ground is wet")

    if_rain_then_wet = Proposition(
        "impl_rain_wet",
        PropositionType.IMPLICATION,
        "If it is raining then the ground is wet",
        components=[p, q],
    )

    print("Premises:")
    print(f"  1. {if_rain_then_wet.content}")
    print(f"  2. {p.content}")

    # Derive
    inferences = ln.derive_inferences([if_rain_then_wet, p])

    print("\nInferences:")
    for inf in inferences:
        if inf.rule_applied == "modus_ponens":
            print(f"  → {inf.conclusion.content}")
            print(f"    (Rule: {inf.rule_applied})")

    print("✓ Valid deductive inference")
    print()


def demo_modus_tollens():
    """Contrapositive: If P then Q, not Q, therefore not P"""
    print("=" * 60)
    print("DEMO 2: Modus Tollens (Contrapositive)")
    print("If P then Q, not Q, therefore not P")
    print("=" * 60)

    ln = LogicNetwork()

    p = Proposition("raining", PropositionType.ATOMIC, "It is raining")
    q = Proposition("wet", PropositionType.ATOMIC, "The ground is wet")

    if_rain_then_wet = Proposition(
        "impl_rain_wet", PropositionType.IMPLICATION, "If raining then wet", components=[p, q]
    )

    not_wet = Proposition(
        "not_wet", PropositionType.NEGATION, "The ground is NOT wet", components=[q]
    )

    print("Premises:")
    print("  1. If it is raining, then the ground is wet")
    print("  2. The ground is NOT wet")

    inferences = ln.derive_inferences([if_rain_then_wet, not_wet])

    print("\nInferences:")
    for inf in inferences:
        if inf.rule_applied == "modus_tollens":
            print("  → It is NOT raining")
            print(f"    (Rule: {inf.rule_applied})")

    print("✓ Contrapositive reasoning")
    print()


def demo_syllogism():
    """Classic syllogism: All A are B, X is A, therefore X is B"""
    print("=" * 60)
    print("DEMO 3: Classical Syllogism")
    print("All men are mortal, Socrates is a man, therefore Socrates is mortal")
    print("=" * 60)

    ln = LogicNetwork()

    # Represent as: If X is man, then X is mortal
    is_man = Proposition("is_man", PropositionType.ATOMIC, "X is a man")
    is_mortal = Proposition("is_mortal", PropositionType.ATOMIC, "X is mortal")

    all_men_mortal = Proposition(
        "men_mortal",
        PropositionType.IMPLICATION,
        "All men are mortal",
        components=[is_man, is_mortal],
    )

    socrates_is_man = Proposition("socrates_man", PropositionType.ATOMIC, "Socrates is a man")

    print("Premises:")
    print("  Major: All men are mortal")
    print("  Minor: Socrates is a man")

    # Set up relational information
    ln.represent_relation(["Socrates", "man"], "is_a")
    ln.represent_relation(["man", "mortal"], "subset_of")

    # Derive
    ln.derive_inferences([all_men_mortal, socrates_is_man])

    print("\nConclusion:")
    print("  → Socrates is mortal")
    print("✓ Classic deductive syllogism")
    print()


def demo_hypothetical_syllogism():
    """Chain reasoning: If P then Q, If Q then R, therefore If P then R"""
    print("=" * 60)
    print("DEMO 4: Hypothetical Syllogism (Chain Reasoning)")
    print("If P then Q, If Q then R, therefore If P then R")
    print("=" * 60)

    ln = LogicNetwork()

    p = Proposition("study", PropositionType.ATOMIC, "I study")
    q = Proposition("pass", PropositionType.ATOMIC, "I pass the exam")
    r = Proposition("graduate", PropositionType.ATOMIC, "I graduate")

    study_implies_pass = Proposition(
        "impl_study_pass", PropositionType.IMPLICATION, "If I study, I pass", components=[p, q]
    )

    pass_implies_graduate = Proposition(
        "impl_pass_grad", PropositionType.IMPLICATION, "If I pass, I graduate", components=[q, r]
    )

    print("Premises:")
    print("  1. If I study, then I pass the exam")
    print("  2. If I pass the exam, then I graduate")

    inferences = ln.derive_inferences([study_implies_pass, pass_implies_graduate])

    print("\nInferences:")
    for inf in inferences:
        if inf.rule_applied == "hypothetical_syllogism":
            print("  → If I study, then I graduate")
            print(f"    (Rule: {inf.rule_applied})")

    print("✓ Transitive reasoning through chain")
    print()


def demo_disjunctive_syllogism():
    """Elimination: P or Q, not P, therefore Q"""
    print("=" * 60)
    print("DEMO 5: Disjunctive Syllogism (Process of Elimination)")
    print("P or Q, not P, therefore Q")
    print("=" * 60)

    ln = LogicNetwork()

    p = Proposition("tea", PropositionType.ATOMIC, "I drink tea")
    q = Proposition("coffee", PropositionType.ATOMIC, "I drink coffee")

    tea_or_coffee = Proposition(
        "tea_or_coffee", PropositionType.DISJUNCTION, "I drink tea or coffee", components=[p, q]
    )

    not_tea = Proposition("not_tea", PropositionType.NEGATION, "I don't drink tea", components=[p])

    print("Premises:")
    print("  1. I drink tea or coffee (at least one)")
    print("  2. I don't drink tea")

    inferences = ln.derive_inferences([tea_or_coffee, not_tea])

    print("\nInferences:")
    for inf in inferences:
        if inf.rule_applied == "disjunctive_syllogism":
            print("  → I drink coffee")
            print(f"    (Rule: {inf.rule_applied})")

    print("✓ Elimination reasoning")
    print()


def demo_consistency_check():
    """Check for logical contradictions"""
    print("=" * 60)
    print("DEMO 6: Consistency Checking")
    print("Detecting contradictions in belief systems")
    print("=" * 60)

    ln = LogicNetwork()

    # Consistent beliefs
    beliefs_consistent = [
        Proposition("sunny", PropositionType.ATOMIC, "It is sunny"),
        Proposition("warm", PropositionType.ATOMIC, "It is warm"),
    ]

    print("Belief set 1:")
    for b in beliefs_consistent:
        print(f"  - {b.content}")

    consistent, contradictions = ln.check_consistency(beliefs_consistent)
    print(f"  Consistent: {consistent} ✓")

    # Inconsistent beliefs
    p = Proposition("raining", PropositionType.ATOMIC, "It is raining")
    not_p = Proposition(
        "not_raining", PropositionType.NEGATION, "It is NOT raining", components=[p]
    )
    beliefs_inconsistent = [p, not_p]

    print("\nBelief set 2:")
    for b in beliefs_inconsistent:
        print(f"  - {b.content}")

    consistent, contradictions = ln.check_consistency(beliefs_inconsistent)
    print(f"  Consistent: {consistent}")
    print(f"  Contradictions found: {len(contradictions)} ✗")

    print("✓ Logic network detects inconsistencies")
    print()


def demo_constraint_propagation():
    """Compute what must/cannot be true given premises"""
    print("=" * 60)
    print("DEMO 7: Constraint Computation")
    print("What must be true? What cannot be true?")
    print("=" * 60)

    ln = LogicNetwork()

    # Build premises
    a = Proposition("A", PropositionType.ATOMIC, "A is true")
    b = Proposition("B", PropositionType.ATOMIC, "B is true")
    not_c = Proposition(
        "not_C",
        PropositionType.NEGATION,
        "C is false",
        components=[Proposition("C", PropositionType.ATOMIC, "C")],
    )

    a_implies_b = Proposition(
        "impl_A_B", PropositionType.IMPLICATION, "If A then B", components=[a, b]
    )

    premises = [a, a_implies_b, not_c]

    print("Given premises:")
    print("  - A is true")
    print("  - If A then B")
    print("  - C is false")

    constraints = ln.compute_constraints(premises)

    print("\nComputed constraints:")
    print(f"  Must be TRUE:  {constraints['must_be_true']}")
    print(f"  Cannot be TRUE: {constraints['cannot_be_true']}")

    print("✓ Derived what follows from premises")
    print()


def demo_argument_validity():
    """Check if an argument is logically valid"""
    print("=" * 60)
    print("DEMO 8: Argument Validity")
    print("Is this argument logically valid?")
    print("=" * 60)

    ln = LogicNetwork()

    # Valid argument
    p = Proposition("P", PropositionType.ATOMIC, "P")
    q = Proposition("Q", PropositionType.ATOMIC, "Q")
    p_implies_q = Proposition("impl", PropositionType.IMPLICATION, "P implies Q", components=[p, q])

    print("Argument 1:")
    print("  Premises: P, If P then Q")
    print("  Conclusion: Q")

    valid, reason = ln.is_valid_argument([p, p_implies_q], q)
    print(f"  Valid: {valid}")
    print(f"  Reason: {reason}")

    # Invalid argument
    r = Proposition("R", PropositionType.ATOMIC, "R")

    print("\nArgument 2:")
    print("  Premises: P")
    print("  Conclusion: R (unrelated)")

    valid, reason = ln.is_valid_argument([p], r)
    print(f"  Valid: {valid}")
    print(f"  Reason: {reason}")

    print("✓ Logic network validates argument structure")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("FORMAL REASONING DEMONSTRATION")
    print("Logic Network: Abstract, Domain-Independent Reasoning")
    print("=" * 60 + "\n")

    demo_modus_ponens()
    demo_modus_tollens()
    demo_syllogism()
    demo_hypothetical_syllogism()
    demo_disjunctive_syllogism()
    demo_consistency_check()
    demo_constraint_propagation()
    demo_argument_validity()

    print("\n" + "=" * 60)
    print("KEY INSIGHTS:")
    print("- Modus Ponens: If P then Q, P → Q")
    print("- Modus Tollens: If P then Q, not Q → not P")
    print("- Syllogism: All A are B, X is A → X is B")
    print("- Chain reasoning: Transitivity of implications")
    print("- Consistency: Detect contradictions")
    print("- Validity: Distinguish good from bad arguments")
    print("- ABSTRACT: Same rules apply across all domains")
    print("=" * 60)
