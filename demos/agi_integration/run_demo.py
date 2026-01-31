#!/usr/bin/env python3
"""
AGI Integration Demo Runner

Demonstrates the IntegratedAGIAgent in action across several scenarios:
1. Medical diagnosis reasoning
2. Physics concept learning
3. Language understanding

Usage:
    python run_demo.py
    python run_demo.py --scenario medical
    python run_demo.py --scenario physics
    python run_demo.py --scenario all
"""

import argparse
import numpy as np
from agent import IntegratedAGIAgent, AgentConfig


def demo_medical_diagnosis():
    """Demonstrate medical reasoning scenario."""
    print("\n" + "="*60)
    print("SCENARIO 1: Medical Diagnosis Reasoning")
    print("="*60)

    agent = IntegratedAGIAgent(AgentConfig(random_seed=42))

    # Learn medical concepts
    print("\n1. Learning medical concepts...")

    # Symptoms
    fever_examples = [np.random.randn(256) + np.array([1.0] + [0]*255) for _ in range(5)]
    agent.learn_concept("fever", fever_examples, category="semantic")

    cough_examples = [np.random.randn(256) + np.array([0, 1.0] + [0]*254) for _ in range(5)]
    agent.learn_concept("cough", cough_examples, category="semantic")

    fatigue_examples = [np.random.randn(256) + np.array([0, 0, 1.0] + [0]*253) for _ in range(5)]
    agent.learn_concept("fatigue", fatigue_examples, category="semantic")

    # Diseases
    flu_examples = [np.random.randn(256) + np.array([1.0, 0.8, 0.6] + [0]*253) for _ in range(10)]
    agent.learn_concept("flu", flu_examples, category="semantic")

    cold_examples = [np.random.randn(256) + np.array([0.3, 1.0, 0.2] + [0]*253) for _ in range(10)]
    agent.learn_concept("cold", cold_examples, category="semantic")

    print(f"   Learned {len(agent.list_concepts())} concepts")

    # Add causal relations
    print("\n2. Building causal model...")
    agent.add_causal_relation("flu", "fever", strength=0.9)
    agent.add_causal_relation("flu", "cough", strength=0.7)
    agent.add_causal_relation("flu", "fatigue", strength=0.8)
    agent.add_causal_relation("cold", "cough", strength=0.9)
    agent.add_causal_relation("cold", "fatigue", strength=0.3)

    stats = agent.statistics()
    print(f"   Established {stats['causal_relations']} causal relations")

    # Causal reasoning queries
    print("\n3. Causal reasoning queries...")

    # Intervention query: If we treat flu, what happens to fever?
    result = agent.reason_causally("intervention", "flu", "fever", intervention_value=0.0)
    print(f"\n   Q: If flu is treated (set to 0), what happens to fever?")
    if result.abstained:
        print(f"   A: Abstained due to uncertainty ({result.uncertainty:.2f})")
    else:
        print(f"   A: Effect = {result.answer:.2f} (confidence: {result.confidence:.2f})")
    print(f"   Trace: {' -> '.join(result.reasoning_trace)}")

    # Counterfactual: What if patient didn't have flu?
    result = agent.reason_causally("counterfactual", "flu", "fatigue")
    print(f"\n   Q: What would fatigue be if patient hadn't had flu?")
    if result.abstained:
        print(f"   A: Abstained due to uncertainty ({result.uncertainty:.2f})")
    else:
        print(f"   A: Effect = {result.answer:.2f} (confidence: {result.confidence:.2f})")

    # Consolidate learning
    print("\n4. Sleep consolidation...")
    stats = agent.sleep_consolidate(n_cycles=3)
    print(f"   Replayed {stats.memories_replayed} memories")
    print(f"   Strengthened {stats.memories_strengthened} memories")

    # Query with uncertainty
    print("\n5. Prediction with uncertainty...")
    prediction, confidence, abstained = agent.predict_with_uncertainty("flu")
    print(f"   Query: 'flu'")
    print(f"   Confidence: {confidence:.2f}")
    print(f"   Abstained: {abstained}")

    print(f"\nAgent statistics: {agent.statistics()}")


def demo_physics_learning():
    """Demonstrate physics concept learning scenario."""
    print("\n" + "="*60)
    print("SCENARIO 2: Physics Concept Learning")
    print("="*60)

    agent = IntegratedAGIAgent(AgentConfig(random_seed=123))

    # Learn physics concepts
    print("\n1. Learning physics concepts...")

    np.random.seed(123)

    # Forces
    force_base = np.zeros(256)
    force_base[0:10] = 1.0
    force_examples = [force_base + np.random.randn(256) * 0.1 for _ in range(8)]
    agent.learn_concept("force", force_examples, category="semantic")

    # Mass
    mass_base = np.zeros(256)
    mass_base[10:20] = 1.0
    mass_examples = [mass_base + np.random.randn(256) * 0.1 for _ in range(8)]
    agent.learn_concept("mass", mass_examples, category="semantic")

    # Acceleration
    accel_base = np.zeros(256)
    accel_base[20:30] = 1.0
    accel_examples = [accel_base + np.random.randn(256) * 0.1 for _ in range(8)]
    agent.learn_concept("acceleration", accel_examples, category="semantic")

    # Velocity
    vel_base = np.zeros(256)
    vel_base[30:40] = 1.0
    vel_examples = [vel_base + np.random.randn(256) * 0.1 for _ in range(8)]
    agent.learn_concept("velocity", vel_examples, category="semantic")

    # Position
    pos_base = np.zeros(256)
    pos_base[40:50] = 1.0
    pos_examples = [pos_base + np.random.randn(256) * 0.1 for _ in range(8)]
    agent.learn_concept("position", pos_examples, category="semantic")

    print(f"   Learned {len(agent.list_concepts())} concepts")

    # Build causal model (Newton's laws)
    print("\n2. Building causal model (Newton's laws)...")
    agent.add_causal_relation("force", "acceleration", strength=1.0)  # F = ma
    agent.add_causal_relation("mass", "acceleration", strength=0.8)   # Inverse relation
    agent.add_causal_relation("acceleration", "velocity", strength=1.0)  # a = dv/dt
    agent.add_causal_relation("velocity", "position", strength=1.0)  # v = dx/dt

    # Causal chain reasoning
    print("\n3. Causal chain reasoning...")

    result = agent.reason_causally("intervention", "force", "velocity", intervention_value=2.0)
    print(f"\n   Q: If force is doubled, how does velocity change?")
    if result.abstained:
        print(f"   A: Abstained due to uncertainty")
    else:
        print(f"   A: Effect = {result.answer:.2f} (indirect via acceleration)")

    result = agent.reason_causally("counterfactual", "force", "position")
    print(f"\n   Q: What would position be if there were no force?")
    if result.abstained:
        print(f"   A: Abstained")
    else:
        print(f"   A: Effect = {result.answer:.2f}")
        print(f"   Confidence: {result.confidence:.2f}")

    # Time evolution with decay
    print("\n4. Simulating memory decay over time...")
    for day in range(5):
        agent.advance_time(1.0)
        force_concept = agent.get_concept("force")
        print(f"   Day {day+1}: force concept strength = {force_concept.strength:.3f}")

    # Consolidation restores strength
    print("\n5. Sleep consolidation to restore memories...")
    stats = agent.sleep_consolidate(n_cycles=5)
    force_concept = agent.get_concept("force")
    print(f"   After consolidation: force concept strength = {force_concept.strength:.3f}")

    print(f"\nAgent statistics: {agent.statistics()}")


def demo_language_concepts():
    """Demonstrate language concept learning scenario."""
    print("\n" + "="*60)
    print("SCENARIO 3: Language Understanding")
    print("="*60)

    agent = IntegratedAGIAgent(AgentConfig(
        random_seed=456,
        uncertainty_threshold=0.4,  # More lenient
    ))

    np.random.seed(456)

    # Learn word concepts with roles
    print("\n1. Learning word concepts...")

    # Verbs
    chase_base = np.zeros(256)
    chase_base[0:20] = np.random.randn(20)
    chase_examples = [chase_base + np.random.randn(256) * 0.1 for _ in range(5)]
    agent.learn_concept("chase", chase_examples, category="semantic")

    eat_base = np.zeros(256)
    eat_base[20:40] = np.random.randn(20)
    eat_examples = [eat_base + np.random.randn(256) * 0.1 for _ in range(5)]
    agent.learn_concept("eat", eat_examples, category="semantic")

    # Nouns
    dog_base = np.zeros(256)
    dog_base[40:60] = np.random.randn(20)
    dog_examples = [dog_base + np.random.randn(256) * 0.1 for _ in range(5)]
    agent.learn_concept("dog", dog_examples, category="semantic")

    cat_base = np.zeros(256)
    cat_base[60:80] = np.random.randn(20)
    cat_examples = [cat_base + np.random.randn(256) * 0.1 for _ in range(5)]
    agent.learn_concept("cat", cat_examples, category="semantic")

    food_base = np.zeros(256)
    food_base[80:100] = np.random.randn(20)
    food_examples = [food_base + np.random.randn(256) * 0.1 for _ in range(5)]
    agent.learn_concept("food", food_examples, category="semantic")

    print(f"   Learned {len(agent.list_concepts())} concepts")

    # Build semantic relations (not strictly causal)
    print("\n2. Building semantic relations...")
    agent.add_causal_relation("dog", "chase", strength=0.8)  # Dogs often chase
    agent.add_causal_relation("chase", "cat", strength=0.7)  # Chase targets cats
    agent.add_causal_relation("dog", "eat", strength=0.9)    # Dogs eat
    agent.add_causal_relation("eat", "food", strength=1.0)   # Eating involves food

    # Query semantic relationships
    print("\n3. Semantic reasoning...")

    result = agent.reason_causally("association", "dog", "food")
    print(f"\n   Q: How associated are 'dog' and 'food'?")
    if result.abstained:
        print(f"   A: Abstained")
    else:
        print(f"   A: Association = {result.answer:.2f}")

    result = agent.reason_causally("association", "cat", "food")
    print(f"\n   Q: How associated are 'cat' and 'food'?")
    if result.abstained:
        print(f"   A: Abstained - no direct relation found")
    else:
        print(f"   A: Association = {result.answer:.2f}")

    # Test uncertainty-based abstention
    print("\n4. Testing uncertainty-based abstention...")

    # Add a high-uncertainty concept
    vague_examples = [np.random.randn(256) for _ in range(2)]  # High variance
    agent.learn_concept("ambiguous", vague_examples, category="semantic")

    ambig = agent.get_concept("ambiguous")
    print(f"   'ambiguous' concept uncertainty: {ambig.uncertainty:.2f}")

    # Build relation
    agent.add_causal_relation("ambiguous", "dog", strength=0.5)

    result = agent.reason_causally("association", "ambiguous", "dog")
    print(f"\n   Q: Association between 'ambiguous' and 'dog'?")
    if result.abstained:
        print(f"   A: Abstained due to uncertainty ({result.uncertainty:.2f})")
    else:
        print(f"   A: Association = {result.answer:.2f}")

    # Consolidation reduces uncertainty
    print("\n5. Consolidation to reduce uncertainty...")
    agent.sleep_consolidate(n_cycles=5)
    ambig = agent.get_concept("ambiguous")
    print(f"   After consolidation: 'ambiguous' uncertainty = {ambig.uncertainty:.2f}")

    result = agent.reason_causally("association", "ambiguous", "dog")
    print(f"\n   Retry query after consolidation:")
    if result.abstained:
        print(f"   A: Still abstained")
    else:
        print(f"   A: Association = {result.answer:.2f}, confidence = {result.confidence:.2f}")

    print(f"\nAgent statistics: {agent.statistics()}")


def main():
    parser = argparse.ArgumentParser(description="AGI Integration Demo")
    parser.add_argument(
        "--scenario",
        choices=["medical", "physics", "language", "all"],
        default="all",
        help="Which scenario to run"
    )
    args = parser.parse_args()

    print("\n" + "#"*60)
    print("#" + " "*18 + "AGI INTEGRATION DEMO" + " "*18 + "#")
    print("#"*60)
    print("\nThis demo shows how neuro-causal, neuro-abstract, neuro-robust,")
    print("and sleep-consolidation modules work together in an integrated agent.")

    if args.scenario in ["medical", "all"]:
        demo_medical_diagnosis()

    if args.scenario in ["physics", "all"]:
        demo_physics_learning()

    if args.scenario in ["language", "all"]:
        demo_language_concepts()

    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
