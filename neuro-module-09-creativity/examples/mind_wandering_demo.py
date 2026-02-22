"""
Mind Wandering Demo - Default Mode Network in Action

Demonstrates spontaneous thought generation and distant association finding.

Key insight: Mind wandering allows creative exploration of semantic space,
finding unexpected connections between concepts.
"""

import sys

sys.path.insert(0, ".")

from src.networks import DefaultModeNetwork


def main():
    print("=" * 60)
    print("MIND WANDERING DEMONSTRATION")
    print("Default Mode Network - Spontaneous Thought Generation")
    print("=" * 60)

    # Create DMN
    dmn = DefaultModeNetwork()

    # Setup a rich semantic network about art and nature
    print("\n--- Setting up Semantic Memory ---")

    concepts = [
        # Nature concepts
        ("ocean", "Vast body of saltwater", {"nature": 0.9, "blue": 0.8, "vast": 0.9, "calm": 0.6}),
        (
            "mountain",
            "Large natural elevation",
            {"nature": 0.9, "tall": 0.9, "solid": 0.8, "majestic": 0.7},
        ),
        (
            "forest",
            "Dense tree coverage",
            {"nature": 1.0, "green": 0.9, "life": 0.8, "mysterious": 0.6},
        ),
        ("river", "Flowing water", {"nature": 0.9, "flow": 0.9, "journey": 0.7, "change": 0.6}),
        ("sky", "Atmosphere above", {"nature": 0.8, "blue": 0.7, "vast": 0.8, "freedom": 0.7}),
        (
            "sunset",
            "Evening light",
            {"nature": 0.8, "warm": 0.9, "beautiful": 0.9, "ephemeral": 0.8},
        ),
        # Art concepts
        ("painting", "Visual art on canvas", {"art": 1.0, "visual": 0.9, "expression": 0.8}),
        ("music", "Auditory art form", {"art": 1.0, "auditory": 0.9, "emotion": 0.9, "flow": 0.7}),
        (
            "poetry",
            "Literary art form",
            {"art": 1.0, "linguistic": 0.9, "emotion": 0.8, "rhythm": 0.6},
        ),
        ("dance", "Movement art", {"art": 1.0, "kinetic": 0.9, "flow": 0.8, "expression": 0.8}),
        # Abstract concepts
        ("freedom", "Liberation", {"abstract": 0.9, "positive": 0.8, "vast": 0.6}),
        ("journey", "Travel or progress", {"abstract": 0.8, "change": 0.7, "growth": 0.7}),
        ("emotion", "Feeling state", {"abstract": 1.0, "inner": 0.9, "human": 0.9}),
        ("time", "Temporal flow", {"abstract": 1.0, "change": 0.8, "flow": 0.7}),
    ]

    for concept_id, content, features in concepts:
        dmn.add_concept(concept_id, content, features)
        print(f"  Added: {concept_id}")

    print(f"\nTotal concepts: {len(dmn.concepts)}")

    # Create associations
    print("\n--- Creating Associations ---")

    associations = [
        # Nature-nature associations
        ("ocean", "sky", 0.7, "visual"),
        ("ocean", "river", 0.8, "water"),
        ("mountain", "sky", 0.8, "visual"),
        ("forest", "river", 0.6, "natural"),
        ("sunset", "sky", 0.9, "visual"),
        ("sunset", "ocean", 0.7, "visual"),
        # Art associations
        ("painting", "sunset", 0.8, "subject"),
        ("music", "emotion", 0.9, "evokes"),
        ("poetry", "emotion", 0.9, "expresses"),
        ("dance", "music", 0.9, "accompanies"),
        ("painting", "emotion", 0.7, "evokes"),
        # Abstract connections
        ("river", "journey", 0.8, "metaphor"),
        ("river", "time", 0.7, "metaphor"),
        ("sky", "freedom", 0.8, "metaphor"),
        ("ocean", "emotion", 0.6, "metaphor"),
        ("music", "river", 0.5, "flow"),  # Creative connection
        ("dance", "river", 0.4, "flow"),  # Distant connection
        # Cross-domain (creative) associations
        ("forest", "poetry", 0.4, "inspiration"),
        ("mountain", "emotion", 0.5, "evokes"),
        ("sunset", "music", 0.4, "inspiration"),
    ]

    for source, target, strength, assoc_type in associations:
        dmn.create_association(source, target, strength, assoc_type)

    print(f"Created {len(associations)} associations")

    # Demonstrate spontaneous thought
    print("\n" + "=" * 60)
    print("SPONTANEOUS THOUGHT GENERATION")
    print("=" * 60)

    print("\n--- Seeded thought (starting from 'ocean') ---")
    for i in range(3):
        thought = dmn.generate_spontaneous_thought(seed="ocean")
        print(f"\nThought {i + 1}:")
        print(f"  Concepts: {thought.concepts}")
        print(f"  Novelty: {thought.novelty_score:.3f}")
        print(f"  Coherence: {thought.coherence_score:.3f}")
        print(f"  Associations: {thought.associations_used}")

    print("\n--- Unseeded thought (random starting point) ---")
    for i in range(3):
        thought = dmn.generate_spontaneous_thought()
        print(f"\nThought {i + 1}:")
        print(f"  Concepts: {thought.concepts}")
        print(f"  Novelty: {thought.novelty_score:.3f}")
        print(f"  Coherence: {thought.coherence_score:.3f}")

    # Demonstrate mind wandering
    print("\n" + "=" * 60)
    print("MIND WANDERING")
    print("Free-flowing thought generation")
    print("=" * 60)

    print("\n--- 10 steps of mind wandering ---")
    thoughts = dmn.mind_wander(duration_steps=10)

    for i, thought in enumerate(thoughts):
        print(f"\nStep {i + 1}:")
        print(f"  Visited: {thought.concepts}")
        print(f"  Novelty: {thought.novelty_score:.3f} | Coherence: {thought.coherence_score:.3f}")

    # Find distant associations (key for creativity)
    print("\n" + "=" * 60)
    print("DISTANT ASSOCIATION FINDING")
    print("Key mechanism for creative insight")
    print("=" * 60)

    start_concepts = ["ocean", "music", "forest"]

    for start in start_concepts:
        print(f"\n--- Distant associations from '{start}' ---")
        distant = dmn.find_distant_associations(start, min_distance=2, max_results=5)

        if distant:
            for concept, strength, path in distant:
                print(f"  -> {concept} (strength: {strength:.3f})")
                print(f"     Path: {' -> '.join(path)}")
        else:
            print("  No distant associations found")

    # Statistics
    print("\n" + "=" * 60)
    print("NETWORK STATISTICS")
    print("=" * 60)

    novelties = [t.novelty_score for t in thoughts]
    coherences = [t.coherence_score for t in thoughts]

    print("\nMind wandering statistics:")
    print(f"  Average novelty: {sum(novelties) / len(novelties):.3f}")
    print(f"  Average coherence: {sum(coherences) / len(coherences):.3f}")
    print(f"  Max novelty: {max(novelties):.3f}")
    print(f"  Min novelty: {min(novelties):.3f}")

    print("\n" + "=" * 60)
    print("KEY INSIGHT: Mind wandering enables exploration of semantic space,")
    print("finding unexpected connections that form the basis of creativity.")
    print("=" * 60)


if __name__ == "__main__":
    main()
