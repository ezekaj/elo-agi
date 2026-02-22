#!/usr/bin/env python3
"""
Demo: Conceptual Space Mapping (2025 Discovery)

Demonstrates the key 2025 finding that hippocampal spatial cells
also encode abstract concepts:
- Concept "place cells" for ideas
- Grid-like coding for conceptual distance
- Social distance mapping
- Analogy completion via vector arithmetic
"""

import sys

sys.path.insert(0, "..")

import numpy as np
from src import ConceptCell, SocialDistanceGrid, ConceptualMap


def demo_conceptual_space():
    print("=" * 60)
    print("CONCEPTUAL SPACE DEMO (2025 Discovery)")
    print("=" * 60)

    # 1. CONCEPT CELLS
    print("\n1. CONCEPT CELLS (Like Place Cells for Ideas)")
    print("-" * 40)

    # Create concept cell for "democracy"
    democracy_cell = ConceptCell(
        concept_center=np.array([0.8, 0.7, 0.6]), concept_radius=0.3, associated_concept="democracy"
    )

    # Test activation for different concepts
    concepts = [
        ("democracy", np.array([0.8, 0.7, 0.6])),  # Same position
        ("freedom", np.array([0.7, 0.8, 0.6])),  # Similar
        ("autocracy", np.array([0.2, 0.3, 0.4])),  # Opposite
    ]

    print(f"Concept cell tuned to: {democracy_cell.associated_concept}")
    print(f"Cell center: {democracy_cell.concept_center}")
    print("\nActivation for different concepts:")
    for name, pos in concepts:
        activation = democracy_cell.compute_activation(pos)
        print(f"  {name}: activation = {activation:.3f}")

    # 2. CONCEPTUAL MAP
    print("\n2. CONCEPTUAL MAP (Embedding Concepts)")
    print("-" * 40)

    cmap = ConceptualMap(concept_dimensions=5, n_concept_cells=50, random_seed=42)

    # Embed animal concepts
    animals = {
        "dog": np.array([0.8, 0.2, 0.5, 0.7, 0.3]),
        "cat": np.array([0.7, 0.3, 0.6, 0.5, 0.4]),
        "wolf": np.array([0.9, 0.1, 0.4, 0.8, 0.2]),
        "lion": np.array([0.95, 0.0, 0.3, 0.9, 0.1]),
    }

    # Embed vehicle concepts
    vehicles = {
        "car": np.array([0.1, 0.9, 0.8, 0.2, 0.7]),
        "bicycle": np.array([0.2, 0.8, 0.9, 0.3, 0.8]),
        "truck": np.array([0.0, 0.95, 0.7, 0.1, 0.6]),
    }

    print("Embedding animals:")
    for name, features in animals.items():
        pos = cmap.embed_concept(name, features, category="animal")
        print(f"  {name} → position in concept space")

    print("\nEmbedding vehicles:")
    for name, features in vehicles.items():
        pos = cmap.embed_concept(name, features, category="vehicle")
        print(f"  {name} → position in concept space")

    # 3. CONCEPTUAL DISTANCE
    print("\n3. CONCEPTUAL DISTANCE (Grid-Like Coding)")
    print("-" * 40)

    pairs = [
        ("dog", "cat"),
        ("dog", "wolf"),
        ("dog", "car"),
        ("cat", "lion"),
        ("car", "truck"),
    ]

    print("Distances between concepts:")
    for a, b in pairs:
        dist = cmap.conceptual_distance(a, b)
        print(f"  {a} ↔ {b}: {dist:.3f}")

    # 4. FIND SIMILAR CONCEPTS
    print("\n4. FINDING SIMILAR CONCEPTS")
    print("-" * 40)

    target = "dog"
    similar = cmap.find_similar(target, n=3)

    print(f"Concepts most similar to '{target}':")
    for name, dist in similar:
        print(f"  {name}: distance = {dist:.3f}")

    # 5. ANALOGY COMPLETION
    print("\n5. ANALOGY COMPLETION (Vector Arithmetic)")
    print("-" * 40)

    # Add more concepts for analogy
    cmap.embed_concept("puppy", np.array([0.6, 0.2, 0.5, 0.5, 0.3]))
    cmap.embed_concept("kitten", np.array([0.5, 0.3, 0.6, 0.3, 0.4]))

    print("Analogy: puppy : dog :: kitten : ?")
    result = cmap.compute_analogy("puppy", "dog", "kitten")
    if result:
        answer, dist = result
        print(f"Answer: {answer} (distance to target: {dist:.3f})")
    else:
        print("Could not compute analogy")

    # Another analogy with animals
    print("\nAnalogy: dog : wolf :: cat : ?")
    result = cmap.compute_analogy("dog", "wolf", "cat")
    if result:
        answer, dist = result
        print(f"Answer: {answer} (distance to target: {dist:.3f})")

    # 6. NAVIGATE THROUGH CONCEPT SPACE
    print("\n6. NAVIGATING CONCEPT SPACE")
    print("-" * 40)

    start = "dog"
    goal = "car"

    path = cmap.navigate_concepts(start, goal, steps=5)
    print(f"Path from '{start}' to '{goal}':")
    print(f"  Start: {cmap.get_concept_position(start)[:3]}...")

    for i, pos in enumerate(path[1:-1], 1):
        print(f"  Step {i}: {pos[:3]}...")

    print(f"  Goal: {cmap.get_concept_position(goal)[:3]}...")

    # 7. SOCIAL DISTANCE GRID
    print("\n7. SOCIAL DISTANCE MAPPING")
    print("-" * 40)

    social = SocialDistanceGrid(dimensions=2, dimension_names=["power", "affiliation"])

    # Add people with social positions
    people = {
        "CEO": np.array([0.95, 0.4]),
        "Manager": np.array([0.7, 0.6]),
        "Employee1": np.array([0.4, 0.7]),
        "Employee2": np.array([0.35, 0.8]),
        "Intern": np.array([0.2, 0.5]),
    }

    print("Social positions (power, affiliation):")
    for name, pos in people.items():
        social.set_social_position(name, pos)
        print(f"  {name}: power={pos[0]:.2f}, affiliation={pos[1]:.2f}")

    # Social distances
    print("\nSocial distances:")
    for name1 in ["CEO", "Manager"]:
        for name2 in ["Employee1", "Intern"]:
            dist = social.compute_social_distance(name1, name2)
            print(f"  {name1} ↔ {name2}: {dist:.3f}")

    # Find socially similar
    print("\nPeople socially similar to Employee1:")
    similar = social.find_socially_similar("Employee1", threshold=0.3)
    for name, dist in similar:
        print(f"  {name}: distance = {dist:.3f}")

    # 8. CONCEPT CELL ACTIVATIONS
    print("\n8. CONCEPT CELL POPULATION ACTIVITY")
    print("-" * 40)

    for concept in ["dog", "car"]:
        activations = cmap.get_concept_activations(concept)
        active_count = np.sum(activations > 0.5)
        max_activation = np.max(activations)

        print(f"'{concept}':")
        print(f"  Active concept cells: {active_count}")
        print(f"  Max activation: {max_activation:.3f}")

    # 9. PHYSICAL TO CONCEPTUAL MAPPING
    print("\n9. PHYSICAL → CONCEPTUAL SPACE TRANSFER")
    print("-" * 40)

    # This demonstrates the 2025 finding that the same neural
    # machinery handles both physical and conceptual space

    spatial_pos = np.array([0.3, 0.7])
    conceptual = cmap.map_physical_to_conceptual(spatial_pos, scaling=1.0)

    print(f"Physical position: {spatial_pos}")
    print(f"Mapped to concept space: {conceptual[:3]}...")
    print("\nThis demonstrates that the brain uses the SAME")
    print("neural circuits for physical AND conceptual navigation!")

    # 10. SUMMARY
    print("\n" + "=" * 60)
    print("KEY 2025 DISCOVERIES DEMONSTRATED:")
    print("-" * 40)
    print("1. Concept cells fire for abstract ideas (like place cells)")
    print("2. Conceptual distance uses grid-like coding")
    print("3. Social relationships encoded spatially")
    print("4. Analogies solved via vector arithmetic")
    print("5. Same neural machinery for physical & conceptual space")
    print("=" * 60)


if __name__ == "__main__":
    demo_conceptual_space()
