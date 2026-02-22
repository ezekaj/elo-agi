"""
Creative Session Demo - Full Creative Process

Demonstrates the complete creative cognition system:
- DMN for idea generation
- ECN for evaluation and refinement
- Salience Network for dynamic switching
- Network reconfiguration during creativity

Key insight: "Generating creative ideas led to significantly higher
network reconfiguration than generating non-creative ideas"
"""

import sys

sys.path.insert(0, ".")

from src.creative_process import CreativeProcess


def main():
    print("=" * 60)
    print("CREATIVE SESSION DEMONSTRATION")
    print("Full Creative Cognition System")
    print("=" * 60)

    # Create creative process
    cp = CreativeProcess()

    # Setup knowledge base - technology and nature domain
    print("\n--- Setting up Knowledge Base ---")

    concepts = [
        # Technology
        (
            "technology",
            "Digital tools and systems",
            {"artificial": 0.9, "logical": 0.8, "efficient": 0.7},
        ),
        (
            "algorithm",
            "Step-by-step procedure",
            {"logical": 1.0, "precise": 0.9, "artificial": 0.7},
        ),
        ("network", "Connected system", {"connected": 0.9, "complex": 0.8, "dynamic": 0.7}),
        ("data", "Information units", {"structured": 0.8, "objective": 0.7, "digital": 0.9}),
        ("interface", "Connection point", {"interaction": 0.9, "visual": 0.6, "functional": 0.8}),
        # Nature
        ("nature", "Natural world", {"organic": 1.0, "complex": 0.9, "beautiful": 0.8}),
        ("ecosystem", "Living system", {"connected": 0.9, "complex": 1.0, "organic": 0.9}),
        ("growth", "Organic development", {"change": 0.8, "organic": 0.9, "positive": 0.7}),
        ("adaptation", "Environmental response", {"change": 0.9, "survival": 0.8, "organic": 0.7}),
        ("harmony", "Balanced coexistence", {"balance": 0.9, "peaceful": 0.8, "aesthetic": 0.7}),
        # Bridging concepts
        (
            "emergence",
            "Complex from simple",
            {"complexity": 0.9, "surprising": 0.8, "natural": 0.7},
        ),
        (
            "pattern",
            "Recurring structure",
            {"structure": 0.9, "recognition": 0.8, "universal": 0.7},
        ),
        ("flow", "Continuous movement", {"dynamic": 0.9, "smooth": 0.8, "natural": 0.7}),
        ("connection", "Links between entities", {"relationship": 0.9, "network": 0.8}),
        ("beauty", "Aesthetic quality", {"aesthetic": 1.0, "subjective": 0.7, "emotional": 0.8}),
    ]

    cp.setup_knowledge(concepts)
    print(f"  Loaded {len(concepts)} concepts")

    # Create associations
    associations = [
        # Within technology
        ("technology", "algorithm", 0.9, "uses"),
        ("technology", "network", 0.8, "forms"),
        ("technology", "data", 0.9, "processes"),
        ("algorithm", "data", 0.8, "processes"),
        ("network", "interface", 0.7, "has"),
        # Within nature
        ("nature", "ecosystem", 0.95, "contains"),
        ("nature", "growth", 0.8, "exhibits"),
        ("ecosystem", "adaptation", 0.9, "requires"),
        ("ecosystem", "harmony", 0.7, "achieves"),
        ("growth", "adaptation", 0.6, "enables"),
        # Cross-domain (creative connections)
        ("network", "ecosystem", 0.5, "analogous"),  # Key creative link
        ("algorithm", "adaptation", 0.4, "analogous"),  # Computational nature
        ("data", "pattern", 0.7, "contains"),
        ("pattern", "nature", 0.6, "found_in"),
        ("flow", "network", 0.5, "metaphor"),
        ("flow", "nature", 0.8, "characteristic"),
        ("emergence", "ecosystem", 0.8, "observed_in"),
        ("emergence", "network", 0.6, "observed_in"),
        # Aesthetic connections
        ("harmony", "beauty", 0.8, "creates"),
        ("pattern", "beauty", 0.6, "evokes"),
        ("nature", "beauty", 0.9, "embodies"),
        ("technology", "beauty", 0.3, "can_achieve"),  # Weaker but possible
    ]

    cp.create_associations(associations)
    print(f"  Created {len(associations)} associations")

    # Set creative goal
    print("\n--- Setting Creative Goal ---")
    goal_description = "Design technology that embodies natural principles"
    cp.set_creative_goal(
        "biomimetic_tech", goal_description, constraints=["must be feasible", "should be elegant"]
    )
    print(f"  Goal: {goal_description}")

    # Show initial network state
    print("\n" + "=" * 60)
    print("PHASE 1: IDEA GENERATION (DMN Active)")
    print("=" * 60)

    print(f"\nInitial network state: {cp.salience.current_state.value}")
    print(
        f"Initial reconfiguration: {cp.salience.get_network_activity().reconfiguration_level:.3f}"
    )

    # Generate ideas
    print("\n--- Generating Ideas ---")
    ideas = cp.generate_ideas(
        num_ideas=5, seed_concepts=["technology", "nature", "ecosystem"], use_imagery=True
    )

    for idea in ideas:
        print(f"\n  Idea: {idea.id}")
        print(f"    Content: {idea.content}")
        print(f"    Source concepts: {idea.source_concepts}")
        print(f"    Novelty: {idea.novelty:.3f}")
        print(f"    Coherence: {idea.coherence:.3f}")
        if idea.imagery:
            print(f"    Imagery vividness: {idea.imagery.overall_vividness:.3f}")

    # Evaluate ideas
    print("\n" + "=" * 60)
    print("PHASE 2: IDEA EVALUATION (ECN Active)")
    print("=" * 60)

    print(f"\nNetwork state after generation: {cp.salience.current_state.value}")
    print(f"Reconfiguration level: {cp.salience.get_network_activity().reconfiguration_level:.3f}")

    print("\n--- Evaluating Ideas ---")
    evaluations = cp.evaluate_ideas()

    for eval_result in evaluations:
        print(f"\n  {eval_result.idea_id}:")
        print(f"    Overall score: {eval_result.overall_score:.3f}")
        print(f"    Recommendation: {eval_result.recommendation}")
        print(f"    Criteria scores: {eval_result.scores}")

    # Refine ideas
    print("\n" + "=" * 60)
    print("PHASE 3: IDEA REFINEMENT")
    print("=" * 60)

    print("\n--- Refining Ideas that Need Improvement ---")
    refined = cp.refine_ideas(target_score=0.7)

    if refined:
        for idea in refined:
            print(f"\n  Refined: {idea.id}")
            print(f"    Content: {idea.content}")
            print(f"    Novelty: {idea.novelty:.3f}")
            print(f"    Coherence: {idea.coherence:.3f}")
    else:
        print("  No ideas required refinement")

    # Run full creative session
    print("\n" + "=" * 60)
    print("PHASE 4: AUTONOMOUS CREATIVE SESSION")
    print("Dynamic DMN-ECN Switching")
    print("=" * 60)

    print("\n--- Running 5-second creative session ---")
    output = cp.creative_session(
        goal="Create biomimetic technology concept", duration_seconds=5.0, target_good_ideas=3
    )

    print("\n  Session Results:")
    print(f"    Duration: {output.process_duration:.2f} seconds")
    print(f"    Ideas generated: {output.total_generated}")
    print(f"    Ideas evaluated: {output.total_evaluated}")
    print(f"    Network reconfigurations: {output.network_reconfigurations}")
    print(f"    Creativity score: {output.creativity_score:.3f}")

    print("\n  Best Ideas:")
    for i, idea in enumerate(output.best_ideas):
        print(f"\n    #{i + 1}: {idea.id}")
        print(f"        Content: {idea.content}")
        if idea.evaluation:
            print(f"        Score: {idea.evaluation.overall_score:.3f}")

    # Mind wandering exploration
    print("\n" + "=" * 60)
    print("PHASE 5: MIND WANDERING FOR INSPIRATION")
    print("=" * 60)

    print("\n--- Free exploration for unexpected insights ---")
    wandering_ideas = cp.mind_wander_for_ideas(duration_steps=10)

    print(f"\n  Found {len(wandering_ideas)} novel ideas from mind wandering:")
    for idea in wandering_ideas[:3]:  # Show top 3
        print(f"\n    {idea.id}:")
        print(f"      Content: {idea.content}")
        print(f"      Novelty: {idea.novelty:.3f}")

    # Find distant connections
    print("\n" + "=" * 60)
    print("PHASE 6: DISTANT ASSOCIATION DISCOVERY")
    print("=" * 60)

    print("\n--- Finding creative connections between distant concepts ---")
    distant = cp.find_distant_connections("technology", max_results=5)

    if distant:
        print("\n  Distant connections from 'technology':")
        for concept, strength, path in distant:
            print(f"\n    -> {concept}")
            print(f"       Strength: {strength:.3f}")
            print(f"       Path: {' -> '.join(path)}")

    # Final statistics
    print("\n" + "=" * 60)
    print("CREATIVE PROCESS STATISTICS")
    print("=" * 60)

    stats = cp.get_creative_statistics()
    print(f"\n  Total ideas generated: {stats['total_ideas']}")
    print(f"  Ideas evaluated: {stats['evaluated_ideas']}")
    print(f"  Average novelty: {stats['avg_novelty']:.3f}")
    print(f"  Average quality: {stats['avg_quality']:.3f}")
    print(f"  Network switches: {stats['network_switches']}")
    print(f"  Final reconfiguration level: {stats['reconfiguration_level']:.3f}")

    print("\n" + "=" * 60)
    print("KEY INSIGHT: Creative cognition involves dynamic switching")
    print("between generative (DMN) and evaluative (ECN) modes,")
    print("with higher network reconfiguration for creative ideas.")
    print("=" * 60)


if __name__ == "__main__":
    main()
