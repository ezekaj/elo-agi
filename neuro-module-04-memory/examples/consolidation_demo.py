#!/usr/bin/env python3
"""
Demo: Sleep Replay Consolidation

Demonstrates:
- Hippocampal replay during "sleep"
- Transfer from working memory to long-term memory
- Semantic extraction from multiple episodes
"""

import sys
sys.path.insert(0, '..')

import numpy as np
from src import (
    MemoryController,
    EpisodicMemory,
    SemanticMemory
)
from src.memory_processes import MemoryConsolidator


def demo_consolidation():
    """Demonstrate sleep-based consolidation"""
    print("=" * 60)
    print("SLEEP CONSOLIDATION DEMO")
    print("=" * 60)

    # Create memory systems
    mc = MemoryController()

    current_time = 0.0
    mc.set_time_function(lambda: current_time)

    # 1. DAYTIME: ENCODING EXPERIENCES
    print("\n1. DAYTIME: Encoding Experiences")
    print("-" * 40)

    # Simulate a day's worth of experiences
    experiences = [
        {"event": "breakfast", "food": "eggs", "location": "home", "valence": 0.3},
        {"event": "meeting", "topic": "project", "location": "office", "valence": -0.2},
        {"event": "lunch", "food": "sandwich", "location": "cafe", "valence": 0.5},
        {"event": "coffee", "food": "espresso", "location": "cafe", "valence": 0.6},
        {"event": "presentation", "topic": "results", "location": "office", "valence": 0.4},
        {"event": "dinner", "food": "pasta", "location": "restaurant", "valence": 0.7},
    ]

    print("Encoding day's experiences:")
    for exp in experiences:
        current_time += 3600  # 1 hour per event
        episode = mc.encode_experience(
            experience=exp["event"],
            context={
                "food": exp.get("food"),
                "topic": exp.get("topic"),
                "location": exp["location"],
                "time": current_time
            },
            emotional_valence=exp["valence"]
        )
        print(f"  - {exp['event']} at {exp['location']}")

    print(f"\nTotal episodes: {len(mc.episodic)}")
    print(f"Working memory items: {len(mc.working)}")

    # 2. SHOW MEMORY STATE BEFORE SLEEP
    print("\n2. MEMORY STATE BEFORE SLEEP")
    print("-" * 40)

    # Check episode strengths
    recent = mc.episodic.get_recent(n=6)
    print("Episode strengths:")
    for ep in recent:
        print(f"  - {ep.content}: strength={ep.strength:.2f}")

    # 3. SLEEP: CONSOLIDATION
    print("\n3. SLEEP: Running Consolidation Cycle")
    print("-" * 40)

    # Queue working memory items for consolidation
    wm_contents = mc.working.get_contents()
    print(f"Items queued from working memory: {len(wm_contents)}")

    # Run multiple replay cycles (simulating sleep stages)
    for cycle in range(5):
        # Queue recent episodes
        for ep in mc.episodic.get_recent(n=6):
            mc.consolidator.queue_for_consolidation(ep)

        # Run replay
        consolidated = mc.consolidator.replay_cycle(iterations=3)
        print(f"  Sleep cycle {cycle + 1}: replayed {consolidated} items")

    # 4. POST-SLEEP: CHECK STRENGTHENING
    print("\n4. POST-SLEEP: Memory Strengthening")
    print("-" * 40)

    print("Episode strengths after consolidation:")
    strongest = mc.episodic.get_strongest(n=6)
    for ep in strongest:
        print(f"  - {ep.content}: strength={ep.strength:.2f}")

    # 5. SEMANTIC EXTRACTION
    print("\n5. SEMANTIC EXTRACTION from Episodes")
    print("-" * 40)

    # Manually extract patterns (simplified)
    # In real system, this would happen automatically during consolidation

    # Count location patterns
    location_counts = {}
    for ep in mc.episodic._episodes:
        loc = ep.context.get("location")
        if loc:
            location_counts[loc] = location_counts.get(loc, 0) + 1

    print("Extracted location patterns:")
    for loc, count in sorted(location_counts.items(), key=lambda x: -x[1]):
        print(f"  - {loc}: {count} visits")

        # Create semantic concept for frequent locations
        if count >= 2:
            mc.learn_concept(
                name=f"familiar_{loc}",
                features={"visits": float(count), "familiar": 1.0},
                relations=[("is-a", "location", 1.0)]
            )

    print(f"\nSemantic concepts created: {len(mc.semantic)}")

    # 6. TEST RETRIEVAL AFTER CONSOLIDATION
    print("\n6. TEST RETRIEVAL AFTER CONSOLIDATION")
    print("-" * 40)

    # Episodic retrieval
    cafe_memories = mc.recall_context({"location": "cafe"})
    print(f"Memories at cafe: {len(cafe_memories)}")
    for mem in cafe_memories:
        print(f"  - {mem.content}")

    # Emotional retrieval (positive memories)
    positive = mc.recall_emotion(0.5, 1.0)
    print(f"\nPositive memories (valence > 0.5): {len(positive)}")
    for mem in positive:
        print(f"  - {mem.content} (valence: {mem.emotional_valence:.2f})")

    # 7. SIMULATE FORGETTING
    print("\n7. FORGETTING (Passage of Time)")
    print("-" * 40)

    # Advance time significantly
    current_time += 86400 * 7  # 1 week later

    # Apply decay to all memories
    weak_memories = []
    for ep in mc.episodic._episodes:
        # Weaker memories decay more
        if ep.strength < 0.5:
            mc.forgetter.decay(ep, rate=0.2)
            if ep.strength < 0.3:
                weak_memories.append(ep)

    print(f"Weakened memories after 1 week: {len(weak_memories)}")

    # Show final state
    print("\nFinal episode strengths:")
    for ep in mc.episodic.get_strongest(n=6):
        print(f"  - {ep.content}: strength={ep.strength:.2f}")

    # 8. INTERLEAVED REPLAY BENEFIT
    print("\n8. INTERLEAVED REPLAY (Better Consolidation)")
    print("-" * 40)

    # Create fresh memories for comparison
    regular_episodes = []
    interleaved_episodes = []

    for i in range(5):
        ep1 = mc.episodic.encode(f"regular_{i}", context={"type": "regular"})
        ep2 = mc.episodic.encode(f"interleaved_{i}", context={"type": "interleaved"})
        ep1.strength = 0.3
        ep2.strength = 0.3
        regular_episodes.append(ep1)
        interleaved_episodes.append(ep2)

    # Regular replay (in order)
    for ep in regular_episodes:
        mc.consolidator.queue_for_consolidation(ep)
    mc.consolidator.replay_cycle(iterations=2)

    # Interleaved replay
    mc.consolidator.interleave_replay(interleaved_episodes)

    # Compare final strengths
    reg_avg = np.mean([ep.strength for ep in regular_episodes])
    int_avg = np.mean([ep.strength for ep in interleaved_episodes])

    print(f"Average strength after regular replay: {reg_avg:.2f}")
    print(f"Average strength after interleaved replay: {int_avg:.2f}")

    if int_avg >= reg_avg:
        print("Interleaved replay shows equal or better consolidation!")

    print("\n" + "=" * 60)
    print("CONSOLIDATION DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    demo_consolidation()
