#!/usr/bin/env python3
"""
Demo: Full sensory → Working Memory → Long-Term Memory pipeline

Demonstrates:
- Sensory buffer timing (250ms iconic, 3-4s echoic)
- Working memory capacity limits (7±2 items)
- Consolidation to long-term storage
"""

import sys

sys.path.insert(0, "..")

import numpy as np
from src import MemoryController


def demo_memory_flow():
    """Demonstrate the complete memory flow pipeline"""
    print("=" * 60)
    print("MEMORY FLOW DEMO: Sensory → Working → Long-Term")
    print("=" * 60)

    # Create memory controller
    mc = MemoryController(wm_capacity=7)

    # Simulated time for controlled demo
    current_time = 0.0
    mc.set_time_function(lambda: current_time)

    # 1. SENSORY MEMORY
    print("\n1. SENSORY MEMORY")
    print("-" * 40)

    # Capture visual input
    visual_data = np.random.rand(10, 10)  # Simulated image
    mc.process_visual(visual_data)
    print(f"Captured visual data (10x10 array)")
    print(f"Iconic buffer available: {mc.iconic.is_available()}")

    # Check after 100ms
    current_time = 0.1
    _, strength = mc.iconic.read_with_strength()
    print(f"After 100ms - strength: {strength:.2f}")

    # Check after 300ms (should be very weak)
    current_time = 0.3
    _, strength = mc.iconic.read_with_strength()
    print(f"After 300ms - strength: {strength:.2f}")

    # Reset for working memory demo
    current_time = 0.0

    # 2. ATTENTION AND WORKING MEMORY
    print("\n2. WORKING MEMORY (7±2 capacity)")
    print("-" * 40)

    # Process new visual input and attend to it
    mc.process_visual(np.random.rand(5, 5))
    attended = mc.attend("visual")
    print(f"Attended to visual input: {attended is not None}")

    # Try to store more than capacity
    print("\nStoring items in working memory...")
    for i in range(10):
        mc.store_in_working_memory(f"item_{i}")

    print(f"Tried to store 10 items")
    print(f"Working memory load: {mc.working.get_load():.2f}")
    print(f"Actual items: {len(mc.working)}")

    # Show contents
    contents = mc.working.get_contents()
    print(f"Items in WM: {[c[0] for c in contents[:5]]}...")

    # 3. CHUNKING
    print("\n3. CHUNKING (increasing effective capacity)")
    print("-" * 40)

    mc.clear_working_memory()

    # Store items
    mc.store_in_working_memory("A")
    mc.store_in_working_memory("B")
    mc.store_in_working_memory("C")
    print(f"After storing A, B, C: {len(mc.working)} items")

    # Chunk them
    mc.working.chunk(["A", "B", "C"], label="ABC")
    print(f"After chunking A+B+C: {len(mc.working)} item (1 chunk)")

    # Now can store more
    for i in range(6):
        mc.store_in_working_memory(f"extra_{i}")
    print(f"Added 6 more items: {len(mc.working)} total")

    # 4. DECAY AND REHEARSAL
    print("\n4. DECAY AND REHEARSAL")
    print("-" * 40)

    mc.clear_working_memory()
    mc.store_in_working_memory("important_item")
    mc.store_in_working_memory("forgotten_item")

    print(f"Initial items: {len(mc.working)}")

    # Rehearse only one item
    current_time = 15.0  # 15 seconds later
    mc.working.rehearse("important_item")

    # Advance time and decay
    current_time = 35.0  # 35 seconds total
    mc.tick(35.0)

    print(f"After 35s (one rehearsed): {len(mc.working)} items")
    print(f"Important item found: {mc.working.retrieve('important_item') is not None}")

    # 5. LONG-TERM MEMORY ENCODING
    print("\n5. LONG-TERM MEMORY ENCODING")
    print("-" * 40)

    # Encode episodic memory
    episode = mc.encode_experience(
        experience="Had lunch with friend",
        context={"location": "cafe", "person": "John"},
        emotional_valence=0.7,
    )
    print(f"Encoded episodic memory: {episode.content}")

    # Encode semantic memory
    concept = mc.learn_concept(
        name="cafe",
        features={"serves_food": 1.0, "serves_drinks": 1.0},
        relations=[("is-a", "restaurant", 1.0)],
    )
    print(f"Learned concept: {concept.name}")

    # Encode procedural memory
    skill = mc.learn_skill(
        name="order_coffee",
        trigger_features={"location": "cafe", "want": "coffee"},
        action_names=["approach_counter", "order", "pay", "receive"],
    )
    print(f"Learned skill: {skill.name}")

    # 6. RETRIEVAL
    print("\n6. RETRIEVAL")
    print("-" * 40)

    # Episodic retrieval
    results = mc.recall("lunch")
    print(f"Recall 'lunch': {results.content if results else None}")

    # Context retrieval
    context_results = mc.recall_context({"location": "cafe"})
    print(f"Recall context 'cafe': {len(context_results)} episodes")

    # Semantic retrieval
    cafe_concept = mc.recall("cafe", memory_types=["semantic"])
    print(f"Recall concept 'cafe': {cafe_concept.name if cafe_concept else None}")

    # Spreading activation
    print("\nSpreading activation from 'cafe':")
    activations = mc.spread_concepts("cafe", depth=2)
    for name, level in sorted(activations.items(), key=lambda x: -x[1]):
        print(f"  {name}: {level:.2f}")

    # 7. CONSOLIDATION
    print("\n7. CONSOLIDATION (simulated sleep)")
    print("-" * 40)

    # Add items to working memory
    mc.store_in_working_memory({"event": "meeting", "time": "morning"})
    mc.store_in_working_memory({"event": "presentation", "time": "afternoon"})

    print(f"Items in WM before consolidation: {len(mc.working)}")

    # Consolidate (simulate sleep)
    consolidated = mc.consolidate()
    print(f"Consolidated {consolidated} items to long-term memory")

    # 8. STATE SUMMARY
    print("\n8. MEMORY STATE SUMMARY")
    print("-" * 40)

    state = mc.get_state()
    for key, value in state.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    demo_memory_flow()
